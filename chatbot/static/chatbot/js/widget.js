(function() {
    'use strict';

    let currentSessionId = null;
    let isExpanded = false;
    let isStreaming = false;
    let scrollPending = false;

    const widget = document.getElementById('chatWidget');
    const toggleBtn = document.getElementById('chatWidgetToggle');
    const minimizeBtn = document.getElementById('chatWidgetMinimize');
    const widgetWindow = document.getElementById('chatWidgetWindow');
    const messagesArea = document.getElementById('widgetChatMessages');
    const messageForm = document.getElementById('widgetMessageForm');
    const messageInput = document.getElementById('widgetMessageInput');
    const sendBtn = document.getElementById('widgetSendBtn');
    const streamToggle = document.getElementById('widgetStreamToggle');
    const fileInput = document.getElementById('widgetFileInput');
    const chatIcon = toggleBtn.querySelector('.chat-icon');
    const closeIcon = toggleBtn.querySelector('.close-icon');

    function init() {
        setupEventListeners();
        initializeSession();
    }

    function setupEventListeners() {
        toggleBtn.addEventListener('click', toggleWidget);
        minimizeBtn.addEventListener('click', toggleWidget);
        messageForm.addEventListener('submit', handleSendMessage);

        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 100) + 'px';
        });

        messageInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                messageForm.dispatchEvent(new Event('submit'));
            }
        });

        document.querySelectorAll('.quick-action-btn').forEach(btn => btn.addEventListener('click', handleQuickAction));
        document.querySelectorAll('.suggestion-btn').forEach(btn => btn.addEventListener('click', function() {
            messageInput.value = this.dataset.question;
            messageForm.dispatchEvent(new Event('submit'));
        }));

        fileInput.addEventListener('change', handleFileUpload);

        const openFullPageBtn = document.getElementById('openFullPageBtn');
        if (openFullPageBtn) openFullPageBtn.addEventListener('click', openFullPage);
    }

    function toggleWidget() {
        isExpanded = !isExpanded;
        if (isExpanded) {
            widgetWindow.classList.remove('hidden');
            chatIcon.classList.add('hidden');
            closeIcon.classList.remove('hidden');
            messageInput.focus();
        } else {
            widgetWindow.classList.add('hidden');
            chatIcon.classList.remove('hidden');
            closeIcon.classList.add('hidden');
        }
    }

    async function initializeSession() {
        try {
            const response = await fetch('/chatbot/api/sessions/?page_size=1');
            const data = await response.json();
            if (data.results && data.results.length > 0) {
                currentSessionId = data.results[0].id;
                await loadMessages();
            } else {
                await createNewSession();
            }
        } catch (error) {
            console.error('Error initializing session:', error);
        }
    }

    async function createNewSession() {
        try {
            const response = await fetch('/chatbot/api/sessions/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                body: JSON.stringify({ title: 'Quick Chat ' + new Date().toLocaleString() })
            });
            const session = await response.json();
            currentSessionId = session.id;
            clearMessages();
        } catch (error) {
            console.error('Error creating session:', error);
            showError('Failed to start chat session');
        }
    }

    async function loadMessages() {
        if (!currentSessionId) return;
        try {
            const response = await fetch(`/chatbot/api/sessions/${currentSessionId}/messages/`);
            const data = await response.json();
            const messages = data.results || data;
            clearMessages();
            messages.forEach(msg => addMessageToUI(msg.role, msg.content, msg.sources));
            scrollToBottom();
        } catch (error) { console.error('Error loading messages:', error); }
    }

    async function handleSendMessage(e) {
        e.preventDefault();
        if (!currentSessionId) await createNewSession();
        const message = messageInput.value.trim();
        if (!message) return;

        addMessageToUI('user', message);
        messageInput.value = '';
        messageInput.style.height = 'auto';
        setInputState(false);

        if (streamToggle.checked) await sendMessageWithStreaming(message);
        else await sendMessageWithoutStreaming(message);

        setInputState(true);
        messageInput.focus();
    }

    async function sendMessageWithoutStreaming(message) {
        showTypingIndicator();
        try {
            const response = await fetch(`/chatbot/api/sessions/${currentSessionId}/send_message/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                body: JSON.stringify({ message })
            });
            const data = await response.json();
            removeTypingIndicator();
            if (data.assistant_message) addMessageToUI('assistant', data.assistant_message.content, data.assistant_message.sources);
        } catch (error) {
            console.error('Error sending message:', error);
            removeTypingIndicator();
            showError('Failed to get response. Please try again.');
        }
    }

    async function sendMessageWithStreaming(message) {
        isStreaming = true;
        const messageId = 'stream-' + Date.now();
        const messageBubble = createStreamingBubble(messageId);
        messagesArea.appendChild(messageBubble);
        scrollToBottom();
        const contentDiv = messageBubble.querySelector('.message-content');
        messageBubble.classList.add('streaming');
        let accumulatedContent = '';
        let sources = [];
        let scrollPending = false;
        
        // Debounced markdown rendering - render every 200ms or after 10 tokens
        let renderTimeout = null;
        let tokenCount = 0;
        const renderMarkdown = () => {
            if (accumulatedContent) {
                // Remove cursor temporarily for rendering
                const cursor = messageBubble.querySelector('.streaming-cursor');
                const cursorText = cursor ? cursor.textContent : '';
                if (cursor) cursor.remove();
                
                // Render markdown
                contentDiv.innerHTML = parseMarkdown(accumulatedContent);
                
                // Re-add cursor at the end
                if (isStreaming) {
                    const newCursor = document.createElement('span');
                    newCursor.className = 'streaming-cursor';
                    contentDiv.appendChild(newCursor);
                }
                
                if (!scrollPending) {
                    scrollPending = true;
                    requestAnimationFrame(() => {
                        scrollToBottom();
                        scrollPending = false;
                    });
                }
            }
        };
        
        const scheduleMarkdownRender = () => {
            // Clear existing timeout
            if (renderTimeout) clearTimeout(renderTimeout);
            
            // Very short debounce (50ms) for near-instant rendering while still batching DOM updates
            renderTimeout = setTimeout(renderMarkdown, 50);
        };

        try {
            const response = await fetch(`/chatbot/api/sessions/${currentSessionId}/send_message_stream/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                body: JSON.stringify({ message, stream: true })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                // Decode chunk immediately
                buffer += decoder.decode(value, { stream: true });
                
                // Process all complete SSE messages (separated by \n\n)
                let messageEnd;
                while ((messageEnd = buffer.indexOf('\n\n')) !== -1) {
                    const message = buffer.substring(0, messageEnd);
                    buffer = buffer.substring(messageEnd + 2);
                    
                    if (!message.trim()) continue;
                    
                    // Parse SSE message format:
                    // event: stream_token
                    // data: {"type":"token","content":"Hello"}
                    let currentEvent = null;
                    let currentData = null;
                    const lines = message.split('\n');
                    
                    for (const line of lines) {
                        const trimmed = line.trim();
                        if (trimmed.startsWith('event: ')) {
                            currentEvent = trimmed.substring(7).trim();
                        } else if (trimmed.startsWith('data: ')) {
                            try {
                                const jsonStr = trimmed.substring(6).trim();
                                if (jsonStr) {
                                    currentData = JSON.parse(jsonStr);
                                }
                            } catch (e) {
                                // Skip heartbeat and other non-JSON data
                                if (!trimmed.includes('heartbeat')) {
                                    console.debug('Failed to parse SSE data:', trimmed, e);
                                }
                            }
                        }
                    }
                    
                    // Process the parsed event and data IMMEDIATELY
                    if (currentData) {
                        if (currentData.type === 'token') {
                            const token = currentData.content || '';
                            if (token) {
                                accumulatedContent += token;
                                tokenCount++;
                                
                                // Render immediately every token for true streaming feel
                                // But debounce markdown parsing slightly
                                if (tokenCount % 2 === 0) {
                                    // Render every 2 tokens
                                    renderMarkdown();
                                } else {
                                    // Schedule render for single tokens (debounced)
                                    scheduleMarkdownRender();
                                }
                            }
                        } else if (currentData.type === 'source') {
                            if (Array.isArray(currentData.sources)) {
                                sources = currentData.sources;
                            } else {
                                sources.push(currentData);
                            }
                        } else if (currentData.type === 'complete') {
                            // Clear any pending render timeout
                            if (renderTimeout) {
                                clearTimeout(renderTimeout);
                                renderTimeout = null;
                            }
                            
                            // Final markdown render
                            const cursor = messageBubble.querySelector('.streaming-cursor');
                            if (cursor) cursor.remove();
                            messageBubble.classList.remove('streaming');
                            contentDiv.innerHTML = parseMarkdown(accumulatedContent);
                            
                            if (sources.length > 0) addSourcesToMessage(messageBubble, sources);
                            scrollToBottom();
                        } else if (currentData.type === 'error') {
                            if (renderTimeout) {
                                clearTimeout(renderTimeout);
                                renderTimeout = null;
                            }
                            const cursor = messageBubble.querySelector('.streaming-cursor');
                            if (cursor) cursor.remove();
                            messageBubble.classList.remove('streaming');
                            contentDiv.innerHTML = parseMarkdown('Error: ' + (currentData.message || 'Unknown error'));
                        }
                    }
                }
            }
            
            // Process any remaining buffer content
            if (buffer.trim()) {
                // Try to parse remaining buffer as SSE message
                let currentEvent = null;
                let currentData = null;
                const lines = buffer.split('\n');
                
                for (const line of lines) {
                    const trimmed = line.trim();
                    if (trimmed.startsWith('event: ')) {
                        currentEvent = trimmed.substring(7).trim();
                    } else if (trimmed.startsWith('data: ')) {
                        try {
                            const jsonStr = trimmed.substring(6).trim();
                            if (jsonStr) {
                                currentData = JSON.parse(jsonStr);
                            }
                        } catch (e) {
                            // Ignore parse errors for incomplete data
                        }
                    }
                }
                
                if (currentData && currentData.type === 'token') {
                    accumulatedContent += currentData.content || '';
                    renderMarkdown();
                }
            }

            // Handle any remaining buffer content
            if (buffer.trim()) {
                try {
                    if (buffer.startsWith('data: ')) {
                        const data = JSON.parse(buffer.substring(6).trim());
                        if (data.type === 'token') {
                            accumulatedContent += data.content;
                            scheduleMarkdownRender();
                        } else if (data.type === 'complete') {
                            if (renderTimeout) clearTimeout(renderTimeout);
                            const cursor = messageBubble.querySelector('.streaming-cursor');
                            if (cursor) cursor.remove();
                            messageBubble.classList.remove('streaming');
                            contentDiv.innerHTML = parseMarkdown(accumulatedContent);
                            if (sources.length > 0) addSourcesToMessage(messageBubble, sources);
                            scrollToBottom();
                        }
                    }
                } catch (e) { console.debug('Final buffer parse error:', e); }
            }
            
            // Final render if still streaming (safety net)
            if (isStreaming && accumulatedContent) {
                if (renderTimeout) clearTimeout(renderTimeout);
                renderMarkdown();
            }
        } catch (error) {
            console.error('Streaming error:', error);
            if (renderTimeout) clearTimeout(renderTimeout);
            const cursor = messageBubble.querySelector('.streaming-cursor');
            if (cursor) cursor.remove();
            messageBubble.classList.remove('streaming');
            contentDiv.innerHTML = parseMarkdown('Connection failed. Please try again.');
        }

        isStreaming = false;
    }

    function addMessageToUI(role, content, sources = []) {
        const messageBubble = document.createElement('div');
        messageBubble.className = `message-bubble ${role}`;
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = parseMarkdown(content);
        messageBubble.appendChild(messageContent);
        if (sources.length > 0) addSourcesToMessage(messageBubble, sources);

        const welcomeMsg = messagesArea.querySelector('.welcome-message');
        if (welcomeMsg) welcomeMsg.remove();

        messagesArea.appendChild(messageBubble);
        scrollToBottom();
    }

    function createStreamingBubble(id) {
        const messageBubble = document.createElement('div');
        messageBubble.id = id;
        messageBubble.className = 'message-bubble assistant';
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        const cursor = document.createElement('span');
        cursor.className = 'streaming-cursor';
        messageContent.appendChild(cursor);
        messageBubble.appendChild(messageContent);

        const welcomeMsg = messagesArea.querySelector('.welcome-message');
        if (welcomeMsg) welcomeMsg.remove();

        return messageBubble;
    }

    function addSourcesToMessage(messageBubble, sources) {
        const sourcesDiv = document.createElement('div');
        sourcesDiv.className = 'message-sources';
        const title = document.createElement('div');
        title.className = 'message-sources-title';
        title.textContent = '📚 Sources:';
        sourcesDiv.appendChild(title);

        sources.slice(0, 3).forEach(source => {
            const sourceItem = document.createElement('div');
            sourceItem.className = 'source-item';
            sourceItem.textContent = `• ${source.title || 'Document'}`;
            sourcesDiv.appendChild(sourceItem);
        });

        const messageContent = messageBubble.querySelector('.message-content');
        messageContent.appendChild(sourcesDiv);
    }

    function showTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.className = 'message-bubble assistant';
        indicator.id = 'typing-indicator';
        const content = document.createElement('div');
        content.className = 'message-content';
        const typing = document.createElement('div');
        typing.className = 'typing-indicator';
        typing.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';
        content.appendChild(typing);
        indicator.appendChild(content);
        const welcomeMsg = messagesArea.querySelector('.welcome-message');
        if (welcomeMsg) welcomeMsg.remove();
        messagesArea.appendChild(indicator);
        scrollToBottom();
    }

    function removeTypingIndicator() { const indicator = document.getElementById('typing-indicator'); if (indicator) indicator.remove(); }
    function clearMessages() { messagesArea.innerHTML = ''; }
    function setInputState(enabled) { messageInput.disabled = !enabled; sendBtn.disabled = !enabled; }
    function scrollToBottom() { messagesArea.scrollTop = messagesArea.scrollHeight; }
    
    // Initialize markdown parser once (singleton)
    let mdParser = null;
    function getMarkdownParser() {
        if (!mdParser && window.markdownit) {
            mdParser = window.markdownit({
                html: false,
                linkify: true,
                breaks: true,
                typographer: true,
            });
        }
        return mdParser;
    }
    
    function parseMarkdown(md) {
        if (!md) return '';
        
        // Use singleton parser
        const parser = getMarkdownParser();
        if (!parser) {
            // Fallback if markdown-it not loaded
            return md.replace(/\n/g, '<br>');
        }
        
        try {
            const rendered = parser.render(md);
            return window.DOMPurify ? window.DOMPurify.sanitize(rendered) : rendered;
        } catch (e) {
            console.warn('Markdown parsing error:', e);
            return md.replace(/\n/g, '<br>');
        }
    }
    function getCSRFToken() { return document.querySelector('[name=csrfmiddlewaretoken]').value; }
    function handleQuickAction(e) { const action = e.currentTarget.dataset.action; if (action === 'upload') fileInput.click(); else if (action === 'newchat') createNewSession(); }
    function handleFileUpload(e) { console.log('Upload file:', e.target.files); e.target.value = ''; }
    function openFullPage() { window.open('/chatbot/', '_blank'); }
    function showError(msg) { const errorBubble = document.createElement('div'); errorBubble.className = 'message-bubble error'; errorBubble.textContent = msg; messagesArea.appendChild(errorBubble); scrollToBottom(); }

    init();

})();