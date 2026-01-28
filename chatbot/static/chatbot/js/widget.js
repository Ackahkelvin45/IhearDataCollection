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

        try {
            const response = await fetch(`/chatbot/api/sessions/${currentSessionId}/send_message_stream/`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-CSRFToken': getCSRFToken() },
                body: JSON.stringify({ message, stream: true })
            });
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.substring(6).trim());
                            if (data.type === 'token') {
                                accumulatedContent += data.content;
                                contentDiv.appendChild(document.createTextNode(data.content));
                                if (!scrollPending) {
                                    scrollPending = true;
                                    requestAnimationFrame(() => {
                                        scrollToBottom();
                                        scrollPending = false;
                                    });
                                }
                            } else if (data.type === 'source') sources.push(data);
                            else if (data.type === 'complete') {
                                const cursor = messageBubble.querySelector('.streaming-cursor');
                                if (cursor) cursor.remove();
                                messageBubble.classList.remove('streaming');
                                contentDiv.innerHTML = parseMarkdown(accumulatedContent);
                                if (sources.length > 0) addSourcesToMessage(messageBubble, sources);
                                scrollToBottom();
                            } else if (data.type === 'error') {
                                contentDiv.textContent = 'Error: ' + data.message;
                                const cursor = messageBubble.querySelector('.streaming-cursor');
                                if (cursor) cursor.remove();
                            }
                        } catch (e) { if (!line.includes(': heartbeat')) console.debug('Parse error:', e); }
                    }
                }
            }

            if (buffer.trim()) {
                try {
                    if (buffer.startsWith('data: ')) {
                        const data = JSON.parse(buffer.substring(6).trim());
                        if (data.type === 'token') {
                            accumulatedContent += data.content;
                            contentDiv.appendChild(document.createTextNode(data.content));
                        }
                    }
                } catch (e) { console.debug('Final buffer parse error:', e); }
            }
        } catch (error) {
            console.error('Streaming error:', error);
            contentDiv.textContent = 'Connection failed. Please try again.';
            const cursor = messageBubble.querySelector('.streaming-cursor');
            if (cursor) cursor.remove();
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
    function parseMarkdown(md) { return DOMPurify.sanitize(window.markdownit().render(md)); }
    function getCSRFToken() { return document.querySelector('[name=csrfmiddlewaretoken]').value; }
    function handleQuickAction(e) { const action = e.currentTarget.dataset.action; if (action === 'upload') fileInput.click(); else if (action === 'newchat') createNewSession(); }
    function handleFileUpload(e) { console.log('Upload file:', e.target.files); e.target.value = ''; }
    function openFullPage() { window.open('/chatbot/', '_blank'); }
    function showError(msg) { const errorBubble = document.createElement('div'); errorBubble.className = 'message-bubble error'; errorBubble.textContent = msg; messagesArea.appendChild(errorBubble); scrollToBottom(); }

    init();

})();