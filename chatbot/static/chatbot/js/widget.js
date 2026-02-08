(function() {
    'use strict';

    let currentSessionId = null;
    let isExpanded = false;
    let isStreaming = false;
    let scrollPending = false;
    let lastUserMessage = null;
    let onSessionChange = null;

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
    const chatIcon = toggleBtn ? toggleBtn.querySelector('.chat-icon') : null;
    const closeIcon = toggleBtn ? toggleBtn.querySelector('.close-icon') : null;
    const isHomePage = document.getElementById('homeChatContainer') != null;

    function init() {
        if (!messageForm || !messageInput) {
            console.warn('Chat widget: form or message input not found');
            return;
        }
        setupEventListeners();
        if (isHomePage) {
            var loadSessionId = sessionStorage.getItem('loadSessionId');
            var initialMessage = sessionStorage.getItem('initialMessage');
            sessionStorage.removeItem('loadSessionId');
            sessionStorage.removeItem('initialMessage');
            if (loadSessionId) {
                currentSessionId = loadSessionId;
                loadMessages().then(function() {
                    if (initialMessage) {
                        messageInput.value = initialMessage;
                        handleSendMessage(new Event('submit'));
                    }
                });
            } else {
                initializeSession();
            }
        } else {
            initializeSession();
        }
    }

    function setupEventListeners() {
        if (toggleBtn) toggleBtn.addEventListener('click', toggleWidget);
        if (minimizeBtn) minimizeBtn.addEventListener('click', toggleWidget);

        messageForm.addEventListener('submit', function(e) {
            e.preventDefault();
            handleSendMessage(e);
        });

        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 100) + 'px';
        });

        messageInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage(e);
            }
        });

        document.querySelectorAll('.quick-action-btn').forEach(btn => btn.addEventListener('click', handleQuickAction));
        document.querySelectorAll('.suggestion-btn').forEach(btn => btn.addEventListener('click', function() {
            messageInput.value = this.dataset.question;
            handleSendMessage(new Event('submit'));
        }));

        if (fileInput) fileInput.addEventListener('change', handleFileUpload);

        const openFullPageBtn = document.getElementById('openFullPageBtn');
        if (openFullPageBtn) openFullPageBtn.addEventListener('click', openFullPage);
    }

    function toggleWidget() {
        if (!widgetWindow) return;
        isExpanded = !isExpanded;
        if (isExpanded) {
            widgetWindow.classList.remove('hidden');
            if (chatIcon) chatIcon.classList.add('hidden');
            if (closeIcon) closeIcon.classList.remove('hidden');
            messageInput.focus();
        } else {
            widgetWindow.classList.add('hidden');
            if (chatIcon) chatIcon.classList.remove('hidden');
            if (closeIcon) closeIcon.classList.add('hidden');
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
            if (typeof onSessionChange === 'function') onSessionChange();
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
            messages.forEach(msg => addMessageToUI(msg.role, msg.content, msg.sources, msg.metadata || null, null));
            scrollToBottom();
        } catch (error) { console.error('Error loading messages:', error); }
    }

    async function handleSendMessage(e) {
        if (e && e.preventDefault) e.preventDefault();
        if (!currentSessionId) await createNewSession();
        const message = messageInput.value.trim();
        if (!message) return;

        lastUserMessage = message;
        addMessageToUI('user', message);
        messageInput.value = '';
        messageInput.style.height = 'auto';
        setInputState(false);

        var useStreaming = !streamToggle || streamToggle.checked;
        if (useStreaming) await sendMessageWithStreaming(message);
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
            if (data.assistant_message) {
                addMessageToUI(
                    'assistant',
                    data.assistant_message.content,
                    data.assistant_message.sources,
                    data.assistant_message.metadata,
                    lastUserMessage
                );
            }
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
                // Remove cursor temporarily for rendering (only from this bubble)
                const cursor = messageBubble.querySelector('.streaming-cursor');
                if (cursor) cursor.remove();
                
                // Render markdown
                contentDiv.innerHTML = parseMarkdown(accumulatedContent);
                
                // Re-add cursor only to this (current) streaming message
                if (isStreaming && messageBubble.classList.contains('streaming')) {
                    removeStreamingCursorFromOtherBubbles(messageBubble);
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
                        if (currentData.type === 'querying') {
                            showQueryingState(contentDiv);
                        } else if (currentData.type === 'token') {
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
                        } else if (currentData.type === 'table') {
                            addTableToMessage(messageBubble, currentData.table, lastUserMessage, currentData.pagination);
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

    function addMessageToUI(role, content, sources = [], metadata = null, originQuestion = null) {
        const messageBubble = document.createElement('div');
        messageBubble.className = `message-bubble ${role}`;
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.innerHTML = parseMarkdown(content);
        messageBubble.appendChild(messageContent);
        if (sources && sources.length > 0) addSourcesToMessage(messageBubble, sources);

        if (metadata && metadata.table) {
            addTableToMessage(messageBubble, metadata.table, originQuestion, metadata.pagination);
        }

        const welcomeMsg = messagesArea.querySelector('.welcome-message');
        if (welcomeMsg) welcomeMsg.remove();

        messagesArea.appendChild(messageBubble);
        scrollToBottom();
    }

    /** Remove streaming cursor from any assistant bubble that is not the current one */
    function removeStreamingCursorFromOtherBubbles(currentBubble) {
        messagesArea.querySelectorAll('.message-bubble.assistant').forEach(function(bubble) {
            if (currentBubble != null && bubble === currentBubble) return;
            const c = bubble.querySelector('.streaming-cursor');
            if (c) c.remove();
        });
    }

    function createStreamingBubble(id) {
        removeStreamingCursorFromOtherBubbles(null);
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

    function addTableToMessage(messageBubble, tableData, originQuestion, paginationOverride) {
        if (!tableData || !messageBubble) return;

        const existing = messageBubble.querySelector('.message-table');
        if (existing) existing.remove();

        const container = document.createElement('div');
        container.className = 'message-table';

        const tableEl = document.createElement('table');
        tableEl.className = 'data-table';

        const columns = Array.isArray(tableData.columns) ? tableData.columns : [];
        const rows = Array.isArray(tableData.rows) ? tableData.rows : [];

        if (columns.length > 0) {
            const thead = document.createElement('thead');
            const headerRow = document.createElement('tr');
            columns.forEach(col => {
                const th = document.createElement('th');
                th.textContent = col;
                headerRow.appendChild(th);
            });
            thead.appendChild(headerRow);
            tableEl.appendChild(thead);
        }

        const tbody = document.createElement('tbody');
        if (rows.length === 0) {
            const emptyRow = document.createElement('tr');
            const td = document.createElement('td');
            td.colSpan = Math.max(columns.length, 1);
            td.textContent = 'No rows returned.';
            emptyRow.appendChild(td);
            tbody.appendChild(emptyRow);
        } else {
            rows.forEach(row => {
                const tr = document.createElement('tr');
                columns.forEach(col => {
                    const td = document.createElement('td');
                    const val = row[col];
                    if (val === null || val === undefined) {
                        td.textContent = '';
                    } else if (typeof val === 'object') {
                        td.textContent = JSON.stringify(val);
                    } else {
                        td.textContent = String(val);
                    }
                    tr.appendChild(td);
                });
                tbody.appendChild(tr);
            });
        }
        tableEl.appendChild(tbody);
        container.appendChild(tableEl);

        const pagination = paginationOverride || tableData;
        const page = parseInt(pagination && pagination.page, 10) || 1;
        const pageSize = parseInt(pagination && pagination.page_size, 10) || rows.length || 1;
        const hasMore = !!(pagination && pagination.has_more);

        const meta = document.createElement('div');
        meta.className = 'table-meta';
        const metaText = document.createElement('div');
        metaText.textContent = `Rows: ${rows.length} · Page: ${page}`;
        meta.appendChild(metaText);

        if (originQuestion && (page > 1 || hasMore)) {
            const controls = document.createElement('div');
            controls.className = 'pagination-controls';

            const prevBtn = document.createElement('button');
            prevBtn.className = 'pagination-btn';
            prevBtn.textContent = 'Prev';
            prevBtn.disabled = page <= 1 || isStreaming;
            prevBtn.addEventListener('click', function() {
                const nextQuestion = buildPaginatedQuestion(originQuestion, page - 1, pageSize);
                sendPaginationQuestion(nextQuestion);
            });

            const nextBtn = document.createElement('button');
            nextBtn.className = 'pagination-btn';
            nextBtn.textContent = 'Next';
            nextBtn.disabled = !hasMore || isStreaming;
            nextBtn.addEventListener('click', function() {
                const nextQuestion = buildPaginatedQuestion(originQuestion, page + 1, pageSize);
                sendPaginationQuestion(nextQuestion);
            });

            controls.appendChild(prevBtn);
            controls.appendChild(nextBtn);
            meta.appendChild(controls);
        }

        container.appendChild(meta);

        messageBubble.appendChild(container);
        scrollToBottom();
    }

    function buildPaginatedQuestion(question, page, pageSize) {
        let base = String(question || '');
        base = base.replace(/\bpage\s+\d+\b/ig, '').replace(/\bpage\s*size\s+\d+\b/ig, '');
        base = base.replace(/\bpagesize\s+\d+\b/ig, '').replace(/\blimit\s+\d+\b/ig, '').replace(/\boffset\s+\d+\b/ig, '');
        base = base.replace(/\s{2,}/g, ' ').trim();
        return `${base} page ${page} limit ${pageSize}`.trim();
    }

    function sendPaginationQuestion(question) {
        if (!question) return;
        messageInput.value = question;
        handleSendMessage(new Event('submit'));
    }

    /** Show "Querying database" with 3-dot animation in the given content div (streaming bubble) */
    function showQueryingState(contentDiv) {
        if (!contentDiv) return;
        contentDiv.innerHTML = '<span class="querying-text">Querying database</span><div class="typing-indicator typing-indicator-inline"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div>';
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
    function getCSRFToken() {
        var el = document.querySelector('[name=csrfmiddlewaretoken]');
        return el ? el.value : '';
    }
    function handleQuickAction(e) { const action = e.currentTarget.dataset.action; if (action === 'upload') fileInput.click(); else if (action === 'newchat') createNewSession(); }
    function handleFileUpload(e) { console.log('Upload file:', e.target.files); e.target.value = ''; }
    function openFullPage() { window.open('/chatbot/', '_blank'); }
    function showError(msg) { const errorBubble = document.createElement('div'); errorBubble.className = 'message-bubble error'; errorBubble.textContent = msg; messagesArea.appendChild(errorBubble); scrollToBottom(); }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    window.ChatWidget = {
        loadSession: function(id) {
            currentSessionId = id;
            loadMessages();
            if (typeof onSessionChange === 'function') onSessionChange();
        },
        createNewSession: createNewSession,
        sendMessage: function(msg) {
            if (msg) messageInput.value = msg;
            handleSendMessage(new Event('submit'));
        },
        getCurrentSessionId: function() { return currentSessionId; },
        setOnSessionChange: function(fn) { onSessionChange = fn; }
    };

})();
