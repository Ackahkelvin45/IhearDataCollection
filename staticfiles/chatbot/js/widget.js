// Chat Widget JavaScript
(function() {
    'use strict';

    let currentSessionId = null;
    let isExpanded = false;
    let isStreaming = false;

    // DOM Elements
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

    // Initialize
    function init() {
        if (!messageForm || !messageInput) {
            console.warn('Chat widget: form or message input not found');
            return;
        }
        setupEventListeners();
        initializeSession();
    }

    // Setup event listeners
    function setupEventListeners() {
        if (toggleBtn) toggleBtn.addEventListener('click', toggleWidget);
        if (minimizeBtn) minimizeBtn.addEventListener('click', toggleWidget);

        // Message form - prevent default and call handler directly so submit always works
        messageForm.addEventListener('submit', function(e) {
            e.preventDefault();
            handleSendMessage(e);
        });

        // Auto-resize textarea
        messageInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 100) + 'px';
        });

        // Enter to send (Shift+Enter for new line) - call handler directly so it works reliably
        messageInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSendMessage(e);
            }
        });

        // Quick actions
        document.querySelectorAll('.quick-action-btn').forEach(btn => {
            btn.addEventListener('click', handleQuickAction);
        });

        // Suggestion buttons
        document.querySelectorAll('.suggestion-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const question = this.dataset.question;
                messageInput.value = question;
                handleSendMessage(new Event('submit'));
            });
        });

        if (fileInput) fileInput.addEventListener('change', handleFileUpload);

        // Open full page button
        const openFullPageBtn = document.getElementById('openFullPageBtn');
        if (openFullPageBtn) {
            openFullPageBtn.addEventListener('click', openFullPage);
        }
    }

    // Toggle widget open/close
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

    // Initialize or create session
    async function initializeSession() {
        try {
            // Try to get the most recent active session
            const response = await fetch('/chatbot/api/sessions/?page_size=1');
            const data = await response.json();

            if (data.results && data.results.length > 0) {
                currentSessionId = data.results[0].id;
                await loadMessages();
            } else {
                // Create a new session
                await createNewSession();
            }
        } catch (error) {
            console.error('Error initializing session:', error);
        }
    }

    // Create new session
    async function createNewSession() {
        try {
            const response = await fetch('/chatbot/api/sessions/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCSRFToken(),
                },
                body: JSON.stringify({
                    title: 'Quick Chat ' + new Date().toLocaleString()
                })
            });

            const session = await response.json();
            currentSessionId = session.id;
            clearMessages();
        } catch (error) {
            console.error('Error creating session:', error);
            showError('Failed to start chat session');
        }
    }

    // Load messages
    async function loadMessages() {
        if (!currentSessionId) return;

        try {
            const response = await fetch(`/chatbot/api/sessions/${currentSessionId}/messages/`);
            const data = await response.json();
            const messages = data.results || data;

            clearMessages();

            if (messages.length === 0) {
                // Show welcome message
                return;
            }

            messages.forEach(msg => {
                addMessageToUI(msg.role, msg.content, msg.sources);
            });

            scrollToBottom();
        } catch (error) {
            console.error('Error loading messages:', error);
        }
    }

    // Handle send message
    async function handleSendMessage(e) {
        if (e && e.preventDefault) e.preventDefault();

        if (!currentSessionId) {
            await createNewSession();
        }

        const message = messageInput.value.trim();
        if (!message) return;

        // Add user message to UI
        addMessageToUI('user', message);
        messageInput.value = '';
        messageInput.style.height = 'auto';

        // Disable input
        setInputState(false);

        // Check if streaming is enabled (default to true when toggle is missing, e.g. commented out in template)
        const useStreaming = !streamToggle || streamToggle.checked;

        if (useStreaming) {
            await sendMessageWithStreaming(message);
        } else {
            await sendMessageWithoutStreaming(message);
        }

        // Re-enable input
        setInputState(true);
        messageInput.focus();
    }

    // Send message without streaming
    async function sendMessageWithoutStreaming(message) {
        // Show typing indicator
        showTypingIndicator();

        try {
            const response = await fetch(`/chatbot/api/sessions/${currentSessionId}/send_message/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCSRFToken(),
                },
                body: JSON.stringify({ message })
            });

            const data = await response.json();

            // Remove typing indicator
            removeTypingIndicator();

            if (data.assistant_message) {
                addMessageToUI('assistant', data.assistant_message.content, data.assistant_message.sources);
            }
        } catch (error) {
            console.error('Error sending message:', error);
            removeTypingIndicator();
            showError('Failed to get response. Please try again.');
        }
    }

    // Send message with streaming
    async function sendMessageWithStreaming(message) {
        isStreaming = true;

        // Create streaming message bubble
        const messageId = 'stream-' + Date.now();
        const messageBubble = createStreamingBubble(messageId);
        messageBubble.classList.add('streaming');
        messagesArea.appendChild(messageBubble);
        scrollToBottom();

        const contentDiv = messageBubble.querySelector('.message-content');
        let accumulatedContent = '';
        let sources = [];

        try {
            const response = await fetch(`/chatbot/api/sessions/${currentSessionId}/send_message_stream/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCSRFToken(),
                },
                body: JSON.stringify({ message, stream: true })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');

                for (const line of lines) {
                    if (line.startsWith('data:')) {
                        try {
                            const data = JSON.parse(line.substring(5).trim());

                            if (data.type === 'querying') {
                                showQueryingState(contentDiv);
                            } else if (data.type === 'token') {
                                accumulatedContent += data.content;
                                contentDiv.textContent = accumulatedContent;
                                scrollToBottom();
                            } else if (data.type === 'source') {
                                sources.push(data);
                            } else if (data.type === 'complete') {
                                messageBubble.classList.remove('streaming');
                                const cursor = messageBubble.querySelector('.streaming-cursor');
                                if (cursor) cursor.remove();

                                if (sources.length > 0) {
                                    addSourcesToMessage(messageBubble, sources);
                                }
                            } else if (data.type === 'error') {
                                messageBubble.classList.remove('streaming');
                                contentDiv.textContent = 'Error: ' + data.message;
                                const cursor = messageBubble.querySelector('.streaming-cursor');
                                if (cursor) cursor.remove();
                            }
                        } catch (e) {
                            // Ignore parse errors (heartbeats)
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Streaming error:', error);
            messageBubble.classList.remove('streaming');
            contentDiv.textContent = 'Connection failed. Please try again.';
            const cursor = messageBubble.querySelector('.streaming-cursor');
            if (cursor) cursor.remove();
        }

        isStreaming = false;
    }

    // Add message to UI
    function addMessageToUI(role, content, sources = []) {
        const messageBubble = document.createElement('div');
        messageBubble.className = `message-bubble ${role}`;

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        messageContent.textContent = content;

        messageBubble.appendChild(messageContent);

        // Add sources if present
        if (sources && sources.length > 0) {
            addSourcesToMessage(messageBubble, sources);
        }

        // Remove welcome message if present
        const welcomeMsg = messagesArea.querySelector('.welcome-message');
        if (welcomeMsg) {
            welcomeMsg.remove();
        }

        messagesArea.appendChild(messageBubble);
        scrollToBottom();
    }

    function showQueryingState(contentDiv) {
        if (!contentDiv) return;
        contentDiv.innerHTML = '<span class="querying-text">Querying database</span><div class="typing-indicator typing-indicator-inline"><span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span></div>';
    }

    // Remove streaming cursor from any other assistant bubble (only current message should show it)
    function removeStreamingCursorFromOtherBubbles(currentBubble) {
        messagesArea.querySelectorAll('.message-bubble.assistant').forEach(function(bubble) {
            if (currentBubble != null && bubble === currentBubble) return;
            const c = bubble.querySelector('.streaming-cursor');
            if (c) c.remove();
        });
    }

    // Create streaming bubble
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
        if (welcomeMsg) {
            welcomeMsg.remove();
        }

        return messageBubble;
    }

    // Add sources to message
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

    // Show typing indicator
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

        // Remove welcome message if present
        const welcomeMsg = messagesArea.querySelector('.welcome-message');
        if (welcomeMsg) {
            welcomeMsg.remove();
        }

        messagesArea.appendChild(indicator);
        scrollToBottom();
    }

    // Remove typing indicator
    function removeTypingIndicator() {
        const indicator = document.getElementById('typing-indicator');
        if (indicator) {
            indicator.remove();
        }
    }

    // Clear messages
    function clearMessages() {
        messagesArea.innerHTML = `
            <div class="welcome-message">

                <h4 class="welcome-title">Welcome!</h4>
                <p class="welcome-text">I'm your AI assistant. Ask me anything about your project.</p>
            </div>
        `;
    }

    // Handle quick actions
    function handleQuickAction(e) {
        const action = e.currentTarget.dataset.action;

        if (action === 'upload') {
            fileInput.click();
        } else if (action === 'newchat') {
            createNewSession();
        }
    }

    // Handle file upload
    async function handleFileUpload(e) {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);
        formData.append('title', file.name);

        try {
            addMessageToUI('system', `Uploading ${file.name}...`);

            const response = await fetch('/chatbot/api/documents/', {
                method: 'POST',
                headers: {
                    'X-CSRFToken': getCSRFToken(),
                },
                body: formData
            });

            if (response.ok) {
                addMessageToUI('system', `✓ Document uploaded successfully! Processing in background...`);
            } else {
                const error = await response.json();
                showError('Upload failed: ' + (error.file?.[0] || 'Unknown error'));
            }

            // Clear file input
            e.target.value = '';
        } catch (error) {
            console.error('Error uploading file:', error);
            showError('Failed to upload document');
        }
    }

    // Show error
    function showError(message) {
        addMessageToUI('system', '❌ ' + message);
    }

    // Set input state
    function setInputState(enabled) {
        messageInput.disabled = !enabled;
        sendBtn.disabled = !enabled;
    }

    // Open full chatbot page with current session
    function openFullPage() {
        // Store the current session ID in sessionStorage if it exists
        if (currentSessionId) {
            sessionStorage.setItem('chatbotSessionId', currentSessionId);
        }
        // Navigate to the full chatbot page
        window.location.href = '/chatbot/';
    }

    // Scroll to bottom
    function scrollToBottom() {
        messagesArea.scrollTop = messagesArea.scrollHeight;
    }

    // Get CSRF token
    function getCSRFToken() {
        const token = document.querySelector('[name=csrfmiddlewaretoken]');
        return token ? token.value : '';
    }

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
