// Chatbot Frontend JavaScript with SSE Streaming Support

let currentSessionId = null;
let isStreaming = false;

// Get CSRF token
function getCSRFToken() {
    return document.querySelector('[name=csrfmiddlewaretoken]').value;
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    loadSessions();
    loadDocuments();
    setupEventListeners();
    
    // Check if there's a session ID from the widget
    const storedSessionId = sessionStorage.getItem('chatbotSessionId');
    if (storedSessionId) {
        // Clear the stored session ID
        sessionStorage.removeItem('chatbotSessionId');
        // Load the session
        selectSession(storedSessionId);
    }
});

// Setup event listeners
function setupEventListeners() {
    // New chat button
    document.getElementById('newChatBtn').addEventListener('click', createNewSession);
    
    // Message form
    document.getElementById('messageForm').addEventListener('submit', sendMessage);
    
    // File input change
    document.getElementById('fileInput').addEventListener('change', uploadDocument);
    
    // Auto-resize textarea
    const textarea = document.getElementById('messageInput');
    textarea.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
    
    // Handle Enter key (without Shift)
    textarea.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            document.getElementById('messageForm').dispatchEvent(new Event('submit'));
        }
    });
}

// Load sessions
async function loadSessions() {
    try {
        const response = await fetch('/chatbot/api/sessions/', {
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        displaySessions(data.results || data);
    } catch (error) {
        console.error('Error loading sessions:', error);
    }
}

// Display sessions
function displaySessions(sessions) {
    const sessionsList = document.getElementById('sessionsList');
    
    if (sessions.length === 0) {
        sessionsList.innerHTML = '<p class="text-sm text-gray-500 italic">No sessions yet</p>';
        return;
    }
    
    sessionsList.innerHTML = sessions.map(session => `
        <div class="session-item p-3 rounded-lg cursor-pointer hover:bg-gray-100 transition ${session.id === currentSessionId ? 'bg-blue-50 border-l-4 border-blue-600' : ''}"
             data-session-id="${session.id}"
             onclick="selectSession('${session.id}')">
            <div class="font-medium text-gray-800 text-sm truncate">${session.title}</div>
            <div class="text-xs text-gray-500">${session.message_count || 0} messages</div>
        </div>
    `).join('');
}

// Load documents
async function loadDocuments() {
    try {
        const response = await fetch('/chatbot/api/documents/', {
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        displayDocuments(data.results || data);
    } catch (error) {
        console.error('Error loading documents:', error);
    }
}

// Display documents
function displayDocuments(documents) {
    const documentsList = document.getElementById('documentsList');
    
    if (documents.length === 0) {
        documentsList.innerHTML = '<p class="text-xs text-gray-500 italic">No documents uploaded</p>';
        return;
    }
    
    documentsList.innerHTML = documents.map(doc => {
        const statusIcon = doc.processing_status === 'completed' ? '✓' : 
                          doc.processing_status === 'processing' ? '⏳' :
                          doc.processing_status === 'error' ? '✗' : '⋯';
        const statusColor = doc.processing_status === 'completed' ? 'text-green-600' : 
                           doc.processing_status === 'processing' ? 'text-yellow-600' :
                           doc.processing_status === 'error' ? 'text-red-600' : 'text-gray-400';
        
        return `
            <div class="flex items-center justify-between p-2 text-xs bg-gray-50 rounded">
                <div class="flex-1 truncate" title="${doc.title}">
                    <span class="${statusColor} mr-1">${statusIcon}</span>
                    ${doc.title}
                </div>
            </div>
        `;
    }).join('');
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
                title: 'New Chat'
            })
        });
        
        const session = await response.json();
        currentSessionId = session.id;
        
        // Reload sessions and select the new one
        await loadSessions();
        selectSession(session.id);
    } catch (error) {
        console.error('Error creating session:', error);
        alert('Failed to create new chat session');
    }
}

// Select a session
async function selectSession(sessionId) {
    currentSessionId = sessionId;
    
    // Update UI
    document.getElementById('messageInput').disabled = false;
    document.getElementById('sendBtn').disabled = false;
    
    // Load messages
    await loadMessages(sessionId);
    
    // Update active session styling
    document.querySelectorAll('.session-item').forEach(item => {
        if (item.dataset.sessionId === sessionId) {
            item.classList.add('bg-blue-50', 'border-l-4', 'border-blue-600');
        } else {
            item.classList.remove('bg-blue-50', 'border-l-4', 'border-blue-600');
        }
    });
}

// Load messages for a session
async function loadMessages(sessionId) {
    try {
        const response = await fetch(`/chatbot/api/sessions/${sessionId}/messages/`, {
            headers: {
                'Content-Type': 'application/json',
            }
        });
        
        const data = await response.json();
        const messages = data.results || data;
        
        displayMessages(messages);
    } catch (error) {
        console.error('Error loading messages:', error);
    }
}

// Display messages
function displayMessages(messages) {
    const chatMessages = document.getElementById('chatMessages');
    
    if (messages.length === 0) {
        chatMessages.innerHTML = `
            <div class="text-center text-gray-500 py-20">
                <p class="text-lg font-medium">Start asking questions</p>
                <p class="text-sm">I'll help you find information from your documents</p>
            </div>
        `;
        return;
    }
    
    chatMessages.innerHTML = messages.map(msg => createMessageHTML(msg)).join('');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Create message HTML
function createMessageHTML(message) {
    const isUser = message.role === 'user';
    
    let html = `
        <div class="flex ${isUser ? 'justify-end' : 'justify-start'}">
            <div class="max-w-3xl ${isUser ? 'bg-blue-600 text-white' : 'bg-white border border-gray-200'} rounded-lg px-4 py-3 shadow-sm">
                <div class="text-sm whitespace-pre-wrap">${escapeHTML(message.content)}</div>
                ${!isUser && message.sources && message.sources.length > 0 ? `
                    <div class="mt-3 pt-3 border-t border-gray-200 text-xs text-gray-500">
                        <div class="font-semibold mb-1">Sources:</div>
                        ${message.sources.map(s => `
                            <div class="truncate">📄 ${s.title} (Chunk ${s.chunk_index || 'N/A'})</div>
                        `).join('')}
                    </div>
                ` : ''}
            </div>
        </div>
    `;
    
    return html;
}

// Send message
async function sendMessage(e) {
    e.preventDefault();
    
    if (!currentSessionId) {
        alert('Please select or create a chat session first');
        return;
    }
    
    const messageInput = document.getElementById('messageInput');
    const message = messageInput.value.trim();
    
    if (!message) return;
    
    // Check if streaming is enabled
    const useStreaming = document.getElementById('streamToggle').checked;
    
    // Add user message to UI
    addMessageToUI('user', message);
    messageInput.value = '';
    messageInput.style.height = 'auto';
    
    // Disable input during processing
    setInputState(false);
    
    if (useStreaming) {
        await sendMessageWithStreaming(message);
    } else {
        await sendMessageWithoutStreaming(message);
    }
    
    // Re-enable input
    setInputState(true);
}

// Send message without streaming
async function sendMessageWithoutStreaming(message) {
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
        
        if (data.assistant_message) {
            addMessageToUI('assistant', data.assistant_message.content, data.assistant_message.sources);
        }
    } catch (error) {
        console.error('Error sending message:', error);
        addMessageToUI('system', 'Error: Failed to get response. Please try again.');
    }
}

// Send message with streaming (SSE)
async function sendMessageWithStreaming(message) {
    isStreaming = true;
    
    // Create placeholder for streaming response
    const assistantMsgId = 'streaming-' + Date.now();
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.id = assistantMsgId;
    messageDiv.className = 'flex justify-start';
    messageDiv.innerHTML = `
        <div class="max-w-3xl bg-white border border-gray-200 rounded-lg px-4 py-3 shadow-sm">
            <div class="streaming-content text-sm whitespace-pre-wrap"></div>
            <div class="streaming-cursor inline-block w-2 h-4 bg-blue-600 ml-1 animate-pulse"></div>
        </div>
    `;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    const streamingContent = messageDiv.querySelector('.streaming-content');
    const streamingCursor = messageDiv.querySelector('.streaming-cursor');
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
                        
                        if (data.type === 'token') {
                            accumulatedContent += data.content;
                            streamingContent.textContent = accumulatedContent;
                            chatMessages.scrollTop = chatMessages.scrollHeight;
                        } else if (data.type === 'source') {
                            sources.push(data);
                        } else if (data.type === 'complete') {
                            // Remove cursor
                            streamingCursor.remove();
                            
                            // Add sources if any
                            if (sources.length > 0) {
                                const sourcesDiv = document.createElement('div');
                                sourcesDiv.className = 'mt-3 pt-3 border-t border-gray-200 text-xs text-gray-500';
                                sourcesDiv.innerHTML = `
                                    <div class="font-semibold mb-1">Sources:</div>
                                    ${sources.map(s => `
                                        <div class="truncate">📄 ${s.title}</div>
                                    `).join('')}
                                `;
                                messageDiv.querySelector('.max-w-3xl').appendChild(sourcesDiv);
                            }
                        } else if (data.type === 'error') {
                            streamingContent.textContent = 'Error: ' + data.message;
                            streamingCursor.remove();
                        }
                    } catch (e) {
                        // Ignore parsing errors for heartbeats
                    }
                }
            }
        }
    } catch (error) {
        console.error('Streaming error:', error);
        streamingContent.textContent = 'Error: Connection failed';
        streamingCursor.remove();
    }
    
    isStreaming = false;
}

// Add message to UI
function addMessageToUI(role, content, sources = []) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.innerHTML = createMessageHTML({ role, content, sources });
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Upload document
async function uploadDocument(e) {
    const file = e.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('title', file.name);
    
    try {
        // Show loading state
        const documentsList = document.getElementById('documentsList');
        documentsList.innerHTML = '<p class="text-xs text-gray-500">Uploading...</p>';
        
        const response = await fetch('/chatbot/api/documents/', {
            method: 'POST',
            headers: {
                'X-CSRFToken': getCSRFToken(),
            },
            body: formData
        });
        
        if (response.ok) {
            // Reload documents
            await loadDocuments();
            
            // Clear file input
            e.target.value = '';
            
            // Show success message
            alert('Document uploaded successfully! Processing in background...');
        } else {
            const error = await response.json();
            alert('Upload failed: ' + (error.file?.[0] || 'Unknown error'));
            await loadDocuments();
        }
    } catch (error) {
        console.error('Error uploading document:', error);
        alert('Failed to upload document');
        await loadDocuments();
    }
}

// Set input state
function setInputState(enabled) {
    document.getElementById('messageInput').disabled = !enabled;
    document.getElementById('sendBtn').disabled = !enabled;
}

// Escape HTML
function escapeHTML(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

