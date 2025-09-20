# 🎵 I Hear Audio Data Bank - Frontend Integration Guide

This guide provides comprehensive instructions for frontend developers to integrate with the I Hear Audio Data Bank ChatSessionView API, including session management, message handling, streaming responses, and visualization implementation.

## 📋 Table of Contents

1. [API Overview](#api-overview)
2. [Authentication](#authentication)
3. [Session Management](#session-management)
4. [Message Handling](#message-handling)
5. [Streaming Responses](#streaming-responses)
6. [Visualization Implementation](#visualization-implementation)
7. [Error Handling](#error-handling)
8. [Complete Examples](#complete-examples)

## 🌐 API Overview

### Base URL
```
http://localhost:8000
```

### Main Endpoints
- `POST /insights/sessions/` - Create new chat session
- `GET /insights/sessions/` - List user sessions
- `GET /insights/sessions/{id}/` - Get session details
- `POST /insights/sessions/{id}/messages/` - Send message to session
- `DELETE /insights/sessions/{id}/messages/{message_id}/` - Delete message
- `POST /insights/sessions/{id}/archive/` - Archive session

## 🔐 Authentication

All API requests require authentication. Include the authentication token in the header:

```javascript
const headers = {
    'Authorization': `Bearer ${authToken}`,
    'Content-Type': 'application/json'
};
```

## 📝 Session Management

### 1. Create New Session

**Endpoint:** `POST /insights/sessions/`

**Request Body:**
```json
{
    "title": null  // Optional - will be auto-generated from first message
}
```

**Response:**
```json
{
    "id": 1,
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "title": null,
    "status": "active",
    "total_messages": 0,
    "created_at": "2025-09-19T16:00:00Z",
    "updated_at": "2025-09-19T16:00:00Z"
}
```

**JavaScript Example:**
```javascript
async function createSession() {
    try {
        const response = await fetch('/insights/sessions/', {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({ title: null })
        });
        
        const session = await response.json();
        console.log('Session created:', session);
        return session;
    } catch (error) {
        console.error('Error creating session:', error);
    }
}
```

### 2. List User Sessions

**Endpoint:** `GET /insights/sessions/`

**Response:**
```json
{
    "count": 1,
    "next": null,
    "previous": null,
    "results": [
        {
            "id": 1,
            "session_id": "550e8400-e29b-41d4-a716-446655440000",
            "title": "Show me the distribution of audio files by region",
            "status": "active",
            "total_messages": 5,
            "created_at": "2025-09-19T16:00:00Z",
            "updated_at": "2025-09-19T16:05:00Z",
            "messages": [
                {
                    "id": 1,
                    "user_input": "Show me the distribution of audio files by region",
                    "assistant_response": "Based on your audio data...",
                    "created_at": "2025-09-19T16:00:00Z",
                    "status": "completed",
                    "visulization": { /* visualization data */ }
                }
            ]
        }
    ]
}
```

### 3. Get Session Details

**Endpoint:** `GET /insights/sessions/{id}/`

**Response:** Same as session object above with all messages.

### 4. Archive Session

**Endpoint:** `POST /insights/sessions/{id}/archive/`

**Response:**
```json
{
    "message": "Session archived successfully"
}
```

## 💬 Message Handling

### Send Message

**Endpoint:** `POST /insights/sessions/{id}/messages/`

**Request Body:**
```json
{
    "user_input": "Show me the distribution of audio files by category",
    "ai_answer": true  // Optional - whether to include AI interpretation
}
```

**Response:** Streaming response (see Streaming Responses section)

**JavaScript Example:**
```javascript
async function sendMessage(sessionId, userInput, aiAnswer = true) {
    try {
        const response = await fetch(`/insights/sessions/${sessionId}/messages/`, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({
                user_input: userInput,
                ai_answer: aiAnswer
            })
        });
        
        // Handle streaming response
        return handleStreamingResponse(response);
    } catch (error) {
        console.error('Error sending message:', error);
    }
}
```

### Delete Message

**Endpoint:** `DELETE /insights/sessions/{id}/messages/{message_id}/`

**Response:**
```json
{
    "message": "Message deleted successfully"
}
```

## 📡 Streaming Responses

The message endpoint returns a streaming response with different action types:

### Stream Message Format
```json
{"action": "action_type", "data": data_content}
```

### Action Types

1. **`tool_call`** - AI is calling a tool
2. **`tool_response`** - Tool execution result
3. **`visualization`** - Visualization recommendation
4. **`llm`** - AI response chunk
5. **`completed`** - Message processing complete
6. **`error`** - Error occurred

### JavaScript Streaming Handler

```javascript
async function handleStreamingResponse(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    
    const messageData = {
        toolCalls: [],
        toolResponses: [],
        visualizations: [],
        llmResponse: '',
        completed: null,
        errors: []
    };
    
    try {
        while (true) {
            const { done, value } = await reader.read();
            
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer
            
            for (const line of lines) {
                if (line.trim()) {
                    try {
                        const message = JSON.parse(line);
                        handleStreamMessage(message, messageData);
                    } catch (e) {
                        console.warn('Failed to parse stream message:', line);
                    }
                }
            }
        }
    } catch (error) {
        console.error('Streaming error:', error);
    }
    
    return messageData;
}

function handleStreamMessage(message, messageData) {
    const { action, data } = message;
    
    switch (action) {
        case 'tool_call':
            messageData.toolCalls.push(data);
            break;
            
        case 'tool_response':
            messageData.toolResponses.push(data);
            break;
            
        case 'visualization':
            messageData.visualizations.push(data);
            break;
            
        case 'llm':
            messageData.llmResponse += data;
            break;
            
        case 'completed':
            messageData.completed = data;
            break;
            
        case 'error':
            messageData.errors.push(data);
            break;
    }
}
```

## 📊 Visualization Implementation

### Visualization Response Structure

When the AI recommends a visualization, you'll receive a structured response:

```json
{
    "visualization_type": "pie_chart",
    "visualization_name": "Pie Chart",
    "chart_template": {
        "type": "pie",
        "config": {
            "data": {
                "labels": [],
                "datasets": [{
                    "data": [],
                    "backgroundColor": ["#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0"]
                }]
            },
            "options": {
                "responsive": true,
                "plugins": {
                    "legend": {"position": "bottom"},
                    "title": {"display": true, "text": "Data Distribution"}
                }
            }
        }
    },
    "frontend_data": {
        "type": "pie_chart",
        "name": "Pie Chart",
        "config": { /* Chart.js configuration */ },
        "data_structure": {
            "labels": "Array of category names",
            "data": "Array of values corresponding to labels",
            "description": "For showing proportions/percentages of audio categories"
        },
        "description": "Query asks for proportions or distribution of audio categories/regions"
    }
}
```

### Chart.js Integration

```javascript
// Include Chart.js in your HTML
// <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

function createVisualization(visualizationData, actualData) {
    const frontendData = visualizationData.frontend_data;
    const ctx = document.getElementById('chartCanvas').getContext('2d');
    
    // Get the predefined Chart.js configuration
    const config = JSON.parse(JSON.stringify(frontendData.config));
    
    // Add your actual data
    config.data.labels = actualData.labels;
    config.data.datasets[0].data = actualData.data;
    
    // Update title if needed
    config.options.plugins.title.text = frontendData.description;
    
    // Create the chart
    return new Chart(ctx, config);
}

// Example usage with audio data
const audioData = {
    labels: ['Urban Life and Public Spaces', 'Natural Soundscapes and Biodiversity'],
    data: [1, 1]
};

const chart = createVisualization(visualizationData, audioData);
```

### Supported Chart Types

1. **Pie Chart** - Category distributions
2. **Bar Chart** - Comparisons across categories
3. **Line Chart** - Trends over time
4. **Heatmap** - Correlations and patterns
5. **Scatter Plot** - Relationships between variables
6. **Box Plot** - Distribution analysis
7. **Area Chart** - Cumulative data

### Chart Type Detection

```javascript
function getChartType(visualizationData) {
    return visualizationData.frontend_data.type;
}

function getChartName(visualizationData) {
    return visualizationData.frontend_data.name;
}

function getDataStructure(visualizationData) {
    return visualizationData.frontend_data.data_structure;
}
```

## ⚠️ Error Handling

### Common Error Responses

```json
{
    "action": "error",
    "data": {
        "message": "Error description"
    }
}
```

### Error Handling Implementation

```javascript
function handleError(errorData) {
    const errorMessage = errorData.message;
    
    // Display error to user
    showErrorMessage(errorMessage);
    
    // Log error for debugging
    console.error('API Error:', errorMessage);
    
    // Handle specific error types
    if (errorMessage.includes('authentication')) {
        redirectToLogin();
    } else if (errorMessage.includes('session')) {
        refreshSession();
    }
}
```

## 🎯 Complete Examples

### 1. Full Chat Implementation

```javascript
class AudioDataChat {
    constructor(apiBaseUrl, authToken) {
        this.apiBaseUrl = apiBaseUrl;
        this.headers = {
            'Authorization': `Bearer ${authToken}`,
            'Content-Type': 'application/json'
        };
        this.currentSession = null;
    }
    
    async createSession() {
        const response = await fetch(`${this.apiBaseUrl}/insights/sessions/`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({ title: null })
        });
        
        this.currentSession = await response.json();
        return this.currentSession;
    }
    
    async sendMessage(userInput, onStreamMessage) {
        if (!this.currentSession) {
            await this.createSession();
        }
        
        const response = await fetch(
            `${this.apiBaseUrl}/insights/sessions/${this.currentSession.id}/messages/`,
            {
                method: 'POST',
                headers: this.headers,
                body: JSON.stringify({
                    user_input: userInput,
                    ai_answer: true
                })
            }
        );
        
        return this.handleStreamingResponse(response, onStreamMessage);
    }
    
    async handleStreamingResponse(response, onMessage) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        const messageData = {
            toolCalls: [],
            toolResponses: [],
            visualizations: [],
            llmResponse: '',
            completed: null,
            errors: []
        };
        
        try {
            while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();
                
                for (const line of lines) {
                    if (line.trim()) {
                        try {
                            const message = JSON.parse(line);
                            this.handleStreamMessage(message, messageData);
                            
                            // Call the callback for real-time updates
                            if (onMessage) {
                                onMessage(message, messageData);
                            }
                        } catch (e) {
                            console.warn('Failed to parse stream message:', line);
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Streaming error:', error);
        }
        
        return messageData;
    }
    
    handleStreamMessage(message, messageData) {
        const { action, data } = message;
        
        switch (action) {
            case 'tool_call':
                messageData.toolCalls.push(data);
                break;
            case 'tool_response':
                messageData.toolResponses.push(data);
                break;
            case 'visualization':
                messageData.visualizations.push(data);
                break;
            case 'llm':
                messageData.llmResponse += data;
                break;
            case 'completed':
                messageData.completed = data;
                break;
            case 'error':
                messageData.errors.push(data);
                break;
        }
    }
    
    async deleteMessage(messageId) {
        const response = await fetch(
            `${this.apiBaseUrl}/insights/sessions/${this.currentSession.id}/messages/${messageId}/`,
            {
                method: 'DELETE',
                headers: this.headers
            }
        );
        
        return response.json();
    }
    
    async archiveSession() {
        const response = await fetch(
            `${this.apiBaseUrl}/insights/sessions/${this.currentSession.id}/archive/`,
            {
                method: 'POST',
                headers: this.headers
            }
        );
        
        return response.json();
    }
}

// Usage
const chat = new AudioDataChat('http://localhost:8000', 'your-auth-token');

// Create session and send message
chat.createSession().then(() => {
    chat.sendMessage('Show me the distribution of audio files by category', (message, data) => {
        // Real-time updates
        if (message.action === 'llm') {
            updateChatUI(data.llmResponse);
        } else if (message.action === 'visualization') {
            createVisualization(message.data);
        }
    });
});
```

### 2. React Component Example

```jsx
import React, { useState, useEffect } from 'react';
import { Chart } from 'chart.js';

const AudioDataChat = ({ authToken }) => {
    const [session, setSession] = useState(null);
    const [messages, setMessages] = useState([]);
    const [currentMessage, setCurrentMessage] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [visualizations, setVisualizations] = useState([]);
    
    useEffect(() => {
        createSession();
    }, []);
    
    const createSession = async () => {
        try {
            const response = await fetch('/insights/sessions/', {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${authToken}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ title: null })
            });
            
            const newSession = await response.json();
            setSession(newSession);
        } catch (error) {
            console.error('Error creating session:', error);
        }
    };
    
    const sendMessage = async () => {
        if (!currentMessage.trim() || !session) return;
        
        setIsLoading(true);
        const userMessage = {
            id: Date.now(),
            type: 'user',
            content: currentMessage,
            timestamp: new Date()
        };
        
        setMessages(prev => [...prev, userMessage]);
        setCurrentMessage('');
        
        try {
            const response = await fetch(`/insights/sessions/${session.id}/messages/`, {
                method: 'POST',
                headers: {
                    'Authorization': `Bearer ${authToken}`,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    user_input: currentMessage,
                    ai_answer: true
                })
            });
            
            await handleStreamingResponse(response);
        } catch (error) {
            console.error('Error sending message:', error);
        } finally {
            setIsLoading(false);
        }
    };
    
    const handleStreamingResponse = async (response) => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let assistantMessage = {
            id: Date.now(),
            type: 'assistant',
            content: '',
            timestamp: new Date(),
            visualizations: []
        };
        
        setMessages(prev => [...prev, assistantMessage]);
        
        try {
            while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();
                
                for (const line of lines) {
                    if (line.trim()) {
                        try {
                            const message = JSON.parse(line);
                            
                            if (message.action === 'llm') {
                                assistantMessage.content += message.data;
                                setMessages(prev => 
                                    prev.map(msg => 
                                        msg.id === assistantMessage.id 
                                            ? { ...msg, content: assistantMessage.content }
                                            : msg
                                    )
                                );
                            } else if (message.action === 'visualization') {
                                assistantMessage.visualizations.push(message.data);
                                setVisualizations(prev => [...prev, message.data]);
                            }
                        } catch (e) {
                            console.warn('Failed to parse stream message:', line);
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Streaming error:', error);
        }
    };
    
    const createVisualization = (visualizationData) => {
        const frontendData = visualizationData.frontend_data;
        const ctx = document.getElementById(`chart-${Date.now()}`).getContext('2d');
        
        const config = JSON.parse(JSON.stringify(frontendData.config));
        
        // Add actual data (you'll need to extract this from tool responses)
        config.data.labels = ['Urban Life', 'Natural Soundscapes'];
        config.data.datasets[0].data = [1, 1];
        
        new Chart(ctx, config);
    };
    
    return (
        <div className="audio-data-chat">
            <div className="messages">
                {messages.map(message => (
                    <div key={message.id} className={`message ${message.type}`}>
                        <div className="content">{message.content}</div>
                        {message.visualizations?.map((viz, index) => (
                            <div key={index} className="visualization">
                                <h4>{viz.visualization_name}</h4>
                                <canvas id={`chart-${message.id}-${index}`}></canvas>
                            </div>
                        ))}
                    </div>
                ))}
            </div>
            
            <div className="input-area">
                <input
                    type="text"
                    value={currentMessage}
                    onChange={(e) => setCurrentMessage(e.target.value)}
                    onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
                    placeholder="Ask about your audio data..."
                    disabled={isLoading}
                />
                <button onClick={sendMessage} disabled={isLoading}>
                    {isLoading ? 'Sending...' : 'Send'}
                </button>
            </div>
        </div>
    );
};

export default AudioDataChat;
```

## 🚀 Getting Started

1. **Set up authentication** with your backend
2. **Include Chart.js** in your project
3. **Implement the AudioDataChat class** or React component
4. **Test with sample queries** like:
   - "Show me the distribution of audio files by category"
   - "Compare decibel levels across different devices"
   - "Analyze noise trends over time"

## 📚 Additional Resources

- [Chart.js Documentation](https://www.chartjs.org/docs/)
- [Fetch API Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)
- [Streaming Responses Guide](https://developer.mozilla.org/en-US/docs/Web/API/ReadableStream)

## 🆘 Support

For technical support or questions about the API, please contact the development team or refer to the backend documentation.

---

**Happy coding! 🎵📊**