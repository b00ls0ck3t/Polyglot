#!/usr/bin/env python3
"""
Real-time Czech-to-English Translation Demo
FastAPI + WebSocket + DeepL API
"""

import asyncio
import json
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, asdict
from collections import deque

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
import httpx

# Configuration
DEEPL_API_URL = "https://api-free.deepl.com/v2/translate"
MAX_HISTORY = 100

# Data models
@dataclass
class TranslationEntry:
    timestamp: str
    czech_text: str
    english_text: str
    speaker: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)


class TranslationManager:
    def __init__(self):
        self.history = deque(maxlen=MAX_HISTORY)
        self.clients = set()
        self.deepl_api_key: Optional[str] = None
    
    def set_api_key(self, api_key: str):
        self.deepl_api_key = api_key
    
    async def translate_text(self, text: str) -> str:
        """Translate Czech text to English using DeepL API"""
        if not self.deepl_api_key:
            return "[Translation unavailable - API key not set]"
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    DEEPL_API_URL,
                    data={
                        "auth_key": self.deepl_api_key,
                        "text": text,
                        "source_lang": "CS",
                        "target_lang": "EN-US"
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result["translations"][0]["text"]
                elif response.status_code == 403:
                    return "[Translation failed - Invalid API key]"
                elif response.status_code == 456:
                    return "[Translation failed - Quota exceeded]"
                else:
                    return f"[Translation failed - Error {response.status_code}]"
        except Exception as e:
            return f"[Translation error: {str(e)}]"
    
    async def process_input(self, czech_text: str, speaker: Optional[str] = None):
        """Process new Czech input and translate it"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        english_text = await self.translate_text(czech_text)
        
        entry = TranslationEntry(
            timestamp=timestamp,
            czech_text=czech_text,
            english_text=english_text,
            speaker=speaker
        )
        
        self.history.append(entry)
        
        # Broadcast to all connected clients
        await self.broadcast(entry)
        
        return entry
    
    async def broadcast(self, entry: TranslationEntry):
        """Send update to all connected WebSocket clients"""
        message = json.dumps({
            "type": "translation",
            "data": entry.to_dict()
        })
        
        disconnected = set()
        for client in self.clients:
            try:
                await client.send_text(message)
            except:
                disconnected.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected
    
    def get_history(self):
        """Get all translation history"""
        return [entry.to_dict() for entry in self.history]
    
    def clear_history(self):
        """Clear all translation history"""
        self.history.clear()


# Initialize FastAPI app and manager
app = FastAPI(title="Czech-to-English Translation Demo")
manager = TranslationManager()


# HTML Frontend
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Czech → English Translation</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Monaco', 'Menlo', 'Courier New', monospace;
            background: #0a0a0a;
            color: #00ff00;
            height: 100vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .header {
            background: #1a1a1a;
            padding: 15px 20px;
            border-bottom: 2px solid #00ff00;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .title {
            font-size: 18px;
            font-weight: bold;
            color: #00ff00;
        }
        
        .status {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 12px;
        }
        
        .status-indicator {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #ff0000;
            animation: pulse 2s infinite;
        }
        
        .status-indicator.connected {
            background: #00ff00;
            animation: none;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
        
        .container {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .pane {
            flex: 1;
            display: flex;
            flex-direction: column;
            border-bottom: 2px solid #333;
            overflow: hidden;
        }
        
        .pane:last-child {
            border-bottom: none;
        }
        
        .pane-header {
            background: #1a1a1a;
            padding: 10px 20px;
            border-bottom: 1px solid #333;
            font-size: 14px;
            font-weight: bold;
        }
        
        .pane-header.czech {
            color: #00ccff;
        }
        
        .pane-header.english {
            color: #ffcc00;
        }
        
        .pane-content {
            flex: 1;
            overflow-y: auto;
            padding: 15px 20px;
            font-size: 14px;
            line-height: 1.6;
        }
        
        .pane-content::-webkit-scrollbar {
            width: 8px;
        }
        
        .pane-content::-webkit-scrollbar-track {
            background: #1a1a1a;
        }
        
        .pane-content::-webkit-scrollbar-thumb {
            background: #333;
            border-radius: 4px;
        }
        
        .pane-content::-webkit-scrollbar-thumb:hover {
            background: #444;
        }
        
        .entry {
            margin-bottom: 15px;
            padding: 8px 0;
        }
        
        .entry-header {
            display: flex;
            gap: 10px;
            margin-bottom: 5px;
            font-size: 12px;
            opacity: 0.6;
        }
        
        .timestamp {
            color: #666;
        }
        
        .speaker {
            color: #ff6600;
            font-weight: bold;
        }
        
        .speaker-1 { color: #ff6600; }
        .speaker-2 { color: #ff00ff; }
        .speaker-3 { color: #00ffff; }
        
        .entry-text {
            padding-left: 15px;
        }
        
        .czech-text {
            color: #00ccff;
        }
        
        .english-text {
            color: #ffcc00;
        }
        
        .input-section {
            background: #1a1a1a;
            padding: 20px;
            border-top: 2px solid #00ff00;
        }
        
        .input-form {
            display: flex;
            gap: 10px;
            align-items: flex-start;
        }
        
        .input-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
            flex: 1;
        }
        
        label {
            font-size: 11px;
            color: #666;
            text-transform: uppercase;
        }
        
        input[type="text"],
        textarea {
            background: #0a0a0a;
            border: 1px solid #333;
            color: #00ff00;
            padding: 10px;
            font-family: inherit;
            font-size: 14px;
            border-radius: 3px;
        }
        
        textarea {
            resize: vertical;
            min-height: 60px;
        }
        
        input[type="text"]:focus,
        textarea:focus {
            outline: none;
            border-color: #00ff00;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            align-items: flex-end;
        }
        
        button {
            background: #00ff00;
            color: #0a0a0a;
            border: none;
            padding: 12px 20px;
            font-family: inherit;
            font-size: 14px;
            font-weight: bold;
            cursor: pointer;
            border-radius: 3px;
            transition: background 0.2s;
        }
        
        button:hover {
            background: #00cc00;
        }
        
        button:active {
            background: #009900;
        }
        
        button.secondary {
            background: #333;
            color: #00ff00;
        }
        
        button.secondary:hover {
            background: #444;
        }
        
        .api-key-section {
            background: #1a1a1a;
            padding: 15px 20px;
            border-bottom: 1px solid #333;
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .api-key-section input {
            flex: 1;
            max-width: 400px;
        }
        
        .api-key-section button {
            padding: 8px 16px;
        }
        
        .api-status {
            font-size: 12px;
            color: #666;
        }
        
        .api-status.active {
            color: #00ff00;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="title">CZECH → ENGLISH TRANSLATION DEMO</div>
        <div class="status">
            <div class="status-indicator" id="statusIndicator"></div>
            <span id="statusText">Disconnected</span>
        </div>
    </div>
    
    <div class="api-key-section">
        <label>DeepL API Key:</label>
        <input type="password" id="apiKeyInput" placeholder="Enter your DeepL API key...">
        <button onclick="setApiKey()">Set Key</button>
        <span class="api-status" id="apiStatus">Not configured</span>
    </div>
    
    <div class="container">
        <div class="pane">
            <div class="pane-header czech">CZECH INPUT</div>
            <div class="pane-content" id="czechPane"></div>
        </div>
        
        <div class="pane">
            <div class="pane-header english">ENGLISH TRANSLATION</div>
            <div class="pane-content" id="englishPane"></div>
        </div>
    </div>
    
    <div class="input-section">
        <form class="input-form" onsubmit="sendMessage(event)">
            <div class="input-group" style="max-width: 150px;">
                <label for="speakerInput">Speaker (optional)</label>
                <input type="text" id="speakerInput" placeholder="Speaker 1">
            </div>
            <div class="input-group">
                <label for="czechInput">Czech Text</label>
                <textarea id="czechInput" placeholder="Enter Czech text here..." required></textarea>
            </div>
            <div class="button-group">
                <button type="submit">Translate</button>
                <button type="button" class="secondary" onclick="clearHistory()">Clear</button>
            </div>
        </form>
    </div>
    
    <script>
        let ws = null;
        let reconnectInterval = null;
        
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = () => {
                console.log('WebSocket connected');
                updateStatus(true);
                clearInterval(reconnectInterval);
                
                // Request history
                ws.send(JSON.stringify({ type: 'get_history' }));
            };
            
            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                
                if (message.type === 'translation') {
                    addEntry(message.data);
                } else if (message.type === 'history') {
                    clearDisplay();
                    message.data.forEach(entry => addEntry(entry));
                } else if (message.type === 'clear') {
                    clearDisplay();
                }
            };
            
            ws.onclose = () => {
                console.log('WebSocket disconnected');
                updateStatus(false);
                
                // Attempt reconnection
                reconnectInterval = setInterval(() => {
                    console.log('Attempting to reconnect...');
                    connect();
                }, 3000);
            };
            
            ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        }
        
        function updateStatus(connected) {
            const indicator = document.getElementById('statusIndicator');
            const text = document.getElementById('statusText');
            
            if (connected) {
                indicator.classList.add('connected');
                text.textContent = 'Connected';
                text.style.color = '#00ff00';
            } else {
                indicator.classList.remove('connected');
                text.textContent = 'Disconnected';
                text.style.color = '#ff0000';
            }
        }
        
        function addEntry(data) {
            const czechPane = document.getElementById('czechPane');
            const englishPane = document.getElementById('englishPane');
            
            // Czech entry
            const czechEntry = document.createElement('div');
            czechEntry.className = 'entry';
            czechEntry.innerHTML = `
                <div class="entry-header">
                    <span class="timestamp">[${data.timestamp}]</span>
                    ${data.speaker ? `<span class="speaker speaker-${getSpeakerColor(data.speaker)}">${data.speaker}</span>` : ''}
                </div>
                <div class="entry-text czech-text">${escapeHtml(data.czech_text)}</div>
            `;
            czechPane.appendChild(czechEntry);
            
            // English entry
            const englishEntry = document.createElement('div');
            englishEntry.className = 'entry';
            englishEntry.innerHTML = `
                <div class="entry-header">
                    <span class="timestamp">[${data.timestamp}]</span>
                    ${data.speaker ? `<span class="speaker speaker-${getSpeakerColor(data.speaker)}">${data.speaker}</span>` : ''}
                </div>
                <div class="entry-text english-text">${escapeHtml(data.english_text)}</div>
            `;
            englishPane.appendChild(englishEntry);
            
            // Auto-scroll to bottom
            czechPane.scrollTop = czechPane.scrollHeight;
            englishPane.scrollTop = englishPane.scrollHeight;
        }
        
        function clearDisplay() {
            document.getElementById('czechPane').innerHTML = '';
            document.getElementById('englishPane').innerHTML = '';
        }
        
        function getSpeakerColor(speaker) {
            const hash = speaker.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0);
            return (hash % 3) + 1;
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        async function sendMessage(event) {
            event.preventDefault();
            
            const czechInput = document.getElementById('czechInput');
            const speakerInput = document.getElementById('speakerInput');
            
            const message = {
                type: 'translate',
                czech_text: czechInput.value,
                speaker: speakerInput.value || null
            };
            
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify(message));
                czechInput.value = '';
            } else {
                alert('Not connected to server');
            }
        }
        
        async function setApiKey() {
            const apiKey = document.getElementById('apiKeyInput').value;
            
            if (!apiKey) {
                alert('Please enter an API key');
                return;
            }
            
            try {
                const response = await fetch('/api/set-key', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ api_key: apiKey })
                });
                
                if (response.ok) {
                    document.getElementById('apiStatus').textContent = 'API key configured';
                    document.getElementById('apiStatus').classList.add('active');
                    document.getElementById('apiKeyInput').value = '';
                } else {
                    alert('Failed to set API key');
                }
            } catch (error) {
                alert('Error setting API key: ' + error);
            }
        }
        
        function clearHistory() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'clear' }));
            }
        }
        
        // Initialize connection
        connect();
    </script>
</body>
</html>
"""


# Routes
@app.get("/", response_class=HTMLResponse)
async def get_index():
    """Serve the main HTML interface"""
    return HTML_CONTENT


@app.post("/api/set-key")
async def set_api_key(data: dict):
    """Set the DeepL API key"""
    api_key = data.get("api_key")
    if not api_key:
        raise HTTPException(status_code=400, detail="API key required")
    
    manager.set_api_key(api_key)
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    manager.clients.add(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "translate":
                # Process translation request
                await manager.process_input(
                    czech_text=message["czech_text"],
                    speaker=message.get("speaker")
                )
            
            elif message["type"] == "get_history":
                # Send history to client
                await websocket.send_text(json.dumps({
                    "type": "history",
                    "data": manager.get_history()
                }))
            
            elif message["type"] == "clear":
                # Clear history
                manager.clear_history()
                # Broadcast clear to all clients
                for client in manager.clients:
                    try:
                        await client.send_text(json.dumps({"type": "clear"}))
                    except:
                        pass
    
    except WebSocketDisconnect:
        manager.clients.remove(websocket)


if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Czech → English Translation Demo")
    print("=" * 60)
    print("\nStarting server on http://localhost:8000")
    print("\nSetup steps:")
    print("1. Open http://localhost:8000 in your browser")
    print("2. Enter your DeepL API key in the input field")
    print("3. Start translating!")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
