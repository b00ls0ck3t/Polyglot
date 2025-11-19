#!/usr/bin/env python3
"""
Polyglot - Real-Time Speech Translation Service
FastAPI + WebSocket + DeepL API
"""

import asyncio
import json
import os
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

# Get DeepL API key from environment
DEEPL_API_KEY = os.environ.get('DEEPL_API_KEY')

# Data models
@dataclass
class TranslationEntry:
    timestamp: str
    czech_text: str
    english_text: str
    speaker: Optional[str] = None
    processing_time: Optional[float] = None
    
    def to_dict(self):
        return asdict(self)


class TranslationManager:
    def __init__(self):
        self.history = deque(maxlen=MAX_HISTORY)
        self.clients = set()
        self.deepl_api_key = DEEPL_API_KEY
        self.translation_count = 0
        self.total_translation_time = 0.0
    
    async def translate_text(self, text: str) -> tuple[str, float]:
        """Translate Czech text to English using DeepL API. Returns (translation, time_ms)"""
        import time
        start = time.time()
        
        if not self.deepl_api_key:
            return "[Translation unavailable - Set DEEPL_API_KEY environment variable]", 0
        
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
                
                elapsed = (time.time() - start) * 1000  # Convert to ms
                
                if response.status_code == 200:
                    result = response.json()
                    translation = result["translations"][0]["text"]
                    self.translation_count += 1
                    self.total_translation_time += elapsed
                    return translation, elapsed
                elif response.status_code == 403:
                    return "[Translation failed - Invalid API key]", elapsed
                elif response.status_code == 456:
                    return "[Translation failed - Quota exceeded]", elapsed
                else:
                    return f"[Translation failed - Error {response.status_code}]", elapsed
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            return f"[Translation error: {str(e)}]", elapsed
    
    async def process_input(self, czech_text: str, speaker: Optional[str] = None):
        """Process new Czech input and translate it"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        english_text, trans_time = await self.translate_text(czech_text)
        
        entry = TranslationEntry(
            timestamp=timestamp,
            czech_text=czech_text,
            english_text=english_text,
            speaker=speaker,
            processing_time=trans_time
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
    
    def get_stats(self):
        """Get telemetry stats"""
        avg_time = self.total_translation_time / self.translation_count if self.translation_count > 0 else 0
        return {
            "total_translations": self.translation_count,
            "avg_translation_time_ms": round(avg_time, 1)
        }


# Initialize FastAPI app and manager
app = FastAPI(title="Czech-to-English Translation Demo")
manager = TranslationManager()


# Simplified HTML Frontend
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Polyglot - Real-Time Translation</title>
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
            gap: 15px;
            font-size: 12px;
        }
        
        .status-item {
            display: flex;
            align-items: center;
            gap: 8px;
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
        
        .telemetry {
            color: #666;
            font-size: 11px;
        }
        
        .telemetry-value {
            color: #00ff00;
            font-weight: bold;
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
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .pane-header.czech {
            color: #00ccff;
        }
        
        .pane-header.english {
            color: #ffcc00;
        }
        
        .pane-stats {
            font-size: 11px;
            color: #666;
            font-weight: normal;
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
            border-bottom: 1px solid #1a1a1a;
        }
        
        .entry:last-child {
            border-bottom: none;
        }
        
        .entry.pending {
            opacity: 0.6;
            font-style: italic;
        }
        
        .entry-header {
            display: flex;
            gap: 10px;
            margin-bottom: 5px;
            font-size: 11px;
            opacity: 0.6;
        }
        
        .timestamp {
            color: #666;
        }
        
        .speaker {
            color: #ff6600;
            font-weight: bold;
        }
        
        .processing-time {
            color: #666;
            margin-left: auto;
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
    </style>
</head>
<body>
    <div class="header">
        <div class="title">POLYGLOT - REAL-TIME TRANSLATION</div>
        <div class="status">
            <div class="status-item">
                <div class="status-indicator" id="statusIndicator"></div>
                <span id="statusText">Disconnected</span>
            </div>
            <div class="telemetry">
                <span id="telemetry">0 translations | 0ms avg</span>
            </div>
        </div>
    </div>
    
    <div class="container">
        <div class="pane">
            <div class="pane-header czech">
                <span>CZECH INPUT</span>
                <span class="pane-stats" id="czechStats">0 entries</span>
            </div>
            <div class="pane-content" id="czechPane"></div>
        </div>
        
        <div class="pane">
            <div class="pane-header english">
                <span>ENGLISH TRANSLATION</span>
                <span class="pane-stats" id="englishStats">0 entries</span>
            </div>
            <div class="pane-content" id="englishPane"></div>
        </div>
    </div>
    
    <script>
        let ws = null;
        let reconnectInterval = null;
        let entryCount = 0;
        let totalTransTime = 0;
        
        function connect() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
            
            ws.onopen = () => {
                console.log('WebSocket connected');
                updateStatus(true);
                clearInterval(reconnectInterval);
                
                // Request history
                ws.send(JSON.stringify({ type: 'get_history' }));
                
                // Request stats
                ws.send(JSON.stringify({ type: 'get_stats' }));
            };
            
            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                
                if (message.type === 'transcription') {
                    // Real-time Czech transcription (no translation yet)
                    addCzechOnly(message.data);
                } else if (message.type === 'translation') {
                    // Full translation (batched)
                    addEntry(message.data);
                } else if (message.type === 'history') {
                    clearDisplay();
                    message.data.forEach(entry => addEntry(entry));
                } else if (message.type === 'stats') {
                    updateTelemetry(message.data);
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
        
        function updateTelemetry(stats) {
            const tel = document.getElementById('telemetry');
            tel.textContent = `${stats.total_translations} translations | ${stats.avg_translation_time_ms}ms avg`;
        }
        
        function addCzechOnly(data) {
            // Add real-time Czech transcription (no English yet)
            const czechPane = document.getElementById('czechPane');
            
            const czechEntry = document.createElement('div');
            czechEntry.className = 'entry pending';
            czechEntry.setAttribute('data-timestamp', data.timestamp);
            czechEntry.setAttribute('data-speaker', data.speaker || '');
            czechEntry.innerHTML = `
                <div class="entry-header">
                    <span class="timestamp">[${data.timestamp}]</span>
                    ${data.speaker ? `<span class="speaker speaker-${getSpeakerColor(data.speaker)}">${data.speaker}</span>` : ''}
                    <span class="processing-time" style="color: #666;">[pending translation]</span>
                </div>
                <div class="entry-text czech-text">${escapeHtml(data.czech_text)}</div>
            `;
            czechPane.appendChild(czechEntry);
            czechPane.scrollTop = czechPane.scrollHeight;
            
            // Update stats
            const currentCount = czechPane.querySelectorAll('.entry').length;
            document.getElementById('czechStats').textContent = `${currentCount} entries`;
        }
        
        function addEntry(data) {
            entryCount++;
            if (data.processing_time) {
                totalTransTime += data.processing_time;
            }
            
            const czechPane = document.getElementById('czechPane');
            const englishPane = document.getElementById('englishPane');
            
            // Update stats
            document.getElementById('englishStats').textContent = `${entryCount} translations`;
            
            // Check if there are pending Czech entries to replace/consolidate
            // For batched translations, we replace all pending entries with the batched version
            const pendingEntries = czechPane.querySelectorAll('.entry.pending');
            if (pendingEntries.length > 0) {
                // Remove all pending entries - they're now part of this batch
                pendingEntries.forEach(entry => entry.remove());
            }
            
            // Add batched Czech entry
            const czechEntry = document.createElement('div');
            czechEntry.className = 'entry';
            const procTime = data.processing_time ? `${Math.round(data.processing_time)}ms` : '';
            czechEntry.innerHTML = `
                <div class="entry-header">
                    <span class="timestamp">[${data.timestamp}]</span>
                    ${data.speaker ? `<span class="speaker speaker-${getSpeakerColor(data.speaker)}">${data.speaker}</span>` : ''}
                    ${procTime ? `<span class="processing-time">${procTime}</span>` : ''}
                </div>
                <div class="entry-text czech-text">${escapeHtml(data.czech_text)}</div>
            `;
            czechPane.appendChild(czechEntry);
            
            // Add English translation
            const englishEntry = document.createElement('div');
            englishEntry.className = 'entry';
            englishEntry.innerHTML = `
                <div class="entry-header">
                    <span class="timestamp">[${data.timestamp}]</span>
                    ${data.speaker ? `<span class="speaker speaker-${getSpeakerColor(data.speaker)}">${data.speaker}</span>` : ''}
                    ${procTime ? `<span class="processing-time">${procTime}</span>` : ''}
                </div>
                <div class="entry-text english-text">${escapeHtml(data.english_text)}</div>
            `;
            englishPane.appendChild(englishEntry);
            
            // Auto-scroll to bottom
            czechPane.scrollTop = czechPane.scrollHeight;
            englishPane.scrollTop = englishPane.scrollHeight;
            
            // Update Czech stats to reflect non-pending count
            const czechCount = czechPane.querySelectorAll('.entry:not(.pending)').length;
            document.getElementById('czechStats').textContent = `${czechCount} entries`;
        }
        
        function clearDisplay() {
            document.getElementById('czechPane').innerHTML = '';
            document.getElementById('englishPane').innerHTML = '';
            entryCount = 0;
            totalTransTime = 0;
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    manager.clients.add(websocket)
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "transcription":
                # Immediate Czech transcription (no translation yet)
                # Broadcast to all clients for real-time display
                timestamp = datetime.now().strftime("%H:%M:%S")
                broadcast_msg = {
                    "type": "transcription",
                    "data": {
                        "timestamp": timestamp,
                        "czech_text": message["czech_text"],
                        "speaker": message.get("speaker")
                    }
                }
                for client in manager.clients:
                    try:
                        await client.send_text(json.dumps(broadcast_msg))
                    except:
                        pass
            
            elif message["type"] == "translate":
                # Full translation (batched Czech text)
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
            
            elif message["type"] == "get_stats":
                # Send telemetry stats
                await websocket.send_text(json.dumps({
                    "type": "stats",
                    "data": manager.get_stats()
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
    print("Polyglot - Translation Service")
    print("=" * 60)
    print("\nStarting server on http://localhost:8000")
    
    if not DEEPL_API_KEY:
        print("\n⚠ WARNING: DEEPL_API_KEY environment variable not set")
        print("  Translation will not work until you set it:")
        print("  export DEEPL_API_KEY='your_api_key_here'")
    else:
        print("\n✓ DeepL API key configured")
    
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)