"""
AIAssistantBridge — sends text notifications to your AI assistant on Windows PC.
Mode:
  "print"  → just print (dev mode)
  "socket" → send to AI assistant listener on Windows PC
  "http"   → HTTP POST to AI assistant API

Phase 2: this bridge will also receive voice commands like "go to kitchen"
"""
import socket
import json
from rover_bridge.vision.utils.logger import RoverLogger


class AIAssistantBridge:
    def __init__(self, settings):
        self.settings = settings
        self.logger   = RoverLogger("AIAssistantBridge")

    def notify(self, message: str):
        """Send a text notification to the AI assistant."""
        self.logger.info(f"→ AI Assistant: {message}")

        if self.settings.AI_BRIDGE_MODE == "print":
            print(f"[AI ASSISTANT] {message}")
            return

        if self.settings.AI_BRIDGE_MODE == "socket":
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.settings.AI_BRIDGE_HOST, self.settings.AI_BRIDGE_PORT))
                s.send(json.dumps({"type": "notify", "message": message}).encode())
                s.close()
            except Exception as e:
                self.logger.error(f"AI bridge socket error: {e}")

    def ask_user(self, question: str) -> str:
        """
        Phase 2: ask the user a question through the AI assistant.
        e.g. "Is this room the kitchen?"
        Returns the user's answer as a string.
        Blocking call — rover should pause while waiting.
        """
        self.notify(f"[QUESTION] {question}")
        # TODO Phase 2: implement response channel (socket listen / HTTP poll)
        return ""
