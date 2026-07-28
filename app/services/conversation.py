import threading
from datetime import datetime, timedelta

from ..config import SESSION_TIMEOUT


class ConversationStore:
    def __init__(self):
        self.sessions = {}
        self.lock = threading.Lock()

    def update_context(self, session_id: str, document_id: str):
        with self.lock:
            self.sessions[session_id] = {
                "document_id": document_id,
                "last_updated": datetime.utcnow(),
            }

    def get_document(self, session_id: str):
        with self.lock:
            session = self.sessions.get(session_id)
            if not session:
                return None
            if datetime.utcnow() - session["last_updated"] > timedelta(seconds=SESSION_TIMEOUT):
                del self.sessions[session_id]
                return None
            return session["document_id"]
