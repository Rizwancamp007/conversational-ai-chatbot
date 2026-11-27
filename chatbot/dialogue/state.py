"""Per-session dialogue state.

One ``DialogueState`` holds everything the bot remembers about a single
conversation: which intent is being handled, which details (slots) have
been collected, which question it is waiting on, and the recent turns.
Sessions live in memory and expire after ``SessionStore``'s TTL (30 minutes
by default) — a stale session is forgotten and the next message starts a
fresh conversation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

DEFAULT_TTL_SECONDS = 30 * 60
MAX_TURNS = 20


def _now():
    return datetime.now(timezone.utc)


@dataclass
class Turn:
    """One message in the conversation, from the user or the bot."""

    role: str  # "user" or "bot"
    text: str
    intent: str = ""
    at: datetime = field(default_factory=_now)

    def to_dict(self):
        return {
            "role": self.role,
            "text": self.text,
            "intent": self.intent,
            "at": self.at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            role=data["role"],
            text=data["text"],
            intent=data.get("intent", ""),
            at=datetime.fromisoformat(data["at"]),
        )


@dataclass
class DialogueState:
    """What the bot remembers about one conversation."""

    session_id: str
    active_intent: str = ""
    slots: dict = field(default_factory=dict)
    pending_slot: str = ""
    last_intent: str = ""
    last_slots: dict = field(default_factory=dict)
    handoff_requested: bool = False
    history: list = field(default_factory=list)
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)

    def touch(self, now=None):
        self.updated_at = now or _now()

    def is_expired(self, ttl_seconds=DEFAULT_TTL_SECONDS, now=None):
        now = now or _now()
        return (now - self.updated_at) > timedelta(seconds=ttl_seconds)

    def add_turn(self, role, text, intent=""):
        self.history.append(Turn(role=role, text=text, intent=intent))
        self.history = self.history[-MAX_TURNS:]
        self.touch()

    def reset_flow(self):
        """Forget the intent being handled, keep the turn history."""
        self.active_intent = ""
        self.slots = {}
        self.pending_slot = ""
        self.touch()

    def finish_flow(self):
        """The active intent was answered: remember it, then clear the flow."""
        self.last_intent = self.active_intent
        self.last_slots = dict(self.slots)
        self.reset_flow()

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "active_intent": self.active_intent,
            "slots": dict(self.slots),
            "pending_slot": self.pending_slot,
            "last_intent": self.last_intent,
            "last_slots": dict(self.last_slots),
            "handoff_requested": self.handoff_requested,
            "history": [t.to_dict() for t in self.history],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            session_id=data["session_id"],
            active_intent=data.get("active_intent", ""),
            slots=dict(data.get("slots", {})),
            pending_slot=data.get("pending_slot", ""),
            last_intent=data.get("last_intent", ""),
            last_slots=dict(data.get("last_slots", {})),
            handoff_requested=data.get("handoff_requested", False),
            history=[Turn.from_dict(t) for t in data.get("history", [])],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )


class SessionStore:
    """In-memory sessions, keyed by session id.

    ``get_or_create`` hands back the live state for a session, or a fresh
    one when the session is new or its old state expired. ``prune`` drops
    expired states; call it now and then so idle sessions do not pile up.
    """

    def __init__(self, ttl_seconds=DEFAULT_TTL_SECONDS):
        self.ttl_seconds = ttl_seconds
        self._sessions = {}

    def get_or_create(self, session_id, now=None):
        now = now or _now()
        state = self._sessions.get(session_id)
        if state is not None and not state.is_expired(self.ttl_seconds, now):
            return state
        state = DialogueState(session_id=session_id, created_at=now,
                              updated_at=now)
        self._sessions[session_id] = state
        return state

    def reset(self, session_id):
        self._sessions.pop(session_id, None)

    def prune(self, now=None):
        now = now or _now()
        expired = [sid for sid, s in self._sessions.items()
                   if s.is_expired(self.ttl_seconds, now)]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    def __len__(self):
        return len(self._sessions)
