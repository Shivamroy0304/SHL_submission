"""Pydantic schemas for SHL recommender API."""
from pydantic import BaseModel, field_validator


class Message(BaseModel):
    """Single chat message in the conversation transcript."""

    role: str
    content: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        """Ensure message content is non-empty."""
        if not value or not value.strip():
            raise ValueError("content cannot be empty")
        return value.strip()


class ChatRequest(BaseModel):
    """Incoming chat request containing full stateless history."""

    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, value: list[Message]) -> list[Message]:
        """Validate roles and ensure final message is user-authored."""
        if not value:
            raise ValueError("messages list cannot be empty")
        for msg in value:
            if msg.role not in ("user", "assistant"):
                raise ValueError(f"Invalid role: {msg.role}")
        if value[-1].role != "user":
            raise ValueError("Last message must be from user")
        return value


class Recommendation(BaseModel):
    """Single recommended SHL assessment from catalog."""

    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    """API response for every chat turn."""

    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool

    @field_validator("reply")
    @classmethod
    def validate_reply(cls, value: str) -> str:
        """Ensure reply is non-empty."""
        if not value or not value.strip():
            raise ValueError("reply must be a non-empty string")
        return value.strip()


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
