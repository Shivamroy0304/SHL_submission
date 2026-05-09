"""Conversation memory reconstruction for stateless API requests."""

from langchain.memory import ConversationBufferMemory

from models import Message


def build_memory_from_messages(messages: list[Message]) -> ConversationBufferMemory:
    """Rebuild ConversationBufferMemory from request messages each call."""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    for msg in messages[:-1]:
        if msg.role == "user":
            memory.chat_memory.add_user_message(msg.content)
        else:
            memory.chat_memory.add_ai_message(msg.content)
    return memory
