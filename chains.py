"""LangChain chain builders for intent, clarification, recommendation, and compare."""

from __future__ import annotations

import os
from typing import Any

from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

from prompts import (
    CLARIFY_PROMPT,
    COMPARE_PROMPT,
    INTENT_CLASSIFIER_PROMPT,
    RECOMMEND_PROMPT,
)


def build_llm() -> ChatGroq:
    """Create shared Groq LLM instance. Fast, free, generous limits."""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def build_intent_chain(llm: ChatGroq) -> LLMChain:
    """Build LLMChain for one-word intent classification."""
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["conversation"],
            template=INTENT_CLASSIFIER_PROMPT
        ),
        verbose=False,
    )


def build_clarify_chain(llm: ChatGroq) -> LLMChain:
    """Build LLMChain that produces exactly one clarifying question."""
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["conversation"],
            template=CLARIFY_PROMPT
        ),
        verbose=False,
    )


def build_recommend_chain(
    llm: ChatGroq, retriever: Any
) -> ConversationalRetrievalChain:
    """Build retrieval-augmented recommendation chain."""
    recommend_template = (
        RECOMMEND_PROMPT.replace("{conversation}", "{chat_history}\nUSER: {question}")
        .replace("{candidates}", "{context}")
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        combine_docs_chain_kwargs={
            "prompt": PromptTemplate(
                input_variables=["context", "question", "chat_history"],
                template=recommend_template,
            )
        },
        return_source_documents=True,
        verbose=False,
    )


def build_compare_chain(llm: ChatGroq) -> LLMChain:
    """Build compare response chain using catalog data only."""
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["conversation", "assessment_data"],
            template=COMPARE_PROMPT
        ),
        verbose=False,
    )