"""LangChain chain builders for intent, clarification, recommendation, and compare."""

from __future__ import annotations

import os
from typing import Any

from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from prompts import (
    CLARIFY_PROMPT,
    COMPARE_PROMPT,
    INTENT_CLASSIFIER_PROMPT,
    RECOMMEND_PROMPT,
)


def build_llm() -> ChatGoogleGenerativeAI:
    """Create shared Gemini 2.0 Flash chat model instance."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.2,
        google_api_key=api_key,
        convert_system_message_to_human=True,
    )


def build_intent_chain(llm: ChatGoogleGenerativeAI) -> LLMChain:
    """Build LLMChain for one-word intent classification."""
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["conversation"],
            template=INTENT_CLASSIFIER_PROMPT
        ),
        verbose=False,
    )


def build_clarify_chain(llm: ChatGoogleGenerativeAI) -> LLMChain:
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
    llm: ChatGoogleGenerativeAI, retriever: Any
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


def build_compare_chain(llm: ChatGoogleGenerativeAI) -> LLMChain:
    """Build compare response chain using catalog data only."""
    return LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["conversation", "assessment_data"],
            template=COMPARE_PROMPT
        ),
        verbose=False,
    )