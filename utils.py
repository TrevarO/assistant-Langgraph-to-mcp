"""
Utility functions for the LangGraph MCP system.
"""
import os
import logging
from typing import List, Sequence

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

def get_message_text(message: BaseMessage) -> str:
    """Extract text content from a message.
    
    Args:
        message: The message to extract text from
        
    Returns:
        The text content of the message
    """
    if isinstance(message, HumanMessage):
        return message.content
    return str(message.content)

def format_docs(docs: Sequence[Document]) -> str:
    """Format a sequence of documents into a string.
    
    Args:
        docs: Sequence of documents to format
        
    Returns:
        Formatted string representation
    """
    return "\n\n".join(doc.page_content for doc in docs)

def load_chat_model(model_string: str) -> ChatOpenAI:
    """Load a chat model based on a model string.
    
    Args:
        model_string: String in format 'provider/model_name'
        
    Returns:
        Configured ChatOpenAI instance
    """
    try:
        if "/" not in model_string:
            # Default to OpenAI if no provider specified
            provider = "openai"
            model = model_string
        else:
            provider, model = model_string.split("/")
        
        logger.debug(f"Loading chat model: provider={provider}, model={model}")
        
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
                
            chat_model = ChatOpenAI(
                model=model,
                temperature=0,
                api_key=api_key,
                streaming=True,  # Enable streaming for better interaction
                frequency_penalty=0,  # Moved out of model_kwargs
                presence_penalty=0    # Moved out of model_kwargs
            )
            logger.debug(f"Successfully created ChatOpenAI instance with model {model}")
            return chat_model
        else:
            raise ValueError(f"Unsupported model provider: {provider}")
            
    except Exception as e:
        logger.error(f"Error loading chat model: {str(e)}", exc_info=True)
        raise