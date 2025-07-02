"""
HEALRAG - Health Enterprise Azure Language Retrieval-Augmented Generation

A comprehensive library for document processing, search indexing, and RAG applications
using Azure services including Blob Storage, Cognitive Search, and OpenAI.
"""

__version__ = "1.0.0"
__author__ = "Kanav Kahol"
__description__ = "Health Enterprise Azure Language RAG Library"

# Core components
from .storage_manager import StorageManager
from .content_manager import ContentManager
from .search_index_manager import SearchIndexManager
from .llm_manager import LLMManager
from .rag_manager import RAGManager

# Convenience imports
__all__ = [
    "StorageManager",
    "ContentManager", 
    "SearchIndexManager",
    "LLMManager",
    "RAGManager"
]

def get_version():
    """Get the current version of HEALRAG."""
    return __version__

def get_components():
    """Get list of available HEALRAG components."""
    return __all__

def create_full_rag_system(azure_storage_connection_string: str,
                          container_name: str,
                          azure_openai_endpoint: str,
                          azure_openai_key: str,
                          azure_openai_chat_deployment: str,
                          azure_openai_embedding_deployment: str,
                          azure_search_endpoint: str,
                          azure_search_key: str,
                          azure_search_index_name: str = "healrag-index"):
    """
    Create a complete HEALRAG system with all components configured.
    
    Args:
        azure_storage_connection_string: Azure Storage connection string
        container_name: Blob container name
        azure_openai_endpoint: Azure OpenAI endpoint
        azure_openai_key: Azure OpenAI API key
        azure_openai_chat_deployment: Chat model deployment (e.g., gpt-4, gpt-35-turbo)
        azure_openai_embedding_deployment: Embedding model deployment (e.g., text-embedding-ada-002)
        azure_search_endpoint: Azure Cognitive Search endpoint
        azure_search_key: Azure Cognitive Search API key
        azure_search_index_name: Search index name
        
    Returns:
        Dict containing all configured HEALRAG components
    """
    
    print("🚀 Initializing Complete HEALRAG System...")
    
    # Initialize Storage Manager
    print("   📦 Initializing Storage Manager...")
    storage_manager = StorageManager(
        connection_string=azure_storage_connection_string,
        container_name=container_name
    )
    
    # Initialize Content Manager
    print("   📚 Initializing Content Manager...")
    content_manager = ContentManager(
        storage_manager=storage_manager,
        azure_openai_endpoint=azure_openai_endpoint,
        azure_openai_key=azure_openai_key,
        azure_openai_deployment=azure_openai_chat_deployment  # Use chat model for image analysis
    )
    
    # Initialize Search Index Manager
    print("   🔍 Initializing Search Index Manager...")
    search_index_manager = SearchIndexManager(
        storage_manager=storage_manager,
        azure_openai_endpoint=azure_openai_endpoint,
        azure_openai_key=azure_openai_key,
        azure_openai_deployment=azure_openai_embedding_deployment,  # Use embedding model
        azure_search_endpoint=azure_search_endpoint,
        azure_search_key=azure_search_key,
        azure_search_index_name=azure_search_index_name
    )
    
    # Initialize LLM Manager
    print("   🤖 Initializing LLM Manager...")
    llm_manager = LLMManager(
        azure_openai_endpoint=azure_openai_endpoint,
        azure_openai_key=azure_openai_key,
        azure_openai_deployment=azure_openai_chat_deployment  # Use chat model
    )
    
    # Initialize RAG Manager
    print("   🎯 Initializing RAG Manager...")
    rag_manager = RAGManager(
        search_index_manager=search_index_manager,
        llm_manager=llm_manager
    )
    
    print("   ✅ HEALRAG system initialized successfully!")
    
    return {
        "storage_manager": storage_manager,
        "content_manager": content_manager,
        "search_index_manager": search_index_manager,
        "llm_manager": llm_manager,
        "rag_manager": rag_manager
    }

def create_rag_from_existing_index(azure_openai_endpoint: str,
                                  azure_openai_key: str,
                                  azure_openai_chat_deployment: str,
                                  azure_openai_embedding_deployment: str,
                                  azure_search_endpoint: str,
                                  azure_search_key: str,
                                  azure_search_index_name: str = "healrag-index"):
    """
    Create a RAG system from an existing search index (without storage/content managers).
    
    Args:
        azure_openai_endpoint: Azure OpenAI endpoint
        azure_openai_key: Azure OpenAI API key
        azure_openai_chat_deployment: Chat model deployment
        azure_openai_embedding_deployment: Embedding model deployment
        azure_search_endpoint: Azure Cognitive Search endpoint
        azure_search_key: Azure Cognitive Search API key
        azure_search_index_name: Search index name
        
    Returns:
        Dict containing RAG components
    """
    
    print("🎯 Initializing RAG System from Existing Index...")
    
    # Create minimal search index manager (without storage manager)
    print("   🔍 Initializing Search Index Manager...")
    search_index_manager = SearchIndexManager(
        storage_manager=None,  # Not needed for searching existing index
        azure_openai_endpoint=azure_openai_endpoint,
        azure_openai_key=azure_openai_key,
        azure_openai_deployment=azure_openai_embedding_deployment,
        azure_search_endpoint=azure_search_endpoint,
        azure_search_key=azure_search_key,
        azure_search_index_name=azure_search_index_name
    )
    
    # Initialize LLM Manager
    print("   🤖 Initializing LLM Manager...")
    llm_manager = LLMManager(
        azure_openai_endpoint=azure_openai_endpoint,
        azure_openai_key=azure_openai_key,
        azure_openai_deployment=azure_openai_chat_deployment
    )
    
    # Initialize RAG Manager
    print("   🎯 Initializing RAG Manager...")
    rag_manager = RAGManager(
        search_index_manager=search_index_manager,
        llm_manager=llm_manager
    )
    
    print("   ✅ RAG system initialized successfully!")
    
    return {
        "search_index_manager": search_index_manager,
        "llm_manager": llm_manager,
        "rag_manager": rag_manager
    }