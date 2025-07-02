#!/usr/bin/env python3
"""
RAG Context Debug Script
========================

This script helps debug why documents are retrieved but not used in RAG context.
"""

import os
from dotenv import load_dotenv
load_dotenv()

from healraglib import RAGManager, LLMManager, SearchIndexManager

def debug_rag_context():
    """Debug the RAG context building process."""
    
    print("🔍 RAG Context Building Debug")
    print("=" * 50)
    
    # Configuration
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
    azure_openai_chat_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    azure_openai_embedding_deployment = os.getenv("AZURE_TEXT_EMBEDDING_MODEL")
    azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    azure_search_key = os.getenv("AZURE_SEARCH_KEY")
    azure_search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "security-index")
    
    # Initialize components
    search_manager = SearchIndexManager(
        storage_manager=None,
        azure_openai_endpoint=azure_openai_endpoint,
        azure_openai_key=azure_openai_key,
        azure_openai_deployment=azure_openai_embedding_deployment,
        azure_search_endpoint=azure_search_endpoint,
        azure_search_key=azure_search_key,
        azure_search_index_name=azure_search_index_name
    )
    
    llm_manager = LLMManager(
        azure_openai_endpoint=azure_openai_endpoint,
        azure_openai_key=azure_openai_key,
        azure_openai_deployment=azure_openai_chat_deployment,
        default_temperature=0.7,
        default_max_tokens=500
    )
    
    rag_manager = RAGManager(
        search_index_manager=search_manager,
        llm_manager=llm_manager,
        default_top_k=3,
        max_context_tokens=6000,
        relevance_threshold=0.0  # Allow all documents
    )
    
    # Test query
    test_query = "What is our incident management process?"
    print(f"🔍 Testing query: '{test_query}'")
    
    # Step 1: Test direct search
    print(f"\n📋 Step 1: Direct Search Test")
    search_results = search_manager.search_similar_chunks(test_query, 3)
    print(f"   Retrieved {len(search_results)} documents:")
    
    for i, doc in enumerate(search_results):
        print(f"\n   Document {i+1}:")
        print(f"     ID: {doc.get('id', 'N/A')}")
        print(f"     Source: {doc.get('source_file', 'N/A')}")
        print(f"     Section: {doc.get('section', 'N/A')}")
        print(f"     Score: {doc.get('score', 0):.6f}")
        print(f"     Content length: {len(doc.get('content', ''))}")
        print(f"     Content preview: {doc.get('content', '')[:200]}...")
    
    # Step 2: Test context building manually
    print(f"\n🔧 Step 2: Manual Context Building Test")
    
    if search_results:
        # Access the private method for testing (removed force_context parameter)
        context, sources = rag_manager._build_context(search_results, test_query)
        
        print(f"   Context length: {len(context)}")
        print(f"   Sources used: {len(sources)}")
        print(f"   Context preview: {context[:500]}...")
        
        if len(sources) == 0:
            print(f"   ❌ No sources used! Investigating...")
            
            # Test with different relevance thresholds
            print(f"\n   🔬 Testing different relevance thresholds:")
            
            for threshold in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]:
                rag_manager.relevance_threshold = threshold
                _, test_sources = rag_manager._build_context(search_results, test_query)
                print(f"     Threshold {threshold:.3f}: {len(test_sources)} sources")
            
            # Reset threshold
            rag_manager.relevance_threshold = 0.0
        
        # Step 3: Test full RAG - Note: generate_rag_response doesn't have force_context parameter
        print(f"\n🎯 Step 3: Full RAG Test")
        
        rag_response = rag_manager.generate_rag_response(
            query=test_query,
            top_k=3,
            include_search_details=True
        )
        
        print(f"   Success: {rag_response['success']}")
        print(f"   Sources used: {len(rag_response['sources'])}")
        print(f"   Response preview: {rag_response['response'][:300]}...")
        
        if 'search_details' in rag_response:
            context_used = rag_response['search_details']['context_used']
            print(f"   Context actually used length: {len(context_used)}")
    
    else:
        print(f"   ❌ No search results returned")

    # Step 4: Additional debugging - check what's happening in document filtering
    print(f"\n🔬 Step 4: Document Filtering Analysis")
    if search_results:
        print(f"   Original documents: {len(search_results)}")
        
        # Check scores
        scores = [doc.get('score', 0) for doc in search_results]
        print(f"   Score range: {min(scores):.6f} - {max(scores):.6f}")
        print(f"   Current relevance threshold: {rag_manager.relevance_threshold}")
        
        # Check if documents pass threshold
        passing_docs = [doc for doc in search_results if doc.get('score', 0) >= rag_manager.relevance_threshold]
        print(f"   Documents passing threshold: {len(passing_docs)}")
        
        # Check token estimation
        if passing_docs:
            for i, doc in enumerate(passing_docs[:3]):  # Check first 3
                content = doc.get('content', '')
                estimated_tokens = rag_manager.llm_manager.estimate_tokens(content)
                print(f"   Doc {i+1} estimated tokens: {estimated_tokens}")

if __name__ == "__main__":
    debug_rag_context()