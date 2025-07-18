#!/usr/bin/env python3
"""
RAG Pipeline Debug Script
=========================

This script debugs the RAG pipeline step by step to identify where documents are being filtered out.
"""

import os
import sys
from dotenv import load_dotenv
load_dotenv()

# Add the parent directory to the path to import healraglib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from healraglib import RAGManager, LLMManager, SearchIndexManager

def debug_rag_pipeline():
    """Debug the RAG pipeline step by step."""
    
    print("🔍 RAG Pipeline Debug")
    print("=" * 50)
    
    # Configuration
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
    azure_openai_chat_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    azure_openai_embedding_deployment = os.getenv("AZURE_TEXT_EMBEDDING_MODEL")
    azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    azure_search_key = os.getenv("AZURE_SEARCH_KEY")
    azure_search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "security-index")
    
    # Test query
    test_query = "what do i do if laptop is lost"
    print(f"🔍 Testing query: '{test_query}'")
    
    # Initialize components
    print("   🔧 Initializing components...")
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
        default_top_k=10,
        max_context_tokens=6000,
        relevance_threshold=0.005
    )
    
    print("   ✅ Components initialized")
    
    # Step 1: Test document retrieval
    print(f"\n📚 Step 1: Document Retrieval")
    retrieved_docs = rag_manager._retrieve_documents(test_query, 10)
    print(f"   Documents retrieved: {len(retrieved_docs)}")
    
    if retrieved_docs:
        print(f"   First document score: {retrieved_docs[0].get('score', 0):.6f}")
        print(f"   Last document score: {retrieved_docs[-1].get('score', 0):.6f}")
        print(f"   Relevance threshold: {rag_manager.relevance_threshold}")
        
        # Check which documents pass threshold
        passing_docs = [doc for doc in retrieved_docs if doc.get('score', 0) >= rag_manager.relevance_threshold]
        print(f"   Documents passing threshold: {len(passing_docs)}")
        
        if len(passing_docs) != len(retrieved_docs):
            print(f"   ⚠️  Some documents filtered out by threshold!")
            for i, doc in enumerate(retrieved_docs):
                score = doc.get('score', 0)
                passes = score >= rag_manager.relevance_threshold
                print(f"     Doc {i+1}: Score={score:.6f}, Passes={passes}")
    else:
        print("   ❌ No documents retrieved")
        return
    
    # Step 2: Test context building
    print(f"\n🔧 Step 2: Context Building")
    context, sources = rag_manager._build_context(retrieved_docs, test_query)
    print(f"   Context length: {len(context)} characters")
    print(f"   Sources built: {len(sources)}")
    
    if sources:
        print(f"   First source file: {sources[0].get('source_file', 'Unknown')}")
        print(f"   First source score: {sources[0].get('score', 0):.6f}")
        print(f"   Context preview: {context[:200]}...")
    else:
        print("   ❌ No sources built!")
        
        # Debug token estimation
        print(f"   🔬 Debugging token estimation...")
        for i, doc in enumerate(retrieved_docs[:3]):
            content = doc.get('content', '')
            estimated_tokens = rag_manager.llm_manager.estimate_tokens(content)
            print(f"     Doc {i+1}: {estimated_tokens} tokens, content length: {len(content)}")
            
            # Check if it would exceed token limit
            if estimated_tokens > rag_manager.max_context_tokens:
                print(f"     ⚠️  Doc {i+1} exceeds token limit!")
    
    # Step 3: Test system message creation
    print(f"\n💬 Step 3: System Message Creation")
    system_message = rag_manager._create_rag_system_message(context, None, None)
    print(f"   System message length: {len(system_message)} characters")
    print(f"   System message preview: {system_message[:200]}...")
    
    # Step 4: Test full RAG response
    print(f"\n🎯 Step 4: Full RAG Response")
    rag_response = rag_manager.generate_rag_response(
        query=test_query,
        top_k=10,
        include_search_details=True
    )
    
    print(f"   Success: {rag_response.get('success', 'N/A')}")
    print(f"   Sources in response: {len(rag_response.get('sources', []))}")
    print(f"   Data points: {len(rag_response.get('context', {}).get('data_points', []))}")
    
    if 'metadata' in rag_response:
        metadata = rag_response['metadata']
        if 'retrieval' in metadata:
            retrieval = metadata['retrieval']
            print(f"   Documents found: {retrieval.get('documents_found', 'N/A')}")
            print(f"   Documents used: {retrieval.get('documents_used', 'N/A')}")
    
    if 'search_details' in rag_response:
        search_details = rag_response['search_details']
        print(f"   Retrieved documents in details: {len(search_details.get('retrieved_documents', []))}")
        print(f"   Context used length: {len(search_details.get('context_used', ''))}")
    
    # Step 5: Test with different thresholds
    print(f"\n🔬 Step 5: Testing Different Thresholds")
    for threshold in [0.0, 0.001, 0.005, 0.01, 0.02, 0.03]:
        rag_manager.relevance_threshold = threshold
        test_docs = rag_manager._retrieve_documents(test_query, 10)
        passing_docs = [doc for doc in test_docs if doc.get('score', 0) >= threshold]
        print(f"   Threshold {threshold:.3f}: {len(passing_docs)}/{len(test_docs)} documents pass")
    
    # Reset threshold
    rag_manager.relevance_threshold = 0.005
    
    print(f"\n✅ Debug complete!")

if __name__ == "__main__":
    debug_rag_pipeline() 