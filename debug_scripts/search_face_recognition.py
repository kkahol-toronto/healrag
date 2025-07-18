#!/usr/bin/env python3
"""
Search for "face recognition" in the indexed content
===================================================

This script demonstrates searching for "face recognition" using the
now-working vector search system with text-embedding-ada-002.
"""

import os
import json
from typing import List, Dict, Any

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import HEALRAG components
try:
    from healraglib import SearchIndexManager, StorageManager, RAGManager, LLMManager
    HEALRAG_AVAILABLE = True
except ImportError:
    print("❌ HEALRAG library not available")
    HEALRAG_AVAILABLE = False

def search_face_recognition():
    """Search for face recognition related content."""
    
    if not HEALRAG_AVAILABLE:
        return False
    
    print("🔍 Searching for 'face recognition'")
    print("=" * 50)
    
    # Get configuration
    azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    azure_openai_key = os.getenv('AZURE_OPENAI_KEY')
    azure_text_embedding_model = os.getenv('AZURE_TEXT_EMBEDDING_MODEL')
    azure_search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
    azure_search_key = os.getenv('AZURE_SEARCH_KEY')
    azure_search_index_name = os.getenv('AZURE_SEARCH_INDEX_NAME')
    azure_storage_connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    azure_container_name = os.getenv('AZURE_CONTAINER_NAME')
    
    print(f"🔧 Configuration:")
    print(f"   Embedding Model: {azure_text_embedding_model}")
    print(f"   Search Index: {azure_search_index_name}")
    print(f"   Storage Container: {azure_container_name}")
    
    if not all([azure_openai_endpoint, azure_openai_key, azure_text_embedding_model]):
        print("\n❌ Missing Azure OpenAI configuration!")
        return False
    
    if not all([azure_search_endpoint, azure_search_key, azure_search_index_name]):
        print("\n❌ Missing Azure Search configuration!")
        return False
    
    if not all([azure_storage_connection_string, azure_container_name]):
        print("\n❌ Missing Azure Storage configuration!")
        return False
    
    try:
        # Initialize components
        print(f"\n🔧 Initializing components...")
        
        storage_manager = StorageManager(
            connection_string=azure_storage_connection_string,
            container_name=azure_container_name
        )
        print(f"   ✅ Storage Manager initialized")
        
        search_manager = SearchIndexManager(
            storage_manager=storage_manager,
            azure_openai_endpoint=azure_openai_endpoint,
            azure_openai_key=azure_openai_key,
            azure_openai_deployment=azure_text_embedding_model,
            azure_search_endpoint=azure_search_endpoint,
            azure_search_key=azure_search_key,
            azure_search_index_name=azure_search_index_name
        )
        print(f"   ✅ Search Index Manager initialized")
        
        # Initialize LLM Manager for RAG
        llm_manager = LLMManager(
            azure_openai_endpoint=azure_openai_endpoint,
            azure_openai_key=azure_openai_key,
            azure_openai_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
            default_temperature=0.7,
            default_max_tokens=500
        )
        print(f"   ✅ LLM Manager initialized")
        
        # Initialize RAG Manager
        rag_manager = RAGManager(
            search_index_manager=search_manager,
            llm_manager=llm_manager,
            default_top_k=5,
            max_context_tokens=6000,
            relevance_threshold=0.005
        )
        print(f"   ✅ RAG Manager initialized")
        
        # Test 1: Vector Search
        print(f"\n🔍 Test 1: Vector Search for 'face recognition'")
        print("-" * 40)
        
        try:
            results = search_manager.search_similar_chunks("face recognition", top_k=10)
            print(f"   ✅ Vector search successful!")
            print(f"   Found {len(results)} results")
            
            print(f"\n📋 Top Results:")
            for i, result in enumerate(results[:5]):  # Show top 5 results
                print(f"\n   Result {i+1}:")
                print(f"     Score: {result.get('@search.score', 'N/A')}")
                print(f"     Source: {result.get('source_file', 'N/A')}")
                print(f"     Section: {result.get('section', 'N/A')}")
                content = result.get('content', 'N/A')
                if len(content) > 200:
                    content = content[:200] + "..."
                print(f"     Content: {content}")
                
        except Exception as e:
            print(f"   ❌ Error in vector search: {e}")
            return False
        
        # Test 2: RAG Question Answering
        print(f"\n🤖 Test 2: RAG Question Answering")
        print("-" * 40)
        
        questions = [
            "What is face recognition?",
            "How does face recognition work?",
            "What are the security implications of face recognition?",
            "What technologies are used in face recognition systems?"
        ]
        
        for question in questions:
            print(f"\n❓ Question: {question}")
            try:
                # Get streaming response using the correct method name
                response_stream = rag_manager.generate_streaming_rag_response(question)
                
                print(f"   🤖 Answer:")
                full_response = ""
                sources = []
                for chunk in response_stream:
                    if chunk.get('type') == 'chunk' and chunk.get('content'):
                        content = chunk['content']
                        print(f"      {content}", end='', flush=True)
                        full_response += content
                    elif chunk.get('type') == 'context_ready':
                        sources = chunk.get('sources', [])
                
                print(f"\n   📊 Response Stats:")
                print(f"      Total length: {len(full_response)} characters")
                print(f"      Sources used: {len(sources)}")
                
                # Show sources
                if sources:
                    print(f"      📚 Sources:")
                    for j, source in enumerate(sources[:3]):  # Show top 3 sources
                        print(f"         {j+1}. {source.get('source_file', 'Unknown')}")
                        print(f"            Score: {source.get('score', 'N/A')}")
                
            except Exception as e:
                print(f"   ❌ Error in RAG: {e}")
        
        # Test 3: Search Statistics
        print(f"\n📊 Test 3: Search Statistics")
        print("-" * 40)
        
        try:
            # Get all documents to analyze
            all_results = search_manager.search_similar_chunks("", top_k=1000)  # Get many results
            
            # Analyze file types
            file_types = {}
            face_recognition_files = []
            
            for result in all_results:
                source_file = result.get('source_file', '')
                if source_file:
                    # Extract file extension
                    if '.' in source_file:
                        ext = '.' + source_file.split('.')[-1]
                        file_types[ext] = file_types.get(ext, 0) + 1
                    
                    # Check if this file contains face recognition content
                    content = result.get('content', '').lower()
                    if 'face' in content and 'recognition' in content:
                        face_recognition_files.append(source_file)
            
            print(f"   📁 File Type Distribution:")
            for ext, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
                print(f"      {ext}: {count} documents")
            
            print(f"\n   🎯 Face Recognition Related Files:")
            unique_files = list(set(face_recognition_files))
            for i, file in enumerate(unique_files[:10]):  # Show top 10
                print(f"      {i+1}. {file}")
            
            if len(unique_files) > 10:
                print(f"      ... and {len(unique_files) - 10} more files")
            
        except Exception as e:
            print(f"   ❌ Error in statistics: {e}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during search: {e}")
        return False

if __name__ == "__main__":
    success = search_face_recognition()
    print(f"\n{'✅' if success else '❌'} Search completed")
    
    if success:
        print("\n🎉 Face recognition search completed successfully!")
        print("The system is working correctly with the fixed embedding model.")
    else:
        print("\n⚠️  Search encountered issues. Check the error messages above.") 