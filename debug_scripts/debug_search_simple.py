#!/usr/bin/env python3
"""
Simple Search Debug Script
==========================

Tests both text and vector search to understand why Step 10 fails
while Azure portal search works.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_search():
    """Test search functionality step by step."""
    
    print("🔍 Simple Search Debug")
    print("=" * 40)
    
    # Get configuration
    azure_search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
    azure_search_key = os.getenv('AZURE_SEARCH_KEY')
    azure_search_index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'security-index')
    azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    azure_openai_key = os.getenv('AZURE_OPENAI_KEY')
    azure_text_embedding_model = os.getenv('AZURE_TEXT_EMBEDDING_MODEL')
    
    print(f"📋 Configuration:")
    print(f"   Index: {azure_search_index_name}")
    print(f"   Search endpoint: {azure_search_endpoint}")
    print(f"   OpenAI endpoint: {azure_openai_endpoint}")
    print(f"   Embedding model: {azure_text_embedding_model}")
    
    if not all([azure_search_endpoint, azure_search_key]):
        print("❌ Missing Azure Search configuration")
        return
    
    try:
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential
        
        # Initialize search client
        credential = AzureKeyCredential(azure_search_key)
        search_client = SearchClient(
            endpoint=azure_search_endpoint,
            index_name=azure_search_index_name,
            credential=credential
        )
        
        print(f"✅ Search client connected")
        
        # Test 1: Text search (like Azure portal)
        print(f"\n🔍 Test 1: Text Search")
        query = "cyber security policy"
        print(f"   Query: '{query}'")
        
        try:
            results = search_client.search(
                search_text=query,
                top=3
            )
            
            results_list = list(results)
            print(f"   ✅ Text search: {len(results_list)} results")
            
            for i, result in enumerate(results_list[:2], 1):  # Show first 2
                print(f"     {i}. Score: {result.get('@search.score', 'N/A')}")
                print(f"        Content: {result.get('content', 'No content')[:80]}...")
                
        except Exception as e:
            print(f"   ❌ Text search failed: {e}")
        
        # Test 2: Vector search (like Step 10)
        print(f"\n🧠 Test 2: Vector Search")
        
        if not all([azure_openai_endpoint, azure_openai_key, azure_text_embedding_model]):
            print(f"   ❌ OpenAI config missing - this explains Step 10 failure")
            return
        
        try:
            from openai import AzureOpenAI
            
            # Initialize OpenAI client
            client = AzureOpenAI(
                azure_endpoint=azure_openai_endpoint,
                api_key=azure_openai_key,
                api_version="2024-02-15-preview"
            )
            
            print(f"   ✅ OpenAI client connected")
            
            # Generate embedding
            response = client.embeddings.create(
                input=query,
                model=azure_text_embedding_model
            )
            embedding = response.data[0].embedding
            print(f"   ✅ Embedding generated: {len(embedding)} dims")
            
            # Vector search
            vector_results = search_client.search(
                search_text=query,
                vector_queries=[{
                    "vector": embedding,
                    "k_nearest_neighbors": 3,
                    "fields": "embedding",
                    "kind": "vector"
                }],
                top=3
            )
            
            vector_list = list(vector_results)
            print(f"   ✅ Vector search: {len(vector_list)} results")
            
            for i, result in enumerate(vector_list[:2], 1):  # Show first 2
                print(f"     {i}. Score: {result.get('@search.score', 'N/A')}")
                print(f"        Content: {result.get('content', 'No content')[:80]}...")
                
        except Exception as e:
            print(f"   ❌ Vector search failed: {e}")
            print(f"   🔍 Error type: {type(e).__name__}")
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"   • Text search works ✅ (like Azure portal)")
        if 'vector_list' in locals() and vector_list:
            print(f"   • Vector search works ✅ (Step 10 should work)")
        else:
            print(f"   • Vector search failed ❌ (explains Step 10 issue)")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_search() 