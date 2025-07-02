#!/usr/bin/env python3
"""
Compatible Debug Search Script
=============================

This script tests your search index with various search terms to verify it works properly.
Compatible with your current Azure Search setup and chunking configuration.
"""

import os
import json
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not available, using system environment variables")
except Exception as e:
    print(f"⚠️  Error loading .env file: {e}")

def test_search_functionality():
    """Test both text and vector search with multiple search terms."""
    
    print("🔍 Compatible Search Index Testing")
    print("=" * 60)
    
    # Get environment variables
    azure_search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
    azure_search_key = os.getenv('AZURE_SEARCH_KEY')
    azure_search_index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'security-index')  # Use your actual index name
    azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    azure_openai_key = os.getenv('AZURE_OPENAI_KEY')
    azure_text_embedding_model = os.getenv('AZURE_TEXT_EMBEDDING_MODEL')
    azure_openai_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')  # Fallback embedding model
    
    print(f"🔧 Configuration Check:")
    print(f"   Search Endpoint: {azure_search_endpoint}")
    print(f"   Search Key: {'***' if azure_search_key else 'Not Set'}")
    print(f"   Index Name: {azure_search_index_name}")
    print(f"   OpenAI Endpoint: {azure_openai_endpoint}")
    print(f"   OpenAI Key: {'***' if azure_openai_key else 'Not Set'}")
    print(f"   Text Embedding Model: {azure_text_embedding_model}")
    print(f"   OpenAI Deployment (fallback): {azure_openai_deployment}")
    
    if not all([azure_search_endpoint, azure_search_key]):
        print("❌ Missing required Azure Search configuration")
        print("💡 Please set AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_KEY")
        return
    
    # Determine embedding model to use
    embedding_model = azure_text_embedding_model or azure_openai_deployment
    if not embedding_model:
        print("⚠️  No embedding model configured - vector search will be skipped")
    
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
        
        print(f"✅ Search client initialized successfully")
        
        # Test 1: Basic index connectivity and document count
        print(f"\n📊 Test 1: Index Status and Document Count")
        try:
            # Get total document count
            count_results = search_client.search(
                search_text="*",
                include_total_count=True,
                top=0  # Don't return documents, just count
            )
            total_docs = count_results.get_count()
            print(f"   ✅ Index connected successfully")
            print(f"   📄 Total documents in index: {total_docs}")
            
            if total_docs == 0:
                print(f"   ❌ Index is empty! No documents to search")
                return
            elif total_docs < 10:
                print(f"   ⚠️  Low document count - expected more chunks")
            else:
                print(f"   ✅ Good document count for testing")
                
        except Exception as e:
            print(f"   ❌ Index connectivity failed: {e}")
            return
        
        # Test 2: Sample document structure
        print(f"\n📋 Test 2: Sample Document Structure")
        try:
            sample_results = search_client.search(
                search_text="*",
                select=["id", "content", "source_file", "section", "chunk_size"],
                top=1
            )
            
            sample_docs = list(sample_results)
            if sample_docs:
                sample_doc = sample_docs[0]
                print(f"   ✅ Sample document retrieved")
                print(f"   📄 Document ID: {sample_doc.get('id', 'N/A')}")
                print(f"   📄 Source File: {sample_doc.get('source_file', 'N/A')}")
                print(f"   📄 Section: {sample_doc.get('section', 'N/A')}")
                print(f"   📄 Chunk Size: {sample_doc.get('chunk_size', 'N/A')} chars")
                print(f"   📄 Content Preview: {sample_doc.get('content', '')[:150]}...")
            else:
                print(f"   ❌ No sample documents found")
                
        except Exception as e:
            print(f"   ❌ Sample document retrieval failed: {e}")
        
        # Test 3: Text search with multiple terms
        print(f"\n🔍 Test 3: Text Search Tests")
        
        # Define test search terms relevant to your cyber security documents
        test_queries = [
            "cyber security policy",
            "incident management", 
            "risk assessment",
            "threat vulnerability",
            "information security",
            "access management",
            "third party security",
            "infrastructure security",
            "asset management",
            "password policy",
            "authentication",
            "data protection"
        ]
        
        successful_searches = 0
        
        for i, query in enumerate(test_queries, 1):
            try:
                print(f"\n   Test 3.{i}: Searching for '{query}'")
                
                text_results = search_client.search(
                    search_text=query,
                    select=["id", "content", "source_file", "section", "chunk_size"],
                    top=3
                )
                
                results_list = list(text_results)
                
                if results_list:
                    print(f"     ✅ Found {len(results_list)} results")
                    successful_searches += 1
                    
                    for j, result in enumerate(results_list, 1):
                        score = result.get('@search.score', 'N/A')
                        source = Path(result.get('source_file', 'Unknown')).name
                        section = result.get('section', 'Unknown')
                        content_preview = result.get('content', '')[:80]
                        
                        print(f"       {j}. {source} (Score: {score})")
                        print(f"          Section: {section}")
                        print(f"          Preview: {content_preview}...")
                else:
                    print(f"     ❌ No results found for '{query}'")
                    
            except Exception as e:
                print(f"     ❌ Search failed for '{query}': {e}")
        
        print(f"\n   📊 Text Search Summary: {successful_searches}/{len(test_queries)} queries successful")
        
        # Test 4: Vector search (if configured)
        if embedding_model and azure_openai_endpoint and azure_openai_key:
            print(f"\n🧠 Test 4: Vector Search Tests")
            
            try:
                from openai import AzureOpenAI
                
                # Initialize OpenAI client
                openai_client = AzureOpenAI(
                    azure_endpoint=azure_openai_endpoint,
                    api_key=azure_openai_key,
                    api_version="2024-02-15-preview"
                )
                
                print(f"   ✅ OpenAI client initialized")
                print(f"   🎯 Using embedding model: {embedding_model}")
                
                # Test vector search with a few key terms
                vector_test_queries = [
                    "cyber security policy",
                    "incident management standard",
                    "risk assessment procedures"
                ]
                
                vector_successful = 0
                
                for i, query in enumerate(vector_test_queries, 1):
                    try:
                        print(f"\n   Test 4.{i}: Vector search for '{query}'")
                        
                        # Generate embedding
                        print(f"     🧠 Generating embedding...")
                        response = openai_client.embeddings.create(
                            input=query,
                            model=embedding_model
                        )
                        query_embedding = response.data[0].embedding
                        print(f"     ✅ Embedding generated: {len(query_embedding)} dimensions")
                        
                        # Perform vector search
                        print(f"     🔍 Performing vector search...")
                        vector_results = search_client.search(
                            search_text=query,
                            vector_queries=[
                                {
                                    "vector": query_embedding,
                                    "k_nearest_neighbors": 3,
                                    "fields": "embedding",
                                    "kind": "vector"
                                }
                            ],
                            select=["id", "content", "source_file", "section"],
                            top=3
                        )
                        
                        vector_results_list = list(vector_results)
                        
                        if vector_results_list:
                            print(f"     ✅ Found {len(vector_results_list)} vector results")
                            vector_successful += 1
                            
                            for j, result in enumerate(vector_results_list, 1):
                                score = result.get('@search.score', 'N/A')
                                source = Path(result.get('source_file', 'Unknown')).name
                                section = result.get('section', 'Unknown')
                                content_preview = result.get('content', '')[:80]
                                
                                print(f"       {j}. {source} (Score: {score})")
                                print(f"          Section: {section}")
                                print(f"          Preview: {content_preview}...")
                        else:
                            print(f"     ❌ No vector results found")
                            
                    except Exception as e:
                        print(f"     ❌ Vector search failed for '{query}': {e}")
                        print(f"     🔍 Error details: {type(e).__name__}")
                
                print(f"\n   📊 Vector Search Summary: {vector_successful}/{len(vector_test_queries)} queries successful")
                
            except Exception as e:
                print(f"   ❌ Vector search setup failed: {e}")
                print(f"   💡 This could be due to:")
                print(f"      - Incorrect embedding model name")
                print(f"      - OpenAI configuration issues")
                print(f"      - Missing embedding field in index")
        else:
            print(f"\n🧠 Test 4: Vector Search - Skipped")
            print(f"   ⚠️  Vector search configuration incomplete:")
            if not embedding_model:
                print(f"      - Missing embedding model (AZURE_TEXT_EMBEDDING_MODEL)")
            if not azure_openai_endpoint:
                print(f"      - Missing OpenAI endpoint")
            if not azure_openai_key:
                print(f"      - Missing OpenAI key")
        
        # Test 5: Index schema validation
        print(f"\n🏗️  Test 5: Index Schema Validation")
        try:
            from azure.search.documents.indexes import SearchIndexClient
            
            index_client = SearchIndexClient(
                endpoint=azure_search_endpoint,
                credential=credential
            )
            
            index = index_client.get_index(azure_search_index_name)
            print(f"   ✅ Index schema retrieved")
            print(f"   📋 Index name: {index.name}")
            print(f"   📋 Total fields: {len(index.fields)}")
            
            # Check key fields
            field_names = [field.name for field in index.fields]
            required_fields = ['id', 'content', 'source_file', 'section', 'embedding']
            
            print(f"   📋 Available fields: {', '.join(field_names)}")
            
            missing_fields = [field for field in required_fields if field not in field_names]
            if missing_fields:
                print(f"   ⚠️  Missing fields: {', '.join(missing_fields)}")
            else:
                print(f"   ✅ All required fields present")
            
            # Check embedding field specifically
            embedding_field = None
            for field in index.fields:
                if field.name == "embedding":
                    embedding_field = field
                    break
            
            if embedding_field:
                print(f"   ✅ Embedding field found")
                print(f"   📋 Type: {embedding_field.type}")
                if hasattr(embedding_field, 'vector_search_dimensions'):
                    print(f"   📋 Vector dimensions: {embedding_field.vector_search_dimensions}")
                else:
                    print(f"   ⚠️  No vector dimensions found")
            else:
                print(f"   ❌ Embedding field not found - vector search won't work")
                
        except Exception as e:
            print(f"   ❌ Schema validation failed: {e}")
        
        # Final Summary
        print(f"\n🎯 Final Test Summary")
        print(f"=" * 60)
        print(f"   Index Status: ✅ Connected ({total_docs} documents)")
        print(f"   Text Search: {'✅ Working' if successful_searches > 0 else '❌ Failed'} ({successful_searches}/{len(test_queries)} queries)")
        
        if 'vector_successful' in locals():
            print(f"   Vector Search: {'✅ Working' if vector_successful > 0 else '❌ Failed'} ({vector_successful}/{len(vector_test_queries)} queries)")
        else:
            print(f"   Vector Search: ⚠️  Not tested (configuration incomplete)")
        
        if successful_searches > 0:
            print(f"\n   🎉 Your search index is working!")
            print(f"   💡 You can now use it for RAG applications")
        else:
            print(f"\n   ❌ Search functionality has issues")
            print(f"   💡 Check your index configuration and content")
        
        print(f"\n📚 Suggested next steps:")
        print(f"   1. Try different search terms relevant to your documents")
        print(f"   2. Adjust chunk size if results are too large/small")
        print(f"   3. Implement this search in your RAG application")
        print(f"   4. Add result filtering by source_file or section as needed")
        
    except Exception as e:
        print(f"❌ General error: {e}")
        print(f"🔍 Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_search_functionality()