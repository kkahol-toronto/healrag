#!/usr/bin/env python3
"""
Trial Search Creation Script
============================

A simple script to test Azure Cognitive Search vector search creation
and document upload with sample text and embeddings.
"""

import os
import json
import logging
from typing import List, Dict
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if all required environment variables are set."""
    required_vars = [
        'AZURE_OPENAI_ENDPOINT',
        'AZURE_OPENAI_KEY', 
        'AZURE_TEXT_EMBEDDING_MODEL',
        'AZURE_SEARCH_ENDPOINT',
        'AZURE_SEARCH_KEY',
        'AZURE_SEARCH_INDEX_NAME'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {missing_vars}")
        return False
    
    print("✅ All required environment variables are set")
    return True

def create_sample_texts():
    """Create sample texts for testing."""
    return [
        {
            "id": "sample_001",
            "content": "This is a sample text about cybersecurity policies and best practices for protecting sensitive information.",
            "source": "test_document",
            "category": "security"
        },
        {
            "id": "sample_002", 
            "content": "Information security standards help organizations maintain data confidentiality, integrity, and availability.",
            "source": "test_document",
            "category": "security"
        },
        {
            "id": "sample_003",
            "content": "Risk assessment is a critical component of any cybersecurity framework and should be conducted regularly.",
            "source": "test_document", 
            "category": "risk"
        }
    ]

def create_embeddings(texts: List[Dict]):
    """Create embeddings for the sample texts using Azure OpenAI."""
    try:
        from openai import AzureOpenAI
        
        # Initialize Azure OpenAI client
        client = AzureOpenAI(
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_KEY'),
            api_version="2024-02-15-preview"
        )
        
        embedding_model = os.getenv('AZURE_TEXT_EMBEDDING_MODEL', 'text-embedding-ada-002')
        print(f"🔧 Using embedding model: {embedding_model}")
        
        results = []
        for i, text in enumerate(texts):
            print(f"🧠 Creating embedding for text {i+1}/{len(texts)}: {text['id']}")
            
            response = client.embeddings.create(
                input=text['content'],
                model=embedding_model
            )
            
            embedding = response.data[0].embedding
            print(f"   ✅ Embedding created: {len(embedding)} dimensions")
            print(f"   📊 First 5 values: {embedding[:5]}")
            
            # Add embedding to text
            text['embedding'] = embedding
            results.append(text)
        
        return results
        
    except Exception as e:
        print(f"❌ Error creating embeddings: {e}")
        return None

def create_search_index():
    """Create Azure Cognitive Search index with vector search using REST API."""
    try:
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential
        
        endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
        key = os.getenv('AZURE_SEARCH_KEY')
        index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'trial-index')
        api_version = '2023-11-01'
        
        # Get vector profile name from environment
        vector_profile = os.getenv('VECTOR_PROFILE_SEARCH', 'trial-vector-profile')
        
        # REST API URL
        url = f"{endpoint}/indexes/{index_name}?api-version={api_version}"
        
        # Index definition
        index_definition = {
            "name": index_name,
            "fields": [
                {"name": "id", "type": "Edm.String", "key": True, "filterable": True},
                {"name": "content", "type": "Edm.String", "searchable": True},
                {"name": "source", "type": "Edm.String"},
                {"name": "category", "type": "Edm.String"},
                {
                    "name": "embedding",
                    "type": "Collection(Edm.Single)",
                    "dimensions": 1536,
                    "vectorSearchConfiguration": vector_profile
                }
            ],
            "vectorSearch": {
                "profiles": [
                    {
                        "name": vector_profile,
                        "algorithm": "trial-algorithm"
                    }
                ],
                "algorithms": [
                    {
                        "name": "trial-algorithm",
                        "kind": "hnsw"
                    }
                ]
            }
        }
        
        # Delete existing index if it exists (SDK is fine for delete)
        try:
            from azure.search.documents.indexes import SearchIndexClient
            credential = AzureKeyCredential(key)
            index_client = SearchIndexClient(endpoint=endpoint, credential=credential)
            index_client.delete_index(index_name)
            print(f"🗑️  Deleted existing index: {index_name}")
        except Exception as e:
            pass
        
        # Create index via REST API
        headers = {
            'Content-Type': 'application/json',
            'api-key': key
        }
        response = requests.put(url, headers=headers, json=index_definition)
        if response.status_code in (200, 201):
            print(f"✅ Search index created successfully: {index_name}")
            return None, index_name
        else:
            print(f"❌ Error creating search index: {response.status_code} {response.text}")
            return None, None
    except Exception as e:
        print(f"❌ Error creating search index: {e}")
        return None, None

def upload_documents(documents: List[Dict], index_name: str):
    """Upload documents to the search index."""
    try:
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential
        
        endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
        key = os.getenv('AZURE_SEARCH_KEY')
        
        credential = AzureKeyCredential(key)
        search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
        
        print(f"📤 Uploading {len(documents)} documents to index: {index_name}")
        
        # Prepare documents for upload
        upload_docs = []
        for doc in documents:
            upload_doc = {
                "id": doc["id"],
                "content": doc["content"],
                "source": doc["source"],
                "category": doc["category"],
                "embedding": doc["embedding"]
            }
            upload_docs.append(upload_doc)
        
        # Upload documents
        result = search_client.upload_documents(upload_docs)
        
        # Check results
        successful = sum(1 for doc in result if doc.succeeded)
        failed = sum(1 for doc in result if not doc.succeeded)
        
        print(f"✅ Upload completed: {successful} successful, {failed} failed")
        
        if failed > 0:
            for doc in result:
                if not doc.succeeded:
                    print(f"   ❌ Failed to upload {doc.key}: {doc.status_code} - {doc.message}")
        
        return successful > 0
        
    except Exception as e:
        print(f"❌ Error uploading documents: {e}")
        return False

def test_search(index_name: str):
    """Test vector search functionality."""
    try:
        from azure.search.documents import SearchClient
        from azure.core.credentials import AzureKeyCredential
        from openai import AzureOpenAI
        
        endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
        key = os.getenv('AZURE_SEARCH_KEY')
        
        credential = AzureKeyCredential(key)
        search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)
        
        # Create query embedding
        client = AzureOpenAI(
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            api_key=os.getenv('AZURE_OPENAI_KEY'),
            api_version="2024-02-15-preview"
        )
        
        query = "cybersecurity policies"
        embedding_model = os.getenv('AZURE_TEXT_EMBEDDING_MODEL', 'text-embedding-ada-002')
        
        print(f"🔍 Testing search with query: '{query}'")
        
        response = client.embeddings.create(
            input=query,
            model=embedding_model
        )
        query_embedding = response.data[0].embedding
        
        # Perform vector search
        search_results = search_client.search(
            search_text=query,
            vector_queries=[
                {
                    "vector": query_embedding,
                    "k_nearest_neighbors": 3,
                    "fields": "embedding",
                    "kind": "vector"
                }
            ],
            select=["id", "content", "source", "category"],
            top=3
        )
        
        print(f"✅ Search completed successfully")
        print(f"📊 Found {len(list(search_results))} results:")
        
        for i, result in enumerate(search_results):
            print(f"   {i+1}. {result['id']} - {result['content'][:50]}...")
            print(f"      Score: {result.get('@search.score', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing search: {e}")
        return False

def main():
    """Main function to run the trial."""
    print("🚀 Trial Search Creation Script")
    print("=" * 50)
    
    # Check environment
    if not check_environment():
        return
    
    # Create sample texts
    print("\n📝 Creating sample texts...")
    texts = create_sample_texts()
    print(f"✅ Created {len(texts)} sample texts")
    
    # Create embeddings
    print("\n🧠 Creating embeddings...")
    documents = create_embeddings(texts)
    if not documents:
        print("❌ Failed to create embeddings")
        return
    
    # Create search index
    print("\n🔧 Creating search index...")
    index_client, index_name = create_search_index()
    if not index_client:
        print("❌ Failed to create search index")
        return
    
    # Upload documents
    print("\n📤 Uploading documents...")
    upload_success = upload_documents(documents, index_name)
    if not upload_success:
        print("❌ Failed to upload documents")
        return
    
    # Test search
    print("\n🔍 Testing search functionality...")
    search_success = test_search(index_name)
    
    # Summary
    print("\n" + "=" * 50)
    print("🎉 TRIAL COMPLETED")
    print("=" * 50)
    print(f"✅ Embeddings created: {len(documents)}")
    print(f"✅ Search index created: {index_name}")
    print(f"✅ Documents uploaded: {upload_success}")
    print(f"✅ Search tested: {search_success}")
    
    if upload_success and search_success:
        print("\n🎯 SUCCESS: Vector search is working correctly!")
    else:
        print("\n⚠️  Some issues were encountered. Check the logs above.")

if __name__ == "__main__":
    main() 