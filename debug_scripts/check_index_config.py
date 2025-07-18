#!/usr/bin/env python3
"""
Check Azure Search Index Configuration
=====================================

This script examines the Azure Cognitive Search index configuration
to understand the embedding field dimensions and other settings.
"""

import os
import json
from typing import Dict, Any, Optional

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from azure.search.documents.indexes import SearchIndexClient
    from azure.core.credentials import AzureKeyCredential
    AZURE_SEARCH_AVAILABLE = True
except ImportError:
    print("❌ azure-search-documents not available. Install with: pip install azure-search-documents")
    AZURE_SEARCH_AVAILABLE = False

def check_index_configuration():
    """Check the Azure Search index configuration."""
    
    if not AZURE_SEARCH_AVAILABLE:
        return False
    
    # Get configuration from environment
    azure_search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
    azure_search_key = os.getenv('AZURE_SEARCH_KEY')
    azure_search_index_name = os.getenv('AZURE_SEARCH_INDEX_NAME')
    
    print("🔍 Azure Search Index Configuration Check")
    print("=" * 50)
    print(f"Search Endpoint: {azure_search_endpoint}")
    print(f"Search Key: {'***' if azure_search_key else 'None'}")
    print(f"Index Name: {azure_search_index_name}")
    
    if not all([azure_search_endpoint, azure_search_key, azure_search_index_name]):
        print("\n❌ Missing Azure Search configuration!")
        print("Please set the following environment variables:")
        print("   AZURE_SEARCH_ENDPOINT")
        print("   AZURE_SEARCH_KEY")
        print("   AZURE_SEARCH_INDEX_NAME")
        return False
    
    try:
        # Initialize Search Index Client
        print(f"\n🔧 Initializing Search Index Client...")
        credential = AzureKeyCredential(azure_search_key)
        index_client = SearchIndexClient(
            endpoint=azure_search_endpoint,
            credential=credential
        )
        print(f"   ✅ Search Index Client created successfully")
        
        # Get index configuration
        print(f"\n📋 Getting index configuration for '{azure_search_index_name}'...")
        index = index_client.get_index(azure_search_index_name)
        
        print(f"\n📊 Index Configuration:")
        print(f"   Name: {index.name}")
        print(f"   Fields: {len(index.fields)}")
        
        # Examine fields
        print(f"\n🔍 Field Analysis:")
        embedding_fields = []
        
        for field in index.fields:
            print(f"\n   Field: {field.name}")
            print(f"     Type: {type(field).__name__}")
            
            if hasattr(field, 'searchable'):
                print(f"     Searchable: {field.searchable}")
            if hasattr(field, 'filterable'):
                print(f"     Filterable: {field.filterable}")
            if hasattr(field, 'sortable'):
                print(f"     Sortable: {field.sortable}")
            if hasattr(field, 'facetable'):
                print(f"     Facetable: {field.facetable}")
            
            # Check for vector fields
            if hasattr(field, 'vector_search_dimensions'):
                print(f"     Vector Dimensions: {field.vector_search_dimensions}")
                embedding_fields.append({
                    'name': field.name,
                    'dimensions': field.vector_search_dimensions
                })
            
            # Check for vector search profiles
            if hasattr(field, 'vector_search_profile_name'):
                print(f"     Vector Profile: {field.vector_search_profile_name}")
        
        # Check vector search configuration
        if hasattr(index, 'vector_search'):
            print(f"\n🧠 Vector Search Configuration:")
            print(f"   Profiles: {len(index.vector_search.profiles)}")
            
            for profile in index.vector_search.profiles:
                print(f"\n     Profile: {profile.name}")
                try:
                    if hasattr(profile, 'algorithm_configuration'):
                        print(f"       Algorithm: {profile.algorithm_configuration.name}")
                        if hasattr(profile.algorithm_configuration, 'parameters'):
                            for param, value in profile.algorithm_configuration.parameters.items():
                                print(f"       {param}: {value}")
                    else:
                        print(f"       Algorithm: {type(profile).__name__}")
                except Exception as e:
                    print(f"       Algorithm: Error reading configuration - {e}")
        
        # Summary
        print(f"\n📈 Summary:")
        print(f"   Total Fields: {len(index.fields)}")
        print(f"   Embedding Fields: {len(embedding_fields)}")
        
        for emb_field in embedding_fields:
            print(f"     - {emb_field['name']}: {emb_field['dimensions']} dimensions")
            
            # Determine compatible embedding models
            if emb_field['dimensions'] == 1536:
                print(f"       ✅ Compatible with: text-embedding-ada-002")
                print(f"       ❌ Incompatible with: text-embedding-3-large (3072 dims)")
            elif emb_field['dimensions'] == 3072:
                print(f"       ✅ Compatible with: text-embedding-3-large")
                print(f"       ❌ Incompatible with: text-embedding-ada-002 (1536 dims)")
            else:
                print(f"       ⚠️  Unknown dimension size: {emb_field['dimensions']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error checking index configuration: {e}")
        return False

def check_index_statistics():
    """Check index statistics and document count."""
    
    if not AZURE_SEARCH_AVAILABLE:
        return False
    
    # Get configuration from environment
    azure_search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
    azure_search_key = os.getenv('AZURE_SEARCH_KEY')
    azure_search_index_name = os.getenv('AZURE_SEARCH_INDEX_NAME')
    
    try:
        from azure.search.documents import SearchClient
        
        print(f"\n📊 Index Statistics:")
        print("=" * 30)
        
        credential = AzureKeyCredential(azure_search_key)
        search_client = SearchClient(
            endpoint=azure_search_endpoint,
            index_name=azure_search_index_name,
            credential=credential
        )
        
        # Get index statistics
        stats = search_client.get_document_count()
        print(f"   Document Count: {stats}")
        
        # Try to get a sample document
        try:
            results = search_client.search("*", top=1)
            sample_doc = next(results, None)
            if sample_doc:
                print(f"\n📄 Sample Document Fields:")
                for key, value in sample_doc.items():
                    if key == 'embedding':
                        print(f"     {key}: [Vector with {len(value)} dimensions]")
                    else:
                        print(f"     {key}: {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
        except Exception as e:
            print(f"   Could not retrieve sample document: {e}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error checking index statistics: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Azure Search Index Configuration Checker")
    print("=" * 50)
    
    success = check_index_configuration()
    if success:
        check_index_statistics()
    
    print(f"\n{'✅' if success else '❌'} Configuration check completed") 