"""
Search Index Manager for HEALRAG

Handles chunking of markdown files, embedding generation using Azure OpenAI,
and Azure Cognitive Search index management.
"""

import os
import json
import logging
import tempfile
import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
import time

try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False

try:
    from azure.search.documents import SearchClient
    from azure.search.documents.indexes import SearchIndexClient
    from azure.search.documents.indexes.models import (
        SearchIndex, 
        SimpleField, 
        SearchField, SearchableField,
        VectorSearch,
        HnswAlgorithmConfiguration,
        VectorSearchProfile
    )
    from azure.core.credentials import AzureKeyCredential
    AZURE_SEARCH_AVAILABLE = True
except ImportError as e:
    print(f"Warning: azure-search-documents not available: {e}")
    AZURE_SEARCH_AVAILABLE = False
    # Set these to None to avoid NameError
    SearchClient = None
    SearchIndexClient = None
    AzureKeyCredential = None

# At the top of the class or file, after imports
VECTOR_PROFILE_SEARCH = os.getenv("VECTOR_PROFILE_SEARCH", "my-vector-config")

class SearchIndexManager:
    """
    Search Index Manager for HEALRAG library.
    
    Handles markdown chunking, embedding generation, and Azure Cognitive Search
    index management for RAG applications.
    """
    
    def __init__(self, 
                 storage_manager,
                 azure_openai_endpoint: Optional[str] = None,
                 azure_openai_key: Optional[str] = None,
                 azure_openai_deployment: Optional[str] = None,
                 azure_search_endpoint: Optional[str] = None,
                 azure_search_key: Optional[str] = None,
                 azure_search_index_name: str = "healrag-index",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200):
        """
        Initialize the Search Index Manager.
        
        Args:
            storage_manager: HEALRAG StorageManager instance
            azure_openai_endpoint: Azure OpenAI endpoint
            azure_openai_key: Azure OpenAI API key
            azure_openai_deployment: Azure OpenAI embedding deployment name
            azure_search_endpoint: Azure Cognitive Search endpoint
            azure_search_key: Azure Cognitive Search API key
            azure_search_index_name: Name of the search index
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        self.storage_manager = storage_manager
        self.azure_openai_endpoint = azure_openai_endpoint
        self.azure_openai_key = azure_openai_key
        self.azure_openai_deployment = azure_openai_deployment
        self.azure_search_endpoint = azure_search_endpoint
        self.azure_search_key = azure_search_key
        self.azure_search_index_name = azure_search_index_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
        
        # Initialize Azure OpenAI client for embeddings
        self.openai_client = None
        if all([azure_openai_endpoint, azure_openai_key, azure_openai_deployment]):
            self._initialize_openai_client()
        
        # Initialize Azure Search clients
        self.search_index_client = None
        self.search_client = None
        if all([azure_search_endpoint, azure_search_key]):
            self._initialize_search_clients()
    
    def _initialize_openai_client(self):
        """Initialize Azure OpenAI client for embeddings."""
        try:
            print(f"🔧 Initializing Azure OpenAI client...")
            print(f"   Endpoint: {self.azure_openai_endpoint}")
            print(f"   Deployment: {self.azure_openai_deployment}")
            print(f"   API Key: {'***' if self.azure_openai_key else 'None'}")
            
            self.openai_client = AzureOpenAI(
                azure_endpoint=self.azure_openai_endpoint,
                api_key=self.azure_openai_key,
                api_version="2024-02-15-preview"
            )
            
            # Test the client with a simple request
            print(f"   Testing Azure OpenAI connection...")
            test_response = self.openai_client.embeddings.create(
                input="test",
                model=self.azure_openai_deployment
            )
            print(f"   ✅ Azure OpenAI client initialized successfully for embeddings")
            print(f"   Test embedding length: {len(test_response.data[0].embedding)}")
            
        except ImportError:
            print(f"   ❌ openai package not installed. Install with: pip install openai")
            self.logger.warning("openai package not installed. Install with: pip install openai")
        except Exception as e:
            print(f"   ❌ Failed to initialize Azure OpenAI client: {e}")
            self.logger.error(f"Failed to initialize Azure OpenAI client: {e}")
    
    def _initialize_search_clients(self):
        """Initialize Azure Cognitive Search clients."""
        if not AZURE_SEARCH_AVAILABLE:
            self.logger.error("Azure Search not available. Install with: pip install azure-search-documents")
            return
            
        try:
            credential = AzureKeyCredential(self.azure_search_key)
            self.search_index_client = SearchIndexClient(
                endpoint=self.azure_search_endpoint,
                credential=credential
            )
            self.search_client = SearchClient(
                endpoint=self.azure_search_endpoint,
                index_name=self.azure_search_index_name,
                credential=credential
            )
            self.logger.info("Azure Cognitive Search clients initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure Search clients: {e}")
    
    def chunk_markdown_content(self, content: str, source_file: str) -> List[Dict]:
        """
        FIXED: Chunk markdown content into smaller pieces with improved algorithm.
        
        Args:
            content: Markdown content to chunk
            source_file: Source file name for reference
            
        Returns:
            List of chunk dictionaries with metadata
        """
        chunks = []
        
        print(f"🔍 Chunking file: {Path(source_file).name}")
        print(f"   Content length: {len(content):,} characters")
        print(f"   Target chunk size: {self.chunk_size} characters")
        
        # Skip if content is too short
        if len(content.strip()) < 50:
            print(f"   ⚠️  Content too short ({len(content)} chars), skipping")
            return chunks
        
        # Clean and preprocess content
        # Remove excessive whitespace but preserve structure
        cleaned_content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        cleaned_content = cleaned_content.strip()
        
        # Filter out image metadata and keep substantial content
        lines = cleaned_content.split('\n')
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip image metadata lines
            if (line.startswith('Image ') and len(line) < 50) or \
               (line == 'Extracted Images') or \
               (line.startswith('Visual Elements') and len(line) < 100) or \
               (re.match(r'^Image \d+$', line)) or \
               (line.startswith('![') and line.endswith(')')) or \
               (line == '') or \
               (len(line) < 5):
                continue
            
            # Keep substantial content lines
            filtered_lines.append(line)
        
        # Reconstruct content without image metadata
        filtered_content = '\n\n'.join(filtered_lines)
        
        print(f"   Original lines: {len(lines)}")
        print(f"   After filtering: {len(filtered_lines)}")
        print(f"   Filtered content length: {len(filtered_content):,} characters")
        
        if len(filtered_content.strip()) < 50:
            print(f"   ⚠️  No substantial content after filtering")
            return chunks
        
        # Try to extract meaningful content sections
        # Split by headers first (## and ###)
        header_pattern = r'^(#{1,6})\s+(.+)$'
        parts = re.split(header_pattern, filtered_content, flags=re.MULTILINE)
        
        # If no headers found, fall back to paragraph-based chunking
        if len(parts) <= 1:
            print(f"   📝 No clear headers found, using paragraph-based chunking")
            return self._chunk_by_paragraphs(filtered_content, source_file)
        
        print(f"   📑 Found {len(parts)//3} header sections")
        
        current_chunk = ""
        current_section = "Introduction"
        chunk_id = 0
        
        i = 0
        while i < len(parts):
            if i % 3 == 0:  # Content before/between headers
                content_part = parts[i].strip()
                if content_part and len(content_part) > 20:  # Only substantial content
                    current_chunk, chunk_id = self._add_to_chunk(
                        current_chunk, content_part, chunk_id, chunks, source_file, current_section
                    )
            elif i % 3 == 1:  # Header level (##, ###, etc.)
                header_level = parts[i]
                if i + 1 < len(parts):
                    header_text = parts[i + 1].strip()
                    current_section = header_text
                    print(f"   📍 Section: {current_section}")
                    
                    # Add header to current chunk
                    header_full = f"{header_level} {header_text}"
                    current_chunk, chunk_id = self._add_to_chunk(
                        current_chunk, header_full, chunk_id, chunks, source_file, current_section
                    )
                i += 1  # Skip header text part
            i += 1
        
        # Add the final chunk if there's content
        if current_chunk.strip() and len(current_chunk.strip()) > 20:
            chunk_dict = {
                "id": f"{Path(source_file).stem}_{chunk_id:04d}",
                "content": current_chunk.strip(),
                "source_file": source_file,
                "section": current_section,
                "chunk_size": len(current_chunk.strip()),
                "chunk_index": chunk_id
            }
            chunks.append(chunk_dict)
            print(f"   ✅ Final chunk {chunk_id}: {len(current_chunk.strip())} chars")
        
        print(f"   📊 Created {len(chunks)} chunks total")
        return chunks
    
    def _add_to_chunk(self, current_chunk: str, new_content: str, chunk_id: int, 
                      chunks: List[Dict], source_file: str, current_section: str) -> tuple:
        """Helper method to add content to current chunk or create new chunk if size exceeded."""
        
        # Calculate what the new size would be
        separator = "\n\n" if current_chunk else ""
        potential_chunk = current_chunk + separator + new_content
        
        # If adding this content would exceed chunk size, save current chunk and start new one
        if len(potential_chunk) > self.chunk_size and current_chunk.strip() and len(current_chunk.strip()) > 20:
            # Save current chunk
            chunk_dict = {
                "id": f"{Path(source_file).stem}_{chunk_id:04d}",
                "content": current_chunk.strip(),
                "source_file": source_file,
                "section": current_section,
                "chunk_size": len(current_chunk.strip()),
                "chunk_index": chunk_id
            }
            chunks.append(chunk_dict)
            print(f"   ✅ Chunk {chunk_id}: {len(current_chunk.strip())} chars")
            
            # Start new chunk with overlap
            overlap_text = self._get_overlap_text(current_chunk)
            return overlap_text + new_content, chunk_id + 1
        else:
            # Add to current chunk
            return potential_chunk, chunk_id
    
    def _get_overlap_text(self, text: str) -> str:
        """Extract overlap text from the end of current chunk."""
        if len(text) <= self.chunk_overlap:
            return text + "\n\n"
        
        # Try to find a good breaking point for overlap (sentence or paragraph)
        overlap_candidate = text[-self.chunk_overlap:]
        
        # Look for sentence boundaries
        sentence_end = max(
            overlap_candidate.rfind('. '),
            overlap_candidate.rfind('.\n'),
            overlap_candidate.rfind('?\n'),
            overlap_candidate.rfind('!\n')
        )
        
        if sentence_end > self.chunk_overlap // 2:
            return overlap_candidate[sentence_end + 1:].strip() + "\n\n"
        
        # Look for paragraph boundaries
        para_break = overlap_candidate.rfind('\n\n')
        if para_break > self.chunk_overlap // 3:
            return overlap_candidate[para_break:].strip() + "\n\n"
        
        # Fall back to simple character overlap
        return overlap_candidate + "\n\n"
    
    def _chunk_by_paragraphs(self, content: str, source_file: str) -> List[Dict]:
        """Fallback chunking method using paragraph breaks."""
        
        chunks = []
        paragraphs = re.split(r'\n\s*\n', content)
        paragraphs = [p.strip() for p in paragraphs if p.strip() and len(p.strip()) > 20]
        
        print(f"   📝 Chunking {len(paragraphs)} substantial paragraphs")
        
        if not paragraphs:
            return chunks
        
        current_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk.strip():
                # Save current chunk
                chunk_dict = {
                    "id": f"{Path(source_file).stem}_{chunk_id:04d}",
                    "content": current_chunk.strip(),
                    "source_file": source_file,
                    "section": "Content",
                    "chunk_size": len(current_chunk.strip()),
                    "chunk_index": chunk_id
                }
                chunks.append(chunk_dict)
                print(f"   ✅ Paragraph chunk {chunk_id}: {len(current_chunk.strip())} chars")
                chunk_id += 1
                
                # Start new chunk with overlap
                overlap = self._get_overlap_text(current_chunk)
                current_chunk = overlap + para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add final chunk
        if current_chunk.strip() and len(current_chunk.strip()) > 20:
            chunk_dict = {
                "id": f"{Path(source_file).stem}_{chunk_id:04d}",
                "content": current_chunk.strip(),
                "source_file": source_file,
                "section": "Content",
                "chunk_size": len(current_chunk.strip()),
                "chunk_index": chunk_id
            }
            chunks.append(chunk_dict)
            print(f"   ✅ Final paragraph chunk {chunk_id}: {len(current_chunk.strip())} chars")
        
        return chunks
    
    def _split_large_chunk(self, chunk: Dict) -> List[Dict]:
        """Split a large chunk into smaller pieces."""
        content = chunk["content"]
        sub_chunks = []
        
        # Split by sentences first, then by character limit
        sentences = re.split(r'(?<=[.!?])\s+', content)
        
        current_sub_chunk = ""
        sub_chunk_id = 0
        
        for sentence in sentences:
            if len(current_sub_chunk) + len(sentence) > self.chunk_size and current_sub_chunk:
                sub_chunks.append({
                    "id": f"{chunk['id']}_sub_{sub_chunk_id:02d}",
                    "content": current_sub_chunk.strip(),
                    "source_file": chunk["source_file"],
                    "section": chunk["section"],
                    "chunk_size": len(current_sub_chunk.strip()),
                    "chunk_index": chunk["chunk_index"],
                    "sub_chunk_index": sub_chunk_id
                })
                sub_chunk_id += 1
                current_sub_chunk = sentence
            else:
                current_sub_chunk += " " + sentence if current_sub_chunk else sentence
        
        # Add the last sub-chunk
        if current_sub_chunk.strip():
            sub_chunks.append({
                "id": f"{chunk['id']}_sub_{sub_chunk_id:02d}",
                "content": current_sub_chunk.strip(),
                "source_file": chunk["source_file"],
                "section": chunk["section"],
                "chunk_size": len(current_sub_chunk.strip()),
                "chunk_index": chunk["chunk_index"],
                "sub_chunk_index": sub_chunk_id
            })
        
        return sub_chunks
    
    def generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """
        Generate embeddings for text chunks using Azure OpenAI.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of chunks with embeddings added
        """
        if not self.openai_client:
            self.logger.warning("Azure OpenAI not configured. Skipping embedding generation.")
            return chunks
        
        chunks_with_embeddings = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Debug: Print chunk details
                print(f"\n🔍 Processing chunk {i+1}/{len(chunks)}:")
                print(f"   ID: {chunk['id']}")
                print(f"   Source: {chunk['source_file']}")
                print(f"   Section: {chunk['section']}")
                print(f"   Content length: {len(chunk['content'])} characters")
                print(f"   Content preview: {chunk['content'][:100]}...")
                print(f"   Azure OpenAI deployment: {self.azure_openai_deployment}")
                
                # Generate embedding
                print(f"   🧠 Generating embedding...")
                response = self.openai_client.embeddings.create(
                    input=chunk["content"],
                    model=self.azure_openai_deployment
                )
                
                embedding = response.data[0].embedding
                
                # Debug: Print embedding details
                print(f"   ✅ Embedding generated successfully!")
                print(f"   Embedding length: {len(embedding)}")
                print(f"   Embedding type: {type(embedding)}")
                print(f"   First 5 values: {embedding[:5]}")
                print(f"   Last 5 values: {embedding[-5:]}")
                
                # Add embedding to chunk
                chunk_with_embedding = chunk.copy()
                chunk_with_embedding["embedding"] = embedding
                chunk_with_embedding["embedding_model"] = self.azure_openai_deployment
                chunk_with_embedding["embedding_timestamp"] = datetime.now().isoformat()
                
                chunks_with_embeddings.append(chunk_with_embedding)
                
                # Log progress
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Generated embeddings for {i + 1}/{len(chunks)} chunks")
                
                # Rate limiting - small delay between requests
                time.sleep(0.1)
                
            except Exception as e:
                print(f"   ❌ Error generating embedding for chunk {chunk['id']}: {e}")
                print(f"   Error type: {type(e)}")
                self.logger.error(f"Error generating embedding for chunk {chunk['id']}: {e}")
                
                # Add chunk without embedding
                chunk_with_embedding = chunk.copy()
                chunk_with_embedding["embedding"] = None
                chunk_with_embedding["embedding_error"] = str(e)
                chunks_with_embeddings.append(chunk_with_embedding)
        
        # Summary
        successful_embeddings = sum(1 for chunk in chunks_with_embeddings if chunk.get('embedding') is not None)
        failed_embeddings = sum(1 for chunk in chunks_with_embeddings if chunk.get('embedding') is None)
        
        print(f"\n📊 Embedding Generation Summary:")
        print(f"   Total chunks: {len(chunks)}")
        print(f"   Successful embeddings: {successful_embeddings}")
        print(f"   Failed embeddings: {failed_embeddings}")
        
        self.logger.info(f"Generated embeddings for {successful_embeddings}/{len(chunks)} chunks")
        return chunks_with_embeddings
    
    def create_search_index(self) -> bool:
        """
        Create Azure Cognitive Search index with vector search capabilities.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.search_index_client:
            self.logger.error("Azure Search not configured")
            return False
        
        # Check if Azure Search service supports vector search
        try:
            print(f"   🔍 Checking Azure Search service capabilities...")
            service_stats = self.search_index_client.get_service_statistics()
            print(f"   ✅ Azure Search service is accessible")
            
            # Check if vector search is supported by looking at existing indexes
            try:
                existing_indexes = self.search_index_client.list_indexes()
                vector_search_supported = False
                for index in existing_indexes:
                    if hasattr(index, 'vector_search') and index.vector_search:
                        vector_search_supported = True
                        print(f"   ✅ Vector search is supported (found existing vector index: {index.name})")
                        break
                
                if not vector_search_supported:
                    print(f"   ⚠️  No existing vector indexes found, but will attempt to create one")
                    
            except Exception as list_error:
                print(f"   ⚠️  Could not check existing indexes: {list_error}")
                print(f"   💡 Will attempt to create vector search index anyway")
            
        except Exception as service_error:
            print(f"   ❌ Cannot access Azure Search service: {service_error}")
            print(f"   💡 Please check your Azure Search endpoint and API key")
            return False
        
        try:
            # Check if index already exists
            try:
                existing_index = self.search_index_client.get_index(self.azure_search_index_name)
                self.logger.info(f"Search index '{self.azure_search_index_name}' already exists")
                
                # Validate that the index has the required fields
                existing_fields = {field.name: field.type for field in existing_index.fields}
                required_fields = {
                    "id": "Edm.String",
                    "content": "Edm.String", 
                    "source_file": "Edm.String",
                    "section": "Edm.String",
                    "chunk_size": "Edm.Int32",
                    "chunk_index": "Edm.Int32"
                }
                
                missing_fields = [field for field, field_type in required_fields.items() if field not in existing_fields]
                if missing_fields:
                    self.logger.warning(f"Search index missing required fields: {missing_fields}")
                    # For now, we'll try to continue, but this might cause issues
                
                # Check if embedding field exists and has correct type
                embedding_field_ok = True
                if "embedding" in existing_fields:
                    embedding_field_type = existing_fields["embedding"]
                    self.logger.info(f"Embedding field type: {embedding_field_type}")
                    if embedding_field_type != "Collection(Edm.Single)":
                        self.logger.warning(f"Embedding field has wrong type: {embedding_field_type} (expected Collection(Edm.Single))")
                        embedding_field_ok = False
                        print(f"   ⚠️  Existing search index has wrong embedding field type")
                        print(f"   💡 Current type: {embedding_field_type}")
                        print(f"   💡 Expected type: Collection(Edm.Single)")
                        print(f"   💡 The index needs to be recreated with correct vector field")
                else:
                    self.logger.warning("Embedding field not found in existing index")
                    embedding_field_ok = False
                    print(f"   ⚠️  No embedding field found in existing search index")
                
                if not embedding_field_ok:
                    print(f"   🔧 Attempting to recreate search index with correct schema...")
                    try:
                        # Delete the existing index
                        self.search_index_client.delete_index(self.azure_search_index_name)
                        self.logger.info(f"Deleted existing index '{self.azure_search_index_name}'")
                        print(f"   ✅ Deleted existing index")
                        # Continue to create new index below
                    except Exception as delete_error:
                        self.logger.error(f"Failed to delete existing index: {delete_error}")
                        print(f"   ❌ Failed to delete existing index: {delete_error}")
                        print(f"   💡 Please manually delete the index '{self.azure_search_index_name}' in Azure Portal")
                        return False
                else:
                    return True
            except Exception:
                # Index doesn't exist, create it
                pass
            
            # Try to create index with vector search
            try:
                print(f"   🔧 Creating new index with vector search...")
                
                index = SearchIndex(
                    name=self.azure_search_index_name,
                    fields=[
                        SimpleField(name="id", type="Edm.String", key=True),
                        SearchableField(name="content", type="Edm.String"),
                        SearchField(name="source_file", type="Edm.String", filterable=True, facetable=True),
                        SearchField(name="section", type="Edm.String", filterable=True, facetable=True),
                        SearchField(name="chunk_size", type="Edm.Int32", filterable=True),
                        SearchField(name="chunk_index", type="Edm.Int32", filterable=True),
                        SearchField(name="sub_chunk_index", type="Edm.Int32", filterable=True, retrievable=True),
                        SearchField(name="embedding_model", type="Edm.String", filterable=True),
                        SearchField(name="embedding_timestamp", type="Edm.String", filterable=True),
                        SearchField(name="embedding_error", type="Edm.String", searchable=True),
                        SearchField(
                            name="embedding",
                            type="Collection(Edm.Single)",
                            searchable=True,
                            vector_search_dimensions=1536,
                            vector_search_profile_name=VECTOR_PROFILE_SEARCH
                        ),
                    ],
                    vector_search=VectorSearch(
                        profiles=[
                            VectorSearchProfile(
                                name=VECTOR_PROFILE_SEARCH,
                                algorithm_configuration_name="my-algorithms-config"
                            )
                        ],
                        algorithms=[
                            HnswAlgorithmConfiguration(
                                name="my-algorithms-config"
                            )
                        ]
                    )
                )
                
                # Add a small delay to ensure any deletion is processed
                time.sleep(2)
                
                result = self.search_index_client.create_index(index)
                self.logger.info(f"Search index '{self.azure_search_index_name}' created successfully with vector search")
                print(f"   ✅ Search index created successfully with vector search")
                
                # Verify the index was created with the correct field type
                try:
                    created_index = self.search_index_client.get_index(self.azure_search_index_name)
                    embedding_field = next((f for f in created_index.fields if f.name == "embedding"), None)
                    if embedding_field and hasattr(embedding_field, 'vector_search_dimensions'):
                        print(f"   ✅ Verified: embedding field has vector search dimensions: {embedding_field.vector_search_dimensions}")
                    else:
                        print(f"   ⚠️  Warning: embedding field does not have vector search dimensions")
                        print(f"   💡 Field type: {embedding_field.type if embedding_field else 'Not found'}")
                except Exception as verify_error:
                    print(f"   ⚠️  Could not verify index configuration: {verify_error}")
                
                return True
                
            except Exception as vector_error:
                print(f"   ❌ Vector search creation failed: {vector_error}")
                print(f"   🔍 Error details: {type(vector_error).__name__}: {str(vector_error)}")
                
                # Log the full error details for debugging
                self.logger.error(f"Full vector search creation error: {vector_error}")
                if hasattr(vector_error, 'response'):
                    self.logger.error(f"Response status: {vector_error.response.status_code}")
                    self.logger.error(f"Response body: {vector_error.response.text}")
                
                # Check specific vector search errors
                if "BadRequest" in str(vector_error):
                    print(f"   💡 This appears to be a configuration issue with vector search")
                    print(f"   💡 Your Azure Search service may not support vector search")
                    print(f"   💡 Vector search requires Azure Cognitive Search with vector search capabilities")
                elif "Forbidden" in str(vector_error):
                    print(f"   💡 This appears to be a permissions issue")
                    print(f"   💡 Please check your Azure Search API key and permissions")
                elif "Unsupported" in str(vector_error):
                    print(f"   💡 Vector search is not supported by your Azure Search service")
                    print(f"   💡 You need to upgrade to a service that supports vector search")
                elif "Conflict" in str(vector_error):
                    print(f"   💡 Index already exists or is being created")
                    print(f"   💡 Please wait a moment and try again")
                
                print(f"   🚫 HEALRAG requires vector search capabilities")
                print(f"   💡 Please ensure your Azure Search service supports vector search")
                return False
                
        except Exception as e:
            self.logger.error(f"Error creating search index: {e}")
            print(f"   ❌ Failed to create search index: {e}")
            print(f"   🔍 Error details: {type(e).__name__}: {str(e)}")
            
            # Check if it's a permissions issue
            if "Forbidden" in str(e) or "Unauthorized" in str(e):
                print(f"   💡 This appears to be a permissions issue")
                print(f"   💡 Please check your Azure Search API key and permissions")
            elif "Conflict" in str(e):
                print(f"   💡 Index already exists or is being created")
                print(f"   💡 Please wait a moment and try again")
            elif "BadRequest" in str(e):
                print(f"   💡 Invalid index configuration")
                print(f"   💡 Check your Azure Search service capabilities")
            
            return False
    
    def upload_chunks_to_index(self, chunks: List[Dict]) -> bool:
        """
        Upload chunks with embeddings to Azure Cognitive Search index.
        
        Args:
            chunks: List of chunk dictionaries with embeddings
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.search_client:
            self.logger.error("Azure Search not configured")
            return False
        
        try:
            # Prepare documents for upload
            documents = []
            chunks_with_embeddings = 0
            chunks_without_embeddings = 0
            
            for chunk in chunks:
                # Validate required fields
                if not chunk.get("id") or not chunk.get("content"):
                    self.logger.warning(f"Skipping chunk with missing required fields: {chunk.get('id', 'unknown')}")
                    continue
                
                # Sanitize document ID to comply with Azure Search requirements
                # Azure Search only allows letters, digits, underscore (_), dash (-), or equal sign (=)
                import re
                safe_id = re.sub(r'[^a-zA-Z0-9_\-=]', '_', chunk["id"])
                if safe_id != chunk["id"]:
                    self.logger.info(f"Sanitized document ID: '{chunk['id']}' -> '{safe_id}'")
                
                doc = {
                    "id": safe_id,
                    "content": chunk["content"],
                    "source_file": chunk["source_file"],
                    "section": chunk["section"],
                    "chunk_size": chunk["chunk_size"],
                    "chunk_index": chunk["chunk_index"]
                }
                
                # Add optional fields only if they exist and are not None
                if chunk.get("sub_chunk_index") is not None:
                    doc["sub_chunk_index"] = chunk["sub_chunk_index"]
                if chunk.get("embedding_model"):
                    doc["embedding_model"] = chunk["embedding_model"]
                if chunk.get("embedding_timestamp"):
                    doc["embedding_timestamp"] = chunk["embedding_timestamp"]
                if chunk.get("embedding_error"):
                    doc["embedding_error"] = chunk["embedding_error"]
                
                # Add embedding if available
                if chunk.get("embedding") is not None and isinstance(chunk["embedding"], list):
                    # Validate embedding format
                    embedding = chunk["embedding"]
                    if len(embedding) == 1536 and all(isinstance(x, (int, float)) for x in embedding):
                        # HEALRAG only supports vector search - store as vector
                        doc["embedding"] = [float(x) for x in embedding]
                        chunks_with_embeddings += 1
                        print(f"   ✅ Chunk {chunk['id']}: Vector embedding ({len(embedding)} dimensions)")
                    else:
                        self.logger.warning(f"Invalid embedding format for chunk {chunk['id']}: length={len(embedding)}, type={type(embedding[0]) if embedding else 'empty'}")
                        chunks_without_embeddings += 1
                        print(f"   ⚠️  Chunk {chunk['id']}: Invalid embedding format")
                else:
                    chunks_without_embeddings += 1
                    print(f"   ❌ Chunk {chunk['id']}: No embedding (None or not list)")
                
                documents.append(doc)
            
            # Debug: Log first document structure for troubleshooting
            if documents:
                # Create a copy of the first document for debugging (without the embedding)
                debug_doc = documents[0].copy()
                if 'embedding' in debug_doc:
                    debug_doc['embedding'] = f"[{len(debug_doc['embedding'])} values]"
                self.logger.debug(f"Sample document structure: {json.dumps(debug_doc, indent=2, default=str)}")
                
                # Also log the actual embedding format
                if 'embedding' in documents[0]:
                    embedding = documents[0]['embedding']
                    self.logger.debug(f"Embedding format: type={type(embedding)}, length={len(embedding)}, first_5={embedding[:5]}")
            
            print(f"\n📊 Upload Preparation Summary:")
            print(f"   Total chunks to upload: {len(chunks)}")
            print(f"   Chunks with valid embeddings: {chunks_with_embeddings}")
            print(f"   Chunks without embeddings: {chunks_without_embeddings}")
            print(f"   Documents prepared for upload: {len(documents)}")
            
            # Upload in batches
            batch_size = 100
            total_uploaded = 0
            total_failed = 0
            
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                try:
                    # Try to serialize the first document to check for JSON issues
                    if i == 0 and batch:
                        try:
                            test_serialization = json.dumps(batch[0])
                            self.logger.debug(f"First document serializes successfully")
                        except Exception as json_error:
                            self.logger.error(f"JSON serialization error: {json_error}")
                            # Try to fix the document
                            for doc in batch:
                                if 'embedding' in doc and isinstance(doc['embedding'], list):
                                    doc['embedding'] = [float(x) for x in doc['embedding']]
                    
                    result = self.search_client.upload_documents(batch)
                    
                    # Check for errors
                    failed_docs = [doc for doc in result if not doc.succeeded]
                    if failed_docs:
                        failed_count = len(failed_docs)
                        total_failed += failed_count
                        self.logger.warning(f"Failed to upload {failed_count} documents in batch {i//batch_size + 1}")
                        for failed_doc in failed_docs:
                            self.logger.error(f"Failed document: {failed_doc}")
                    
                    successful_count = len(batch) - len(failed_docs)
                    total_uploaded += successful_count
                    self.logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}: {successful_count} successful, {len(failed_docs)} failed")
                    
                except Exception as batch_error:
                    self.logger.error(f"Error uploading batch {i//batch_size + 1}: {batch_error}")
                    self.logger.error(f"Error details: {type(batch_error).__name__}: {str(batch_error)}")
                    total_failed += len(batch)
                    # Continue with next batch instead of failing completely
                    continue
            
            self.logger.info(f"Upload completed: {total_uploaded} successful, {total_failed} failed out of {len(documents)} total chunks")
            
            # Return True if at least some documents were uploaded successfully
            if total_uploaded > 0:
                return True
            else:
                self.logger.error("No documents were uploaded successfully")
                
                # Try uploading without embeddings as a fallback
                self.logger.info("Attempting to upload documents without embeddings...")
                documents_without_embeddings = []
                for doc in documents:
                    doc_without_embedding = {k: v for k, v in doc.items() if k != 'embedding'}
                    documents_without_embeddings.append(doc_without_embedding)
                
                try:
                    result = self.search_client.upload_documents(documents_without_embeddings)
                    successful_uploads = sum(1 for doc in result if doc.succeeded)
                    if successful_uploads > 0:
                        self.logger.info(f"Successfully uploaded {successful_uploads} documents without embeddings")
                        return True
                    else:
                        self.logger.error("Failed to upload documents even without embeddings")
                        return False
                except Exception as fallback_error:
                    self.logger.error(f"Fallback upload failed: {fallback_error}")
                    return False
            
        except Exception as e:
            self.logger.error(f"Error uploading chunks to search index: {e}")
            return False
    
    def process_markdown_files(self, md_folder: str = "md_files") -> Dict:
        """
        Process all markdown files in the specified folder.
        
        Args:
            md_folder: Folder containing markdown files
            
        Returns:
            Dict: Processing results and statistics
        """
        self.logger.info(f"Starting to process markdown files from {md_folder}")
        
        # Get list of markdown files
        md_files = self.storage_manager.get_file_list(as_json=False)
        md_files = [f for f in md_files if f.startswith(f"{md_folder}/") and f.endswith('.md')]
        
        if not md_files:
            self.logger.warning(f"No markdown files found in {md_folder}")
            return {"success": False, "error": "No markdown files found"}
        
        self.logger.info(f"Found {len(md_files)} markdown files to process")
        
        all_chunks = []
        processing_stats = {
            "files_processed": 0,
            "total_chunks": 0,
            "chunks_with_embeddings": 0,
            "chunks_without_embeddings": 0,
            "processing_errors": 0,
            "upload_errors": False
        }
        
        # Process each markdown file
        for i, md_file in enumerate(md_files):
            try:
                files_remaining = len(md_files) - i
                self.logger.info(f"Processing {md_file} ({i+1}/{len(md_files)}, {files_remaining} files remaining)")
                print(f"   📄 Processing: {Path(md_file).name} ({i+1}/{len(md_files)}, {files_remaining} left)")
                
                # Download and read markdown content
                with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as temp_file:
                    temp_file_path = temp_file.name
                
                try:
                    if not self.storage_manager.download_file(md_file, temp_file_path):
                        raise Exception(f"Failed to download {md_file}")
                    
                    with open(temp_file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Chunk the content
                    chunks = self.chunk_markdown_content(content, md_file)
                    all_chunks.extend(chunks)
                    processing_stats["total_chunks"] += len(chunks)
                    processing_stats["files_processed"] += 1
                    
                    print(f"   ✅ Completed: {Path(md_file).name} ({len(chunks)} chunks created)")
                    
                finally:
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                
            except Exception as e:
                self.logger.error(f"Error processing {md_file}: {e}")
                print(f"   ❌ Failed: {Path(md_file).name} - {str(e)}")
                processing_stats["processing_errors"] += 1
        
        # Generate embeddings
        if all_chunks:
            print(f"\n🧠 Starting embedding generation for {len(all_chunks)} chunks...")
            
            # Check Azure OpenAI configuration
            if not self.openai_client:
                print(f"   ❌ Azure OpenAI client not initialized!")
                print(f"   Please check your Azure OpenAI configuration:")
                print(f"     - AZURE_OPENAI_ENDPOINT: {self.azure_openai_endpoint}")
                print(f"     - AZURE_OPENAI_KEY: {'***' if self.azure_openai_key else 'None'}")
                print(f"     - AZURE_OPENAI_DEPLOYMENT: {self.azure_openai_deployment}")
                return {"success": False, "error": "Azure OpenAI not configured"}
            
            print(f"   ✅ Azure OpenAI client ready")
            chunks_with_embeddings = self.generate_embeddings(all_chunks)
            
            # Count embedding statistics
            for chunk in chunks_with_embeddings:
                if chunk.get("embedding"):
                    processing_stats["chunks_with_embeddings"] += 1
                else:
                    processing_stats["chunks_without_embeddings"] += 1
            
            # Create search index
            if not self.create_search_index():
                return {"success": False, "error": "Failed to create search index"}
            
            # Upload to search index
            upload_success = self.upload_chunks_to_index(chunks_with_embeddings)
            if not upload_success:
                self.logger.warning("Failed to upload chunks to search index, but continuing with processing")
                # Don't fail completely, just log the warning
                processing_stats["upload_errors"] = True
        
        processing_stats["success"] = True
        return processing_stats
    
    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of similar chunks
        """
        if not self.search_client or not self.openai_client:
            self.logger.error("Azure Search or OpenAI not configured")
            return []
        
        try:
            # HEALRAG only supports vector search
            print(f"   🔍 Using vector search for query: '{query}'")
            return self._vector_search(query, top_k)
                
        except Exception as e:
            self.logger.error(f"Error searching similar chunks: {e}")
            return []
    
    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform vector search using embeddings."""
        try:
            # Generate embedding for the query
            response = self.openai_client.embeddings.create(
                input=query,
                model=self.azure_openai_deployment
            )
            query_embedding = response.data[0].embedding
            
            # Perform vector search
            search_results = self.search_client.search(
                search_text=query,
                vector_queries=[
                    {
                        "vector": query_embedding,
                        "k_nearest_neighbors": top_k,
                        "fields": "embedding",
                        "kind": "vector"
                    }
                ],
                select=["id", "content", "source_file", "section", "chunk_size", "chunk_index"],
                top=top_k
            )
            
            results = []
            for result in search_results:
                results.append({
                    "id": result["id"],
                    "content": result["content"],
                    "source_file": result["source_file"],
                    "section": result["section"],
                    "chunk_size": result["chunk_size"],
                    "chunk_index": result.get("chunk_index", 0),
                    "score": result.get("@search.score", 0),
                    "search_type": "vector"
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in vector search: {e}")
            return []
    
    def _text_search(self, query: str, top_k: int) -> List[Dict]:
        """Perform text-based search."""
        try:
            # Perform text search
            search_results = self.search_client.search(
                search_text=query,
                select=["id", "content", "source_file", "section", "chunk_size", "chunk_index"],
                top=top_k
            )
            
            results = []
            for result in search_results:
                results.append({
                    "id": result["id"],
                    "content": result["content"],
                    "source_file": result["source_file"],
                    "section": result["section"],
                    "chunk_size": result["chunk_size"],
                    "chunk_index": result.get("chunk_index", 0),
                    "score": result.get("@search.score", 0),
                    "search_type": "text"
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in text search: {e}")
            return []