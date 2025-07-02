# HEALRAG - Azure RAG Library

A comprehensive Python library for building RAG (Retrieval-Augmented Generation) applications on Azure with support for Azure Blob Storage and MarkItDown document processing.

## 🚀 FastAPI Application

HEALRAG now includes a comprehensive FastAPI application (`main.py`) that provides REST API endpoints for all functionality including document processing, search indexing, and RAG retrieval.

### Quick Start with API

```bash
# Install FastAPI dependencies
pip install -r requirements.txt

# Start the API server
python start_api.py

# Or use uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Root**: http://localhost:8000/

## 📚 API Endpoints Documentation

### Health & Configuration Endpoints

#### 🟢 GET `/health` - Comprehensive Health Check
Get detailed system health information including component status and configuration.

**curl command:**
```bash
curl -s http://localhost:8000/health | python -m json.tool
```

**Expected Response:**
```json
{
    "status": "healthy",
    "timestamp": "2025-07-01T20:54:57.288782",
    "components": {
        "storage_manager": true,
        "content_manager": true,
        "search_manager": true,
        "llm_manager": true,
        "rag_manager": true,
        "azure_storage": true,
        "azure_openai": true,
        "rag_system": true
    },
    "configuration": {
        "azure_openai_endpoint": "https://healthmedical.openai.azure.com/",
        "azure_openai_chat_deployment": "gpt-4.1",
        "azure_openai_embedding_deployment": "text-embedding-ada-002",
        "azure_search_endpoint": "https://point32search.search.windows.net",
        "azure_search_index_name": "security-index",
        "azure_storage_container": "security-documents",
        "chunk_size": 1000,
        "chunk_overlap": 200
    }
}
```

#### 🟢 GET `/health/simple` - Simple Health Check
Basic health check endpoint for load balancers.

**curl command:**
```bash
curl -s http://localhost:8000/health/simple
```

**Expected Response:**
```json
{"status": "healthy"}
```

#### ⚙️ GET `/config` - Get System Configuration
Retrieve current system configuration and component status.

**curl command:**
```bash
curl -s http://localhost:8000/config | python -m json.tool
```

**Expected Response:**
```json
{
    "azure_openai_endpoint": "https://healthmedical.openai.azure.com/",
    "azure_openai_chat_deployment": "gpt-4.1",
    "azure_openai_embedding_deployment": "text-embedding-ada-002",
    "azure_search_endpoint": "https://point32search.search.windows.net",
    "azure_search_index_name": "security-index",
    "azure_storage_container": "security-documents",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "components": {
        "storage_manager": true,
        "content_manager": true,
        "search_manager": true,
        "llm_manager": true,
        "rag_manager": true
    },
    "rag_settings": {
        "default_top_k": 3,
        "max_context_tokens": 6000,
        "include_source_info": true,
        "relevance_threshold": 0.02
    }
}
```

#### ⚙️ POST `/config/reload` - Reload Configuration
Reload system configuration and reinitialize components.

**curl command:**
```bash
curl -X POST "http://localhost:8000/config/reload" \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Expected Response:**
```json
{
    "message": "Configuration reloaded successfully",
    "status": "success"
}
```

#### 📊 GET `/storage/stats` - Get Azure Storage Statistics
Get comprehensive statistics about the Azure Storage container.

**curl command:**
```bash
curl -s http://localhost:8000/storage/stats | python -m json.tool
```

**Expected Response:**
```json
{
    "total_files": 20,
    "total_size_bytes": 2582099,
    "file_types": {
        ".pdf": {
            "count": 9,
            "total_size_bytes": 2376784,
            "total_size_mb": 2.27,
            "total_size_gb": 0.0
        },
        ".txt": {
            "count": 1,
            "total_size_bytes": 955
        },
        ".md": {
            "count": 10,
            "total_size_bytes": 204360
        }
    },
    "container_name": "security-documents",
    "total_size_mb": 2.46,
    "total_size_gb": 0.0,
    "last_modified": "2025-07-01T23:35:54+00:00"
}
```

### Training Pipeline Endpoints

#### 🏗️ POST `/training/start` - Start Training Pipeline
Start the document processing and indexing pipeline.

**Parameters:**
- `recreate_index` (query param): Whether to recreate the search index (default: false)
- Request body: TrainingRequest JSON

**curl command (without index recreation):**
```bash
curl -X POST "http://localhost:8000/training/start?recreate_index=false" \
  -H "Content-Type: application/json" \
  -d '{
    "extract_images": true,
    "chunk_size": 1000,
    "chunk_overlap": 200
  }'
```

**curl command (with index recreation):**
```bash
curl -X POST "http://localhost:8000/training/start?recreate_index=true" \
  -H "Content-Type: application/json" \
  -d '{
    "extract_images": true,
    "chunk_size": 1000,
    "chunk_overlap": 200
  }'
```

**Expected Response:**
```json
{
    "message": "Training pipeline started",
    "status": "running"
}
```

#### 📊 GET `/training/status` - Get Training Pipeline Status
Check the current status and progress of the training pipeline.

**curl command:**
```bash
curl -s http://localhost:8000/training/status | python -m json.tool
```

**Expected Response (while running):**
```json
{
    "status": "running",
    "message": "Processing files...",
    "start_time": "2025-07-01T20:56:34.959742",
    "end_time": null,
    "progress": {
        "step": 5,
        "current_task": "Processing markdown files"
    },
    "results": {}
}
```

**Expected Response (completed):**
```json
{
    "status": "completed",
    "message": "Training pipeline completed successfully",
    "start_time": "2025-07-01T20:56:34.959742",
    "end_time": "2025-07-01T20:57:25.182280",
    "progress": {
        "step": 10,
        "current_task": "Completed"
    },
    "results": {
        "container_stats": {
            "total_files": 20,
            "file_types": {".pdf": {"count": 9}, ".md": {"count": 10}}
        },
        "supported_files_count": 10,
        "content_extraction": {
            "files_processed": 10,
            "total_files": 10,
            "processing_time": 39.93
        },
        "search_indexing": {
            "files_processed": 10,
            "total_chunks": 32,
            "chunks_with_embeddings": 32,
            "success": true
        },
        "search_test": {
            "query": "cyber security policy",
            "results_found": 3,
            "success": true
        },
        "rag_test": {
            "success": true,
            "test_query": "What is our incident management process?"
        }
    }
}
```

#### ⛔ POST `/training/stop` - Stop Training Pipeline
Stop the currently running training pipeline.

**curl command:**
```bash
curl -X POST "http://localhost:8000/training/stop" \
  -H "Content-Type: application/json" \
  -d '{}'
```

**Expected Response:**
```json
{
    "detail": "No training pipeline is currently running"
}
```

### RAG Retrieval Endpoints

#### 🤖 POST `/rag/query` - Generate RAG Response (Non-streaming)
Generate a comprehensive response to a query using RAG (Retrieval-Augmented Generation).

**Request Body Parameters:**
- `query` (required): The question or query to answer
- `top_k` (optional): Number of documents to retrieve (1-20, default: 3)
- `temperature` (optional): LLM temperature (0.0-2.0, default: 0.7)
- `max_tokens` (optional): Maximum response tokens (50-2000, default: 500)
- `custom_system_prompt` (optional): Custom system prompt
- `include_search_details` (optional): Include search metadata (default: false)

**curl command:**
```bash
curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the key principles of our cyber security policy?",
    "top_k": 3,
    "temperature": 0.7,
    "max_tokens": 500,
    "include_search_details": false
  }'
```

**Expected Response:**
```json
{
    "success": true,
    "response": "Based on the provided context, the key principles of Point32Health's Cyber & Information Security Policy are:\n\n1. **Protection of Information Assets:** All information assets must be protected against unauthorized disclosure, misuse, modification, compromise, corruption, destruction, and disruption...",
    "sources": [
        {
            "document_id": "Cyber___Information_Security_Policy_0000",
            "source_file": "md_files/Cyber & Information Security Policy.md",
            "section": "Extracted Images",
            "score": 0.03333333507180214,
            "chunk_index": 0,
            "content_preview": "# tmpg4u6jxzv\n\n**Source:** /var/folders/p6/k_w2lvlx3jv1d7cv8dwc86440000gn/T/tmpg4u6jxzv.pdf..."
        }
    ],
    "metadata": {
        "retrieval": {
            "documents_found": 3,
            "documents_used": 3,
            "retrieval_time": 0.244,
            "top_k": 3
        },
        "generation": {
            "model": "gpt-4.1",
            "temperature": 0.7,
            "generation_time": 3.821,
            "usage": {
                "prompt_tokens": 3534,
                "completion_tokens": 500,
                "total_tokens": 4034
            },
            "finish_reason": "length"
        },
        "total_time": 4.065,
        "timestamp": "2025-07-01T20:57:49.453842",
        "context_length": 18716
    },
    "error": null
}
```

#### 🌊 POST `/rag/stream` - Generate Streaming RAG Response
Generate a streaming response using Server-Sent Events (SSE) for real-time output.

**Request Body Parameters:** Same as `/rag/query`

**curl command:**
```bash
curl -X POST "http://localhost:8000/rag/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is our password policy?",
    "top_k": 2,
    "temperature": 0.7,
    "max_tokens": 300
  }' --no-buffer
```

**Expected Response (Server-Sent Events):**
```
data: {"type": "rag_start", "status": "starting_retrieval", "query": "What is our password policy?", "timestamp": "2025-07-01T20:57:59.781723"}

data: {"type": "retrieval_complete", "documents_found": 3, "retrieval_time": 0.253}

data: {"type": "generation_start", "status": "starting_generation"}

data: {"type": "chunk", "content": "Based", "chunk_index": 1, "accumulated_content": "Based"}

data: {"type": "chunk", "content": " on", "chunk_index": 2, "accumulated_content": "Based on"}

data: {"type": "chunk", "content": " the", "chunk_index": 3, "accumulated_content": "Based on the"}

...

data: {"type": "complete", "status": "completed", "full_response": "Based on the provided context...", "finish_reason": "stop"}

data: {"type": "stream_complete"}
```

### Search Endpoints

#### 🔍 POST `/search/documents` - Search Documents
Search for documents using vector similarity search.

**Request Body Parameters:**
- `query` (required): Search query string
- `top_k` (optional): Number of results to return (1-20, default: 5)

**curl command:**
```bash
curl -X POST "http://localhost:8000/search/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "access management authentication",
    "top_k": 2
  }'
```

**Expected Response:**
```json
{
    "success": true,
    "results": [
        {
            "id": "C_IS_007_Cyber___Information_Security_Identity__Credential__and_Access_Management_Standard_0000",
            "content": "# tmp5nmzcnqa\n\n**Source:** /var/folders/p6/k_w2lvlx3jv1d7cv8dwc86440000gn/T/tmp5nmzcnqa.pdf\n\n**Extracted:** 2025-07-01T20:59:33.049475\n\nC&IS 007 – IDENTITY, CREDENTIAL & ACCESS MANAGEMENT SECURITY STANDARD...",
            "source_file": "md_files/C&IS 007_Cyber & Information Security Identity, Credential, and Access Management Standard.md",
            "section": "Extracted Images",
            "chunk_size": 31577,
            "chunk_index": 0,
            "score": 0.03333333507180214,
            "search_type": "vector"
        }
    ],
    "metadata": {
        "documents_found": 2,
        "search_time": 0.315,
        "top_k": 2,
        "relevance_threshold": 0.02,
        "timestamp": "2025-07-01T21:02:34.068258"
    },
    "error": null
}
```

#### 🧪 GET `/search/test` - Test Search Functionality
Test search functionality with a predefined query.

**curl command:**
```bash
curl -s http://localhost:8000/search/test | python -m json.tool
```

**Expected Response:**
```json
{
    "success": true,
    "query": "cyber security policy",
    "results_count": 3,
    "results": [
        {
            "id": "Cyber___Information_Security_Policy_0000",
            "content": "CYBER & INFORMATION SECURITY POLICY...",
            "source_file": "md_files/Cyber & Information Security Policy.md",
            "score": 0.03333333507180214
        }
    ]
}
```

### Root Endpoint

#### 🏠 GET `/` - API Information
Get basic information about the API and available endpoints.

**curl command:**
```bash
curl -s http://localhost:8000/ | python -m json.tool
```

**Expected Response:**
```json
{
    "message": "HEALRAG API",
    "version": "1.0.0",
    "description": "Comprehensive API for the HEALRAG system",
    "endpoints": {
        "health": "/health",
        "training": "/training/start",
        "rag_query": "/rag/query",
        "rag_stream": "/rag/stream",
        "search": "/search/documents",
        "docs": "/docs"
    }
}
```

## 🔧 API Usage Tips

### Testing Workflow
1. **Start with health check**: `GET /health` to verify all components are working
2. **Check storage**: `GET /storage/stats` to see available documents
3. **Run training pipeline**: `POST /training/start` to process documents
4. **Monitor progress**: `GET /training/status` to track completion
5. **Test search**: `GET /search/test` to verify search is working
6. **Try RAG queries**: `POST /rag/query` for question answering

### Training Pipeline Behavior
- **Without index recreation (`recreate_index=false`)**: ~50 seconds processing time
- **With index recreation (`recreate_index=true`)**: ~63 seconds processing time
- Both modes automatically handle index creation and document processing
- The system processes PDFs, extracts content, generates embeddings, and creates search indexes

### Error Handling
All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid parameters)
- `500`: Internal server error
- `422`: Validation error (Pydantic model validation failed)

### Performance Notes
- RAG queries typically take 2-5 seconds depending on context size
- Streaming responses provide real-time output for better user experience
- Search operations are optimized for sub-second response times
- Training pipeline processes 10-20 documents in under 2 minutes

### Environment Variables for API

Create a `.env` file with the following variables:

```env
# Server Configuration
HOST=0.0.0.0
PORT=8000
RELOAD=true
WORKERS=1
LOG_LEVEL=info

# Azure Storage
AZURE_STORAGE_CONNECTION_STRING=your_connection_string
AZURE_CONTAINER_NAME=healrag-documents

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your_openai_key
AZURE_OPENAI_DEPLOYMENT=gpt-4
AZURE_TEXT_EMBEDDING_MODEL=text-embedding-ada-002

# Azure Cognitive Search
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_KEY=your_search_key
AZURE_SEARCH_INDEX_NAME=healrag-index

# Processing Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## Features

- **Azure Blob Storage Integration**: Complete management of files in Azure Blob Storage containers
- **Document Processing**: Integration with MarkItDown for extracting content from various file formats
- **Image Extraction & Analysis**: Extract images from PDFs and Word documents with Azure OpenAI Vision analysis
- **Markdown Generation**: Generate comprehensive markdown files with embedded image descriptions and metadata
- **Search Index Management**: Azure Cognitive Search integration with vector embeddings and semantic search
- **File Operations**: Upload, download, and manage files with comprehensive statistics
- **Bulk Operations**: Support for uploading/downloading entire folders or containers
- **Connection Management**: Robust connection verification and error handling
- **Error Recovery**: Graceful handling of search index upload errors with detailed logging

## Supported File Types

HEALRAG uses MarkItDown to extract content from the following file types:

- **Documents**: PDF, PowerPoint (.pptx, .ppt), Word (.docx, .doc), Excel (.xlsx, .xls)
- **Images**: JPG, JPEG, PNG, GIF, BMP, TIFF (with EXIF metadata and OCR)
- **Audio**: MP3, WAV, M4A (with EXIF metadata and speech transcription)
- **Web**: HTML, HTM
- **Data**: CSV, JSON, XML
- **Archives**: ZIP files (iterates over contents)
- **E-books**: EPub
- **Text**: TXT files
- **Media**: YouTube URLs

## Recent Fixes and Improvements

### Search Index Manager Enhancements (Latest)

The SearchIndexManager has been enhanced with improved error handling and validation:

- **JSON Upload Fix**: Fixed issue with embedding field validation that was causing "StartArray" JSON parsing errors
- **Document Validation**: Added comprehensive validation of document structure before upload
- **Embedding Format Validation**: Ensures embeddings are valid 1536-dimensional vectors before upload
- **Graceful Error Recovery**: Continues processing even if some chunks fail to upload
- **Detailed Logging**: Enhanced logging for better debugging and monitoring
- **Batch Processing**: Improved batch upload with individual error tracking

### Error Handling Improvements

- Better handling of missing or invalid embeddings
- Graceful degradation when search index upload fails
- Detailed error reporting for troubleshooting
- Validation of search index schema compatibility

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd HealRag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from healraglib import StorageManager

# Initialize the storage manager
connection_string = "your_azure_storage_connection_string"
container_name = "your_container_name"

storage_manager = StorageManager(connection_string, container_name)

# Verify connection
if storage_manager.verify_connection():
    print("Successfully connected to Azure Blob Storage!")
else:
    print("Failed to connect to Azure Blob Storage")
```

### Container Statistics

```python
# Get comprehensive statistics about the container
stats = storage_manager.get_container_statistics()
print(f"Total files: {stats['total_files']}")
print(f"Total size: {stats['total_size_mb']} MB")
print(f"File types: {list(stats['file_types'].keys())}")
```

### File Operations

```python
# Upload a single file
success = storage_manager.upload_file("local_file.pdf", "remote_file.pdf")
print(f"Upload successful: {success}")

# Upload an entire folder
results = storage_manager.upload_folder("local_folder", prefix="documents")
print(f"Uploaded {sum(results.values())} files successfully")

# Get list of files
file_list = storage_manager.get_file_list(as_json=False)
print(f"Files in container: {file_list}")

# Download a file
success = storage_manager.download_file("remote_file.pdf", "downloaded_file.pdf")
print(f"Download successful: {success}")

# Download entire container
results = storage_manager.download_entire_container("downloaded_container")
print(f"Downloaded {sum(results.values())} files")
```

### Document Content Extraction and Markdown Generation

```python
from healraglib import StorageManager
from healraglib.content_manager import ContentManager

# Initialize managers
storage_manager = StorageManager(connection_string, container_name)
content_manager = ContentManager(
    storage_manager=storage_manager,
    azure_openai_endpoint=azure_openai_endpoint,
    azure_openai_key=azure_openai_key,
    azure_openai_deployment=azure_openai_deployment
)

# Extract content and generate markdown files
results = content_manager.extract_content_from_files(
    file_list=["document1.pdf", "presentation.pptx"],
    output_folder="md_files",
    image_prompt="Describe this image in detail, including any text, charts, diagrams, or visual elements that would be important for understanding the document content.",
    extract_images=True
)

# Check results
for file_path, result in results.items():
    if result["success"]:
        print(f"✅ Processed {file_path}")
        print(f"   Markdown: {result['markdown_file']}")
        print(f"   Images: {result['images_processed']}")
    else:
        print(f"❌ Failed: {result.get('error')}")
```

## API Reference

### StorageManager Class

#### Constructor
```python
StorageManager(connection_string: str, container_name: str)
```

#### Methods

##### Connection Management
- `verify_connection() -> bool`: Verify connection to Azure Blob Storage

##### Statistics and Information
- `get_container_statistics() -> Dict`: Get comprehensive container statistics
- `get_file_list(as_json: bool = True) -> Union[List[str], str]`: Get list of files
- `get_supported_file_types() -> List[str]`: Get supported file types

##### Upload Operations
- `upload_file(local_file_path: str, blob_name: Optional[str] = None) -> bool`: Upload single file
- `upload_folder(local_folder_path: str, prefix: str = "") -> Dict[str, bool]`: Upload entire folder

##### Download Operations
- `download_file(blob_name: str, local_file_path: str) -> bool`: Download single file
- `download_folder(blob_prefix: str, local_folder_path: str) -> Dict[str, bool]`: Download folder by prefix
- `download_entire_container(local_folder_path: str) -> Dict[str, bool]`: Download entire container

##### Content Extraction
- `get_file_content_with_markitdown(blob_name: str) -> str`: Extract content using MarkItDown

### ContentManager Class

#### Constructor
```python
ContentManager(storage_manager, azure_openai_endpoint=None, azure_openai_key=None, azure_openai_deployment=None)
```

#### Methods

##### Content Processing
- `extract_content_from_files(file_list, output_folder="md_files", image_prompt="...", extract_images=True) -> Dict`: Extract content and generate markdown files
- `get_supported_file_types() -> List[str]`: Get supported file types for content extraction
- `get_image_extraction_support() -> List[str]`: Get file types that support image extraction

### SearchIndexManager Class

#### Constructor
```python
SearchIndexManager(storage_manager, azure_openai_endpoint=None, azure_openai_key=None, azure_openai_deployment=None, azure_search_endpoint=None, azure_search_key=None, azure_search_index_name="healrag-index", chunk_size=1000, chunk_overlap=200)
```

#### Methods

##### Index Management
- `create_search_index() -> bool`: Create or verify Azure Cognitive Search index
- `upload_chunks_to_index(chunks: List[Dict]) -> bool`: Upload chunks with embeddings to search index

##### Content Processing
- `chunk_markdown_content(content: str, source_file: str) -> List[Dict]`: Chunk markdown content into smaller pieces
- `generate_embeddings(chunks: List[Dict]) -> List[Dict]`: Generate embeddings for text chunks using Azure OpenAI
- `process_markdown_files(md_folder: str = "md_files") -> Dict`: Process all markdown files for search indexing

##### Search Operations
- `search_similar_chunks(query: str, top_k: int = 5) -> List[Dict]`: Search for similar chunks using vector similarity

## Configuration

### Environment Variables

Set the following environment variables:

```bash
# Required
export AZURE_STORAGE_CONNECTION_STRING="your_connection_string"
export AZURE_CONTAINER_NAME="your_container_name"

# Optional: For Azure OpenAI integration (image analysis and embeddings)
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_KEY="your_openai_key"
export AZURE_OPENAI_DEPLOYMENT="your_deployment_name"
export AZURE_TEXT_EMBEDDING_MODEL="text-embedding-ada-002"

# Optional: For Azure Cognitive Search integration
export AZURE_SEARCH_ENDPOINT="https://your-search-service.search.windows.net"
export AZURE_SEARCH_KEY="your_search_key"
export AZURE_SEARCH_INDEX_NAME="healrag-index"

# Optional: Search index configuration
export CHUNK_SIZE="1000"
export CHUNK_OVERLAP="200"


```

### Azure Storage Connection String

You can get your Azure Storage connection string from the Azure Portal:

1. Go to your Storage Account
2. Navigate to "Access keys"
3. Copy the connection string

Example connection string format:
```
DefaultEndpointsProtocol=https;AccountName=your_account;AccountKey=your_key;EndpointSuffix=core.windows.net
```

### Container Setup

1. Create a container in your Azure Storage Account
2. Ensure your connection string has appropriate permissions
3. The container name should match what you specify in the StorageManager

## Error Handling

The library includes comprehensive error handling:

```python
try:
    storage_manager = StorageManager(connection_string, container_name)
    if not storage_manager.verify_connection():
        print("Connection failed - check your connection string and container name")
        exit(1)
        
    # Your operations here
    
except ConnectionError as e:
    print(f"Connection error: {e}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Logging

The library uses Python's logging module. You can configure logging as needed:

```python
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Examples

### Complete RAG Pipeline Example

```python
from healraglib import StorageManager
from healraglib.content_manager import ContentManager
from healraglib.search_index_manager import SearchIndexManager
import os

# Initialize managers
storage_manager = StorageManager(connection_string, container_name)
content_manager = ContentManager(
    storage_manager=storage_manager,
    azure_openai_endpoint=azure_openai_endpoint,
    azure_openai_key=azure_openai_key,
    azure_openai_deployment=azure_openai_deployment
)
search_manager = SearchIndexManager(
    storage_manager=storage_manager,
    azure_openai_endpoint=azure_openai_endpoint,
    azure_openai_key=azure_openai_key,
    azure_openai_deployment=azure_text_embedding_model,
    azure_search_endpoint=azure_search_endpoint,
    azure_search_key=azure_search_key,
    azure_search_index_name=azure_search_index_name
)

# Step 1: Upload documents
storage_manager.upload_folder("documents", prefix="corpus")

# Step 2: Extract content and generate markdown
supported_files = content_manager.get_source_files_from_container()
results = content_manager.extract_content_from_files(
    supported_files, 
    output_folder="md_files", 
    extract_images=True
)

# Step 3: Process markdown files for search indexing
index_results = search_manager.process_markdown_files("md_files")

if index_results.get('success'):
    print(f"✅ Indexed {index_results['chunks_with_embeddings']} chunks with embeddings")
    
    # Step 4: Search for similar content
    query = "cyber security policy"
    search_results = search_manager.search_similar_chunks(query, top_k=5)
    
    print(f"Search results for '{query}':")
    for i, result in enumerate(search_results, 1):
        print(f"{i}. {result['source_file']} (Score: {result['score']:.3f})")
        print(f"   Section: {result['section']}")
        print(f"   Content: {result['content'][:100]}...")
else:
    print(f"❌ Search indexing failed: {index_results.get('error')}")
```

### Search Index Management Example

```python
from healraglib.search_index_manager import SearchIndexManager

# Initialize search index manager
search_manager = SearchIndexManager(
    storage_manager=storage_manager,
    azure_openai_endpoint=azure_openai_endpoint,
    azure_openai_key=azure_openai_key,
    azure_openai_deployment=azure_text_embedding_model,
    azure_search_endpoint=azure_search_endpoint,
    azure_search_key=azure_search_key,
    azure_search_index_name="my-search-index"
)

# Create or verify search index
if search_manager.create_search_index():
    print("✅ Search index ready")
    
    # Process markdown files
    results = search_manager.process_markdown_files("md_files")
    print(f"Processed {results['files_processed']} files")
    print(f"Created {results['total_chunks']} chunks")
    print(f"Generated {results['chunks_with_embeddings']} embeddings")
    
    # Search for similar content
    query = "information security"
    results = search_manager.search_similar_chunks(query, top_k=3)
    
    for result in results:
        print(f"Found: {result['source_file']} - {result['section']}")
else:
    print("❌ Failed to create search index")
```

## Testing

The HEALRAG library includes comprehensive unit tests. To run the tests:

```bash
# Run all tests
python tests/run_tests.py

# Run specific test files
python tests/test_storage_manager.py
python tests/test_cli.py

# Run with pytest (if installed)
pytest tests/
```

The tests use mocks to avoid requiring actual Azure credentials, making them safe to run in any environment.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run the test suite: `python tests/run_tests.py`
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions, please open an issue on the GitHub repository. 