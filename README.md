# HEALRAG - Azure RAG Library

A comprehensive Python library for building RAG (Retrieval-Augmented Generation) applications on Azure with support for Azure Blob Storage and MarkItDown document processing.

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