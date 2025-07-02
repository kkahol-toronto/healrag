# HEALRAG - Azure RAG Library

A comprehensive Python library for building RAG (Retrieval-Augmented Generation) applications on Azure with support for Azure Blob Storage and MarkItDown document processing.

## 🚀 FastAPI Application

HEALRAG now includes a comprehensive FastAPI application (`main.py`) that provides REST API endpoints for all functionality including document processing, search indexing, and RAG retrieval.

### Quick Start with API

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server (recommended)
python start_api.py

# Or use uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# For production deployment with Gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
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
    "message": "Training pipeline started successfully",
    "task_id": "training_20250101_123456",
    "status": "started",
    "config": {
        "extract_images": true,
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "recreate_index": false
    }
}
```

#### 📊 GET `/training/status` - Get Training Status
Get the current status of the training pipeline.

**curl command:**
```bash
curl -s http://localhost:8000/training/status | python -m json.tool
```

**Expected Response:**
```json
{
    "status": "completed",
    "task_id": "training_20250101_123456",
    "progress": {
        "current_step": "completed",
        "total_steps": 4,
        "processed_files": 15,
        "total_chunks": 1250,
        "indexed_chunks": 1250
    },
    "last_updated": "2025-01-01T12:45:30.123456"
}
```

#### 🛑 POST `/training/stop` - Stop Training Pipeline
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
    "message": "Training pipeline stopped",
    "status": "stopped"
}
```

### RAG Query Endpoints

#### 🤖 POST `/rag/query` - RAG Question Answering
Ask questions and get AI-powered answers based on your indexed documents. When a `session_id` is provided, the system automatically includes the last 10 conversation exchanges as context for more coherent and contextually aware responses. **Even when no relevant documents are found**, the system will still use conversation history to provide contextual responses.

**curl command:**
```bash
curl -X POST "http://localhost:8000/rag/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "query": "What are the key principles of information security?",
    "session_id": "user123_20250101_session1",
    "top_k": 5,
    "include_sources": true
  }'
```

**Expected Response:**
```json
{
    "query": "What are the key principles of information security?",
    "answer": "The key principles of information security include confidentiality, integrity, and availability (CIA triad). Confidentiality ensures that information is only accessible to authorized individuals...",
    "sources": [
        {
            "source_file": "security_policy.pdf",
            "section": "Core Principles",
            "relevance_score": 0.95,
            "content": "Information security is based on three fundamental principles..."
        }
    ],
    "metadata": {
        "response_time_ms": 1250,
        "sources_used": 3,
        "total_tokens": 1500
    }
}
```

#### 🌊 POST `/rag/stream` - Streaming RAG Response
Get streaming RAG responses for real-time AI interactions. When a `session_id` is provided, the system automatically includes the last 10 conversation exchanges as context for more coherent and contextually aware responses. **Even when no relevant documents are found**, the system will still use conversation history to provide contextual responses.

**curl command:**
```bash
curl -X POST "http://localhost:8000/rag/stream" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "query": "Explain data classification requirements",
    "session_id": "user123_20250101_session1",
    "top_k": 3
  }' \
  --no-buffer
```

**Expected Response (Server-Sent Events):**
```
data: {"type": "start", "query": "Explain data classification requirements"}
data: {"type": "chunk", "content": "Data classification"}
data: {"type": "chunk", "content": " is a critical"}
data: {"type": "chunk", "content": " security practice..."}
data: {"type": "sources", "sources": [{"source_file": "data_classification.pdf", "relevance_score": 0.92}]}
data: {"type": "end", "metadata": {"total_tokens": 1200}}
```

### Search Endpoints

#### 🔍 POST `/search/documents` - Document Search
Search through indexed documents using vector similarity.

**curl command:**
```bash
curl -X POST "http://localhost:8000/search/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "access control policies",
    "top_k": 5,
    "threshold": 0.7
  }'
```

**Expected Response:**
```json
{
    "query": "access control policies",
    "results": [
        {
            "source_file": "access_control_standard.pdf",
            "section": "Policy Framework",
            "content": "Access control policies define who can access what resources...",
            "relevance_score": 0.89,
            "metadata": {
                "chunk_id": "chunk_123",
                "file_type": "pdf"
            }
        }
    ],
    "total_results": 5,
    "search_time_ms": 45
}
```

#### 🧪 GET `/search/test` - Test Search Functionality
Test the search functionality with a predefined query.

**curl command:**
```bash
curl -s http://localhost:8000/search/test | python -m json.tool
```

**Expected Response:**
```json
{
    "message": "Search test completed",
    "test_query": "security policy",
    "results_found": 3,
    "response_time_ms": 125,
    "status": "success"
}
```

### Authentication Endpoints

HEALRAG includes OAuth2-based authentication for secure access to the API.

#### 🔐 GET `/auth/login` - Initiate Login
Start the OAuth2 authentication flow.

**Browser URL:**
```
http://localhost:8000/auth/login
```

#### 🔐 GET `/auth/callback` - OAuth Callback
OAuth2 callback endpoint (handled automatically by the authentication flow).

#### 👤 GET `/auth/me` - Get Current User Info
Get information about the currently authenticated user.

**curl command (with Bearer token):**
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/auth/me
```

**Expected Response:**
```json
{
    "user_id": "user123",
    "email": "user@example.com",
    "name": "John Doe",
    "authenticated": true
}
```

#### 🚪 GET `/auth/logout` - Logout
Log out the current user and invalidate the session.

**Browser URL:**
```
http://localhost:8000/auth/logout
```

#### 🧪 GET `/auth/test-simple` - Test Authentication
Simple endpoint to test if authentication is working.

**curl command:**
```bash
curl -s http://localhost:8000/auth/test-simple
```

### Session Management Endpoints

HEALRAG includes Azure Cosmos DB integration for storing and managing chat sessions and conversation history.

#### 📜 POST `/sessions/history` - Get Session History
Retrieve conversation history for a specific session.

**curl command:**
```bash
curl -X POST "http://localhost:8000/sessions/history" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "user123_20250101_session1",
    "limit": 50,
    "include_metadata": false
  }'
```

**Expected Response:**
```json
{
    "success": true,
    "session_id": "user123_20250101_session1",
    "interactions": [
        {
            "id": "user123_20250101_session1_2025-01-01T10:30:00.123456_1234",
            "sessionID": "user123_20250101_session1",
            "query": "What is our security policy?",
            "response": "Our security policy encompasses several key areas...",
            "user_info": {
                "user_id": "user123",
                "email": "user@company.com",
                "name": "John Doe"
            },
            "sources": [
                {
                    "source_file": "security_policy.pdf",
                    "relevance_score": 0.95
                }
            ],
            "timestamp": "2025-01-01T10:30:00.123456"
        }
    ],
    "total_count": 1,
    "error": null
}
```

#### 👤 GET `/sessions/user` - Get User Sessions
Retrieve all sessions for the current authenticated user.

**curl command:**
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/sessions/user?limit=20"
```

**Expected Response:**
```json
{
    "success": true,
    "user_identifier": "user@company.com",
    "sessions": [
        {
            "sessionID": "user123_20250101_session1",
            "first_interaction": "2025-01-01T10:30:00.123456",
            "last_interaction": "2025-01-01T11:45:00.789012",
            "interaction_count": 15
        },
        {
            "sessionID": "user123_20241231_session2",
            "first_interaction": "2024-12-31T14:20:00.456789",
            "last_interaction": "2024-12-31T15:30:00.123456",
            "interaction_count": 8
        }
    ],
    "total_count": 2
}
```

#### 🗑️ DELETE `/sessions/{session_id}` - Delete Session
Delete a specific session and all its conversation history.

**curl command:**
```bash
curl -X DELETE "http://localhost:8000/sessions/user123_20250101_session1" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Expected Response:**
```json
{
    "success": true,
    "message": "Session user123_20250101_session1 deleted successfully",
    "session_id": "user123_20250101_session1"
}
```

#### 📊 GET `/cosmo/stats` - CosmoDB Statistics
Get statistics about the Cosmos DB chat container.

**curl command:**
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/cosmo/stats
```

**Expected Response:**
```json
{
    "total_interactions": 1250,
    "unique_sessions": 85,
    "unique_users": 12,
    "container_name": "chats",
    "database_name": "healrag-db",
    "partition_key": "/sessionID"
}
```

---

## 🛠️ Local Development Workflow

### Standard Development Process

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment variables
cp .env.example .env  # Edit with your Azure credentials

# 3. Start development server
python start_api.py

# 4. Test the API
curl http://localhost:8000/health/simple

# 5. Run training pipeline (optional)
curl -X POST http://localhost:8000/training/start \
  -H "Content-Type: application/json" \
  -d '{"extract_images": true}'
```

## 🐳 Docker Deployment

HEALRAG includes comprehensive Docker support for local development and Azure cloud deployment.

### 📋 Prerequisites

- Docker Desktop installed and running
- Azure CLI (for Azure deployment)
- Azure Container Registry (ACR) account
- Azure subscription with App Service access

### 🛠️ Docker Files

The project includes several Docker-related files:

- **`Dockerfile`** - Production-ready container with Python 3.11, Gunicorn, and security best practices
- **`.dockerignore`** - Optimizes build performance by excluding unnecessary files
- **`deploy.sh`** - Comprehensive deployment script for local and Azure deployment
- **`requirements.txt`** - Includes all dependencies including FastAPI, Uvicorn, and Gunicorn for production deployment
- **`start_api.py`** - Convenient startup script with customizable server settings

### 🏠 Local Docker Deployment

#### Build and Test Locally

```bash
# Build Docker image
./deploy.sh build

# Run container locally (http://localhost:8000)
./deploy.sh run

# Full local deployment (build + run + test)
./deploy.sh deploy

# View container logs
./deploy.sh logs

# Stop and remove container
./deploy.sh stop
```

#### Manual Docker Commands

```bash
# Build image manually
docker build -t healrag:latest .

# Run container manually
docker run -d --name healrag-container -p 8000:8000 --env-file .env healrag:latest

# View logs
docker logs healrag-container

# Stop container
docker stop healrag-container && docker rm healrag-container
```

### ☁️ Azure Cloud Deployment

The `deploy.sh` script provides comprehensive Azure deployment capabilities with continuous deployment support.

#### 🔐 Azure Prerequisites Setup

1. **Create Azure Container Registry:**
```bash
az acr create --resource-group myResourceGroup --name myregistry --sku Standard --admin-enabled true
```

2. **Get ACR credentials:**
```bash
az acr credential show --name myregistry
```

3. **Add credentials to your `.env` file:**
```bash
AZURE_CONTAINER_REGISTRY=myregistry.azurecr.io
AZURE_CONTAINER_REGISTRY_USERNAME=myregistry
AZURE_CONTAINER_REGISTRY_PASSWORD=your-password-from-step-2
```

#### 🚀 Azure Deployment Commands

##### Full Azure Web App Deployment
Deploy everything: build AMD64 image, push to ACR, create/update App Service, enable continuous deployment:

```bash
./deploy.sh azure-webapp
```

This command will:
- ✅ Build AMD64-compatible Docker image
- ✅ Push to Azure Container Registry with timestamp versioning
- ✅ Create App Service Plan (P3v3 tier by default)
- ✅ Create/update Web App with container configuration
- ✅ Configure all environment variables from your `.env` file
- ✅ Enable continuous deployment with webhook
- ✅ Restart the application and provide testing URLs

##### Custom Azure Deployment
Deploy with custom parameters:

```bash
./deploy.sh azure-webapp-only webapp-name resource-group location plan-name sku

# Example:
./deploy.sh azure-webapp-only healrag-prod myResourceGroup westus2 healrag-plan P2v3
```

##### Update Existing Deployment
Build and push new image version (continuous deployment picks it up automatically):

```bash
./deploy.sh azure-update
```

##### Registry-Only Deployment
Build and push to ACR without webapp changes:

```bash
./deploy.sh azure-deploy
```

#### 🔄 Continuous Deployment Features

The Azure deployment automatically configures:

- **Webhook Integration**: ACR automatically triggers webapp updates
- **Zero-Downtime Deployment**: Azure handles rolling updates
- **Version Tracking**: Each deployment gets timestamped tags
- **Automatic Rollback**: Easy rollback to previous versions
- **Health Monitoring**: Built-in health checks and monitoring

#### 📊 Post-Deployment Verification

After deployment, the script provides URLs for testing:

```bash
# Test deployment
curl https://your-webapp.azurewebsites.net/health/simple

# Full health check
curl https://your-webapp.azurewebsites.net/health

# API documentation
open https://your-webapp.azurewebsites.net/docs
```

### 🔧 Environment Configuration

Your `.env` file is automatically configured for the Azure App Service. Required variables:

#### Core Azure Services
```bash
# Azure Storage
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
AZURE_CONTAINER_NAME=your-container-name

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_KEY=your-openai-key
AZURE_OPENAI_DEPLOYMENT=gpt-4
AZURE_TEXT_EMBEDDING_MODEL=text-embedding-ada-002

# Azure Search
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_KEY=your-search-key
AZURE_SEARCH_INDEX_NAME=your-index-name
```

#### Container Registry (for deployment)
```bash
AZURE_CONTAINER_REGISTRY=yourregistry.azurecr.io
AZURE_CONTAINER_REGISTRY_USERNAME=yourregistry
AZURE_CONTAINER_REGISTRY_PASSWORD=your-acr-password
```

#### Application Configuration
```bash
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
LOG_LEVEL=INFO
```

### 🎯 Deployment Best Practices

#### Security
- ✅ Non-root user in container
- ✅ Minimal base image (Python 3.11 slim)
- ✅ Environment variables for sensitive data
- ✅ Container registry authentication

#### Performance
- ✅ AMD64 architecture for Azure compatibility
- ✅ Optimized layer caching with `.dockerignore`
- ✅ Gunicorn with async workers for production

#### Reliability
- ✅ Health checks built into container
- ✅ Graceful error handling and logging
- ✅ Automatic restart policies
- ✅ Continuous deployment with rollback capability

### 🐛 Troubleshooting

#### Common Issues

**1. Architecture Mismatch Error:**
```
ERROR - no matching manifest for linux/amd64
```
**Solution:** Use `./deploy.sh azure-webapp` which builds AMD64-compatible images.

**2. Authentication Failed:**
```
ERROR - unauthorized: authentication required
```
**Solution:** Verify ACR credentials in `.env` file and run `./deploy.sh azure-validate`.

**3. Application Error on Azure:**
```
:( Application Error
```
**Solution:** Check environment variables are properly set and view logs:
```bash
az webapp log tail --name your-webapp --resource-group your-rg
```

#### Deployment Script Help

```bash
# View all available commands
./deploy.sh

# Validate Azure environment
./deploy.sh azure-validate

# Test Azure CLI connection
./deploy.sh azure-login
```

### 📈 Monitoring and Scaling

#### Application Insights Integration
Add to your `.env` for enhanced monitoring:
```bash
APPLICATIONINSIGHTS_CONNECTION_STRING=your-app-insights-connection
```

#### Scaling Options
```bash
# Scale up (increase VM size)
az appservice plan update --name healrag-plan --resource-group myRG --sku P3v3

# Scale out (increase instance count)
az webapp update --name healrag-security --resource-group myRG --instance-count 3
```

---

## 🚀 Key Features

- **Comprehensive FastAPI Application**: Complete REST API with health checks, training pipelines, RAG querying, and search endpoints
- **Azure Blob Storage Integration**: Full file management with upload, download, and statistics
- **Document Processing**: MarkItDown integration for extracting content from PDFs, Office docs, images, and more
- **RAG System**: Advanced question-answering with streaming responses, source attribution, and conversation context
- **Conversation Context**: Automatically includes last 10 conversation exchanges for contextually aware responses
- **Chat History & Session Management**: Azure Cosmos DB integration for storing conversation history with session-based organization
- **Search Indexing**: Vector embeddings with Azure Cognitive Search for semantic document retrieval
- **Docker & Cloud Ready**: Production-ready containerization with Azure App Service deployment
- **Continuous Deployment**: Automated CI/CD pipeline with webhooks and zero-downtime updates
- **Authentication**: OAuth2-based security for API access control
- **Monitoring**: Comprehensive health checks, logging, and error handling

## 📦 Installation & Setup

### Prerequisites
- Python 3.11 or higher
- Azure Storage Account
- Azure OpenAI Service
- Azure Cognitive Search Service (optional)
- Docker Desktop (for containerization)

### Quick Installation

```bash
# Clone the repository
git clone <repository-url>
cd HealRag

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env  # Edit with your Azure credentials

# Start the API server
python start_api.py
```

### Environment Setup
Create a `.env` file with your Azure credentials:

```bash
# Required: Azure Storage
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=...
AZURE_CONTAINER_NAME=your-container-name

# Required: Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_KEY=your-openai-key
AZURE_OPENAI_DEPLOYMENT=gpt-4
AZURE_TEXT_EMBEDDING_MODEL=text-embedding-ada-002

# Optional: Azure Search
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_KEY=your-search-key
AZURE_SEARCH_INDEX_NAME=healrag-index

# Optional: Azure Cosmos DB (for chat history)
AZURE_COSMO_CONNECTION_STRING=AccountEndpoint=https://your-account.documents.azure.com:443/;AccountKey=your-key;
AZURE_COSMO_DB_NAME=your-database-name
AZURE_COSMO_DB_CONTAINER=chats
```

## 📖 Python Library Usage

### Basic Storage Operations

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
python healraglib/tests/run_tests.py

# Run specific test files
python -m pytest healraglib/tests/test_storage_manager.py
python -m pytest healraglib/tests/test_cli.py
python -m pytest healraglib/tests/test_content_manager.py

# Run with pytest (if installed)
pytest healraglib/tests/
```

The tests use mocks to avoid requiring actual Azure credentials, making them safe to run in any environment.

## 📄 Supported File Types

HEALRAG uses MarkItDown to extract content from various file formats:

- **Documents**: PDF, PowerPoint (.pptx, .ppt), Word (.docx, .doc), Excel (.xlsx, .xls)
- **Images**: JPG, JPEG, PNG, GIF, BMP, TIFF (with EXIF metadata and OCR)
- **Audio**: MP3, WAV, M4A (with speech transcription)
- **Web**: HTML, HTM
- **Data**: CSV, JSON, XML
- **Archives**: ZIP files (iterates over contents)
- **E-books**: EPub
- **Text**: TXT files
- **Media**: YouTube URLs

## 🔧 Recent Improvements

### Search Index Manager Enhancements
- **JSON Upload Fix**: Resolved embedding field validation errors
- **Document Validation**: Comprehensive structure validation before upload
- **Embedding Format Validation**: Ensures valid 1536-dimensional vectors
- **Graceful Error Recovery**: Continues processing despite individual failures
- **Enhanced Logging**: Better debugging and monitoring capabilities
- **Batch Processing**: Improved upload with individual error tracking

### FastAPI Application
- **Complete Endpoint Coverage**: All CRUD operations for documents, training, and RAG
- **Streaming Responses**: Real-time RAG responses with Server-Sent Events
- **Authentication Integration**: OAuth2-based security system
- **Comprehensive Error Handling**: Proper HTTP status codes and validation
- **Health Monitoring**: Detailed component status and configuration endpoints

### Docker & Deployment
- **Production-Ready Containers**: Multi-stage builds with security best practices
- **Azure App Service Integration**: Full cloud deployment with continuous delivery
- **Environment Variable Management**: Automatic configuration from .env files
- **Webhook Integration**: Zero-downtime deployments with ACR integration

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run the test suite: `python healraglib/tests/run_tests.py`
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 💬 Support

For support and questions, please open an issue on the GitHub repository. 