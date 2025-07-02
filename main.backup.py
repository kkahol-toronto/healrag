#!/usr/bin/env python3
"""
HEALRAG FastAPI Application
===========================

A comprehensive FastAPI application for the HEALRAG system including:
- Health check endpoints
- Training pipeline execution
- RAG retrieval (streaming and non-streaming)
- Document search
- System configuration management

Based on training_pipeline.py and rag_pipeline.py
"""

import os
import json
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import HEALRAG components
from healraglib import StorageManager, RAGManager, LLMManager, SearchIndexManager
from healraglib.content_manager import ContentManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress Azure SDK verbose logging
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.ERROR)
logging.getLogger('azure.storage.blob').setLevel(logging.ERROR)

# FastAPI app initialization
app = FastAPI(
    title="HEALRAG API",
    description="Comprehensive API for the HEALRAG system - document processing, indexing, and RAG retrieval",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for component instances
storage_manager: Optional[StorageManager] = None
content_manager: Optional[ContentManager] = None
search_manager: Optional[SearchIndexManager] = None
llm_manager: Optional[LLMManager] = None
rag_manager: Optional[RAGManager] = None

# Training pipeline status tracking
training_status = {
    "status": "idle",  # idle, running, completed, failed
    "message": "",
    "start_time": None,
    "end_time": None,
    "progress": {},
    "results": {}
}

# Pydantic models for request/response
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, bool]
    configuration: Dict[str, Any]

class TrainingRequest(BaseModel):
    container_name: Optional[str] = None
    extract_images: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200

class TrainingStatusResponse(BaseModel):
    status: str
    message: str
    start_time: Optional[str]
    end_time: Optional[str]
    progress: Dict[str, Any]
    results: Dict[str, Any]

class RAGRequest(BaseModel):
    query: str = Field(..., description="The question or query to answer")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of documents to retrieve")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(default=500, ge=50, le=2000, description="Maximum response tokens")
    custom_system_prompt: Optional[str] = Field(None, description="Custom system prompt")
    include_search_details: bool = Field(default=False, description="Include search metadata")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to return")

class RAGResponse(BaseModel):
    success: bool
    response: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    error: Optional[str] = None

class SearchResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    error: Optional[str] = None

def get_configuration() -> Dict[str, Any]:
    """Get current system configuration."""
    return {
        "azure_openai_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
        "azure_openai_chat_deployment": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        "azure_openai_embedding_deployment": os.getenv("AZURE_TEXT_EMBEDDING_MODEL"),
        "azure_search_endpoint": os.getenv("AZURE_SEARCH_ENDPOINT"),
        "azure_search_index_name": os.getenv("AZURE_SEARCH_INDEX_NAME", "healrag-index"),
        "azure_storage_container": os.getenv("AZURE_CONTAINER_NAME", "healrag-documents"),
        "chunk_size": int(os.getenv("CHUNK_SIZE", "1000")),
        "chunk_overlap": int(os.getenv("CHUNK_OVERLAP", "200"))
    }

async def initialize_components():
    """Initialize all HEALRAG components."""
    global storage_manager, content_manager, search_manager, llm_manager, rag_manager
    
    try:
        config = get_configuration()
        
        # Initialize Storage Manager
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
        if connection_string:
            storage_manager = StorageManager(
                connection_string, 
                config["azure_storage_container"]
            )
            logger.info("Storage Manager initialized")
        
        # Initialize Content Manager
        if storage_manager:
            content_manager = ContentManager(
                storage_manager=storage_manager,
                azure_openai_endpoint=config["azure_openai_endpoint"],
                azure_openai_key=os.getenv("AZURE_OPENAI_KEY"),
                azure_openai_deployment=config["azure_openai_chat_deployment"]
            )
            logger.info("Content Manager initialized")
        
        # Initialize Search Index Manager
        if all([config["azure_search_endpoint"], os.getenv("AZURE_SEARCH_KEY")]):
            search_manager = SearchIndexManager(
                storage_manager=storage_manager,
                azure_openai_endpoint=config["azure_openai_endpoint"],
                azure_openai_key=os.getenv("AZURE_OPENAI_KEY"),
                azure_openai_deployment=config["azure_openai_embedding_deployment"],
                azure_search_endpoint=config["azure_search_endpoint"],
                azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
                azure_search_index_name=config["azure_search_index_name"],
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"]
            )
            logger.info("Search Index Manager initialized")
        
        # Initialize LLM Manager
        if all([config["azure_openai_endpoint"], os.getenv("AZURE_OPENAI_KEY")]):
            llm_manager = LLMManager(
                azure_openai_endpoint=config["azure_openai_endpoint"],
                azure_openai_key=os.getenv("AZURE_OPENAI_KEY"),
                azure_openai_deployment=config["azure_openai_chat_deployment"],
                default_temperature=0.7,
                default_max_tokens=500
            )
            logger.info("LLM Manager initialized")
        
        # Initialize RAG Manager
        if search_manager and llm_manager:
            rag_manager = RAGManager(
                search_index_manager=search_manager,
                llm_manager=llm_manager,
                default_top_k=3,
                max_context_tokens=6000,
                relevance_threshold=0.02
            )
            logger.info("RAG Manager initialized")
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    await initialize_components()

# Health Check Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check for all system components."""
    components = {
        "storage_manager": storage_manager is not None,
        "content_manager": content_manager is not None,
        "search_manager": search_manager is not None,
        "llm_manager": llm_manager is not None,
        "rag_manager": rag_manager is not None
    }
    
    # Test connectivity
    connectivity = {}
    try:
        if storage_manager:
            connectivity["azure_storage"] = storage_manager.verify_connection()
        if llm_manager:
            validation = llm_manager.validate_configuration()
            connectivity["azure_openai"] = validation["valid"]
        if rag_manager:
            validation = rag_manager.validate_configuration()
            connectivity["rag_system"] = validation["valid"]
    except Exception as e:
        logger.error(f"Health check error: {e}")
        connectivity["error"] = str(e)
    
    status = "healthy" if all(connectivity.values()) else "unhealthy"
    
    return HealthResponse(
        status=status,
        timestamp=datetime.now().isoformat(),
        components={**components, **connectivity},
        configuration=get_configuration()
    )

@app.get("/health/simple")
async def simple_health():
    """Simple health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

# Training Pipeline Endpoints
@app.post("/training/start")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start the training pipeline in the background."""
    global training_status
    
    if training_status["status"] == "running":
        raise HTTPException(status_code=409, detail="Training pipeline is already running")
    
    if not all([storage_manager, content_manager, search_manager]):
        raise HTTPException(status_code=500, detail="Required components not initialized")
    
    # Reset training status
    training_status.update({
        "status": "running",
        "message": "Training pipeline started",
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "progress": {"step": 1, "total_steps": 10, "current_task": "Initializing"},
        "results": {}
    })
    
    # Start training in background
    background_tasks.add_task(run_training_pipeline, request)
    
    return {"message": "Training pipeline started", "status": "running"}

@app.get("/training/status", response_model=TrainingStatusResponse)
async def get_training_status():
    """Get current training pipeline status."""
    return TrainingStatusResponse(**training_status)

@app.post("/training/stop")
async def stop_training():
    """Stop the training pipeline (if running)."""
    global training_status
    
    if training_status["status"] != "running":
        raise HTTPException(status_code=400, detail="No training pipeline is currently running")
    
    training_status.update({
        "status": "stopped",
        "message": "Training pipeline stopped by user",
        "end_time": datetime.now().isoformat()
    })
    
    return {"message": "Training pipeline stopped", "status": "stopped"}

async def run_training_pipeline(request: TrainingRequest):
    """Execute the complete training pipeline."""
    global training_status
    
    try:
        logger.info("Starting training pipeline execution")
        
        # Step 1: Verify connection
        training_status["progress"].update({
            "step": 1,
            "current_task": "Verifying Azure Storage connection"
        })
        
        if not storage_manager.verify_connection():
            raise Exception("Failed to connect to Azure Blob Storage")
        
        # Step 2: Get container statistics
        training_status["progress"].update({
            "step": 2,
            "current_task": "Analyzing container contents"
        })
        
        stats = storage_manager.get_container_statistics()
        training_status["results"]["container_stats"] = stats
        
        # Step 3: Get supported files
        training_status["progress"].update({
            "step": 3,
            "current_task": "Identifying supported files"
        })
        
        supported_files = content_manager.get_source_files_from_container()
        training_status["results"]["supported_files_count"] = len(supported_files)
        
        # Step 4: Extract content
        training_status["progress"].update({
            "step": 4,
            "current_task": f"Processing {len(supported_files)} files"
        })
        
        start_time = time.time()
        results = content_manager.extract_content_from_files(
            supported_files, 
            output_folder="md_files", 
            extract_images=request.extract_images
        )
        extraction_time = time.time() - start_time
        
        files_processed = sum(1 for r in results.values() if isinstance(r, dict) and r.get('success'))
        training_status["results"]["content_extraction"] = {
            "files_processed": files_processed,
            "total_files": len(supported_files),
            "processing_time": extraction_time,
            "results": results
        }
        
        # Step 5: Initialize search indexing
        training_status["progress"].update({
            "step": 5,
            "current_task": "Initializing search index manager"
        })
        
        if not search_manager:
            raise Exception("Search Index Manager not available")
        
        # Step 6: Process markdown files
        training_status["progress"].update({
            "step": 6,
            "current_task": "Processing markdown files for search index"
        })
        
        index_start_time = time.time()
        index_results = search_manager.process_markdown_files("md_files")
        index_time = time.time() - index_start_time
        
        training_status["results"]["search_indexing"] = {
            **index_results,
            "processing_time": index_time
        }
        
        # Step 7: Test search functionality
        training_status["progress"].update({
            "step": 7,
            "current_task": "Testing search functionality"
        })
        
        if index_results.get('success') and index_results.get('chunks_with_embeddings', 0) > 0:
            test_query = "cyber security policy"
            search_results = search_manager.search_similar_chunks(test_query, top_k=3)
            training_status["results"]["search_test"] = {
                "query": test_query,
                "results_found": len(search_results) if search_results else 0,
                "success": bool(search_results)
            }
        
        # Step 8: Validate RAG system
        training_status["progress"].update({
            "step": 8,
            "current_task": "Validating RAG system"
        })
        
        if rag_manager:
            test_result = rag_manager.test_rag_pipeline("What is our incident management process?")
            training_status["results"]["rag_test"] = test_result
        
        # Step 9: Generate summary
        training_status["progress"].update({
            "step": 9,
            "current_task": "Generating training summary"
        })
        
        total_time = time.time() - start_time
        training_status["results"]["summary"] = {
            "total_processing_time": total_time,
            "files_processed": files_processed,
            "chunks_created": index_results.get('total_chunks', 0),
            "chunks_indexed": index_results.get('chunks_with_embeddings', 0),
            "search_test_passed": training_status["results"].get("search_test", {}).get("success", False),
            "rag_test_passed": training_status["results"].get("rag_test", {}).get("success", False)
        }
        
        # Complete successfully
        training_status.update({
            "status": "completed",
            "message": "Training pipeline completed successfully",
            "end_time": datetime.now().isoformat(),
            "progress": {"step": 10, "current_task": "Completed"}
        })
        
        logger.info("Training pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        training_status.update({
            "status": "failed",
            "message": f"Training pipeline failed: {str(e)}",
            "end_time": datetime.now().isoformat()
        })

# RAG Retrieval Endpoints
@app.post("/rag/query", response_model=RAGResponse)
async def rag_query(request: RAGRequest):
    """Generate a RAG response for the given query (non-streaming)."""
    if not rag_manager:
        raise HTTPException(status_code=500, detail="RAG Manager not initialized")
    
    try:
        response = rag_manager.generate_rag_response(
            query=request.query,
            top_k=request.top_k,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            custom_system_prompt=request.custom_system_prompt,
            include_search_details=request.include_search_details
        )
        
        return RAGResponse(**response)
        
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

@app.post("/rag/stream")
async def rag_stream(request: RAGRequest):
    """Generate a streaming RAG response for the given query."""
    if not rag_manager:
        raise HTTPException(status_code=500, detail="RAG Manager not initialized")
    
    async def generate_stream():
        """Generate streaming response."""
        try:
            for chunk in rag_manager.generate_streaming_rag_response(
                query=request.query,
                top_k=request.top_k,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                custom_system_prompt=request.custom_system_prompt
            ):
                # Format chunk as Server-Sent Events
                chunk_data = json.dumps(chunk)
                yield f"data: {chunk_data}\n\n"
                
                # Add small delay to prevent overwhelming the client
                await asyncio.sleep(0.01)
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'stream_complete'})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming RAG error: {e}")
            error_chunk = {
                "type": "error",
                "error": str(e)
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

# Document Search Endpoints
@app.post("/search/documents", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search for documents using the given query."""
    if not rag_manager:
        raise HTTPException(status_code=500, detail="RAG Manager not initialized")
    
    try:
        result = rag_manager.search_documents(
            query=request.query,
            top_k=request.top_k
        )
        
        # Transform the result to match SearchResponse model
        if result.get("success", False):
            search_response = SearchResponse(
                success=result["success"],
                results=result.get("documents", []),  # Map 'documents' to 'results'
                metadata=result.get("metadata", {}),
                error=None
            )
        else:
            search_response = SearchResponse(
                success=False,
                results=[],
                metadata=result.get("metadata", {}),
                error=result.get("error", "Unknown error")
            )
        
        return search_response
        
    except Exception as e:
        logger.error(f"Document search error: {e}")
        raise HTTPException(status_code=500, detail=f"Document search failed: {str(e)}")

@app.get("/search/test")
async def test_search():
    """Test search functionality with a predefined query."""
    if not search_manager:
        raise HTTPException(status_code=500, detail="Search Manager not initialized")
    
    try:
        test_query = "cyber security policy"
        results = search_manager.search_similar_chunks(test_query, top_k=3)
        
        return {
            "success": True,
            "query": test_query,
            "results_count": len(results) if results else 0,
            "results": results[:3] if results else []
        }
        
    except Exception as e:
        logger.error(f"Search test error: {e}")
        raise HTTPException(status_code=500, detail=f"Search test failed: {str(e)}")

# Configuration Endpoints
@app.get("/config")
async def get_config():
    """Get current system configuration."""
    config = get_configuration()
    
    # Add component status
    config["components"] = {
        "storage_manager": storage_manager is not None,
        "content_manager": content_manager is not None,
        "search_manager": search_manager is not None,
        "llm_manager": llm_manager is not None,
        "rag_manager": rag_manager is not None
    }
    
    if rag_manager:
        rag_config = rag_manager.get_configuration_info()
        config["rag_settings"] = rag_config.get("rag_settings", {})
    
    return config

@app.post("/config/reload")
async def reload_configuration():
    """Reload system configuration and reinitialize components."""
    try:
        await initialize_components()
        return {"message": "Configuration reloaded successfully", "status": "success"}
    except Exception as e:
        logger.error(f"Configuration reload error: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration reload failed: {str(e)}")

# Utility Endpoints
@app.get("/storage/stats")
async def get_storage_stats():
    """Get Azure Storage container statistics."""
    if not storage_manager:
        raise HTTPException(status_code=500, detail="Storage Manager not initialized")
    
    try:
        stats = storage_manager.get_container_statistics()
        return stats
    except Exception as e:
        logger.error(f"Storage stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get storage stats: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
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

if __name__ == "__main__":
    # Development server configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    print(f"🚀 Starting HEALRAG API server...")
    print(f"📍 Host: {host}:{port}")
    print(f"📚 Docs: http://{host}:{port}/docs")
    print(f"🔄 Reload: {reload}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        access_log=True
    ) 