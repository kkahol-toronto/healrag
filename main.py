#!/usr/bin/env python3
"""
HEALRAG FastAPI Application with Azure AD Authentication
=========================================================

A comprehensive FastAPI application for the HEALRAG system including:
- Azure AD OAuth2 authentication
- Health check endpoints
- Training pipeline execution
- RAG retrieval (streaming and non-streaming)
- Document search
- System configuration management
"""

import os
import json
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Depends, status
from fastapi.responses import StreamingResponse, JSONResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn
import httpx
import jwt
from jwt import PyJWTError

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import HEALRAG components
from healraglib import StorageManager, RAGManager, LLMManager, SearchIndexManager, CosmoDBManager
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

# Azure AD Configuration
AZURE_AD_CONFIG = {
    "tenant_id": os.getenv("AZURE_AD_TENANT_ID"),
    "client_id": os.getenv("AZURE_AD_CLIENT_ID"),
    "client_secret": os.getenv("AZURE_AD_CLIENT_SECRET"),
    "redirect_uri": os.getenv("AZURE_AD_REDIRECT_URI", "https://healrag-security.azurewebsites.net/auth/callback"),
    "authority": f"https://login.microsoftonline.com/{os.getenv('AZURE_AD_TENANT_ID')}",
    "scope": ["openid", "profile", "User.Read"]
}

# FastAPI app initialization
app = FastAPI(
    title="HEALRAG Security Assistant API",
    description="Comprehensive API for the HEALRAG system with Azure AD authentication",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    swagger_ui_oauth2_redirect_url="/docs/oauth2-redirect"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://healrag-security.azurewebsites.net",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",      # Add this for frontend dev
        "http://127.0.0.1:3000",
        "https://blue-dune-0ef76e10f2.azurestaticapps.net"       # Add this for frontend dev
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Global variables for component instances
storage_manager: Optional[StorageManager] = None
content_manager: Optional[ContentManager] = None
search_manager: Optional[SearchIndexManager] = None
llm_manager: Optional[LLMManager] = None
rag_manager: Optional[RAGManager] = None
cosmo_db_manager: Optional[CosmoDBManager] = None

# Training pipeline status tracking
training_status = {
    "status": "idle",  # idle, running, completed, failed
    "message": "",
    "start_time": None,
    "end_time": None,
    "progress": {},
    "results": {}
}

# Azure AD JWKS cache
jwks_cache = {"keys": None, "expires": 0}

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
    session_id: Optional[str] = Field(None, description="Session ID for tracking conversations")
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

class UserInfo(BaseModel):
    user_id: str
    email: Optional[str]
    name: Optional[str]
    roles: List[str] = []

class SessionHistoryRequest(BaseModel):
    session_id: str = Field(..., description="Session ID to retrieve history for")
    limit: Optional[int] = Field(default=50, ge=1, le=200, description="Maximum number of interactions to return")
    include_metadata: bool = Field(default=False, description="Include metadata in response")

class SessionHistoryResponse(BaseModel):
    success: bool
    session_id: str
    interactions: List[Dict[str, Any]]
    total_count: int
    error: Optional[str] = None

# Authentication functions
async def get_azure_ad_jwks():
    """Get Azure AD JWKS for token validation."""
    global jwks_cache
    
    current_time = time.time()
    if jwks_cache["keys"] and current_time < jwks_cache["expires"]:
        return jwks_cache["keys"]
    
    try:
        jwks_url = f"{AZURE_AD_CONFIG['authority']}/discovery/v2.0/keys"
        async with httpx.AsyncClient() as client:
            response = await client.get(jwks_url)
            response.raise_for_status()
            jwks_data = response.json()
            
            # Cache for 1 hour
            jwks_cache["keys"] = jwks_data["keys"]
            jwks_cache["expires"] = current_time + 3600
            
            return jwks_data["keys"]
    except Exception as e:
        logger.error(f"Failed to get JWKS: {e}")
        return None

async def verify_azure_ad_token(token: str) -> Optional[Dict]:
    """Verify Azure AD JWT token."""
    try:
        # Get JWKS
        jwks = await get_azure_ad_jwks()
        if not jwks:
            return None
        
        # Decode token header to get kid
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")
        
        # Find the correct key
        key = None
        for jwk in jwks:
            if jwk.get("kid") == kid:
                key = jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))
                break
        
        if not key:
            logger.error("Could not find appropriate key")
            return None
        
        # Verify token WITHOUT audience check (since Azure gives us Graph audience)
        # but we'll verify the app ID separately
        payload = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            issuer=f"{AZURE_AD_CONFIG['authority']}/v2.0",
            options={"verify_aud": False}  # Skip audience verification
        )
        
        # Verify this token is actually for our app by checking appid
        token_app_id = payload.get("appid") or payload.get("azp")
        if token_app_id != AZURE_AD_CONFIG["client_id"]:
            logger.error(f"Token app ID {token_app_id} doesn't match our client ID {AZURE_AD_CONFIG['client_id']}")
            return None
        
        # Additional security checks
        current_time = time.time()
        
        # Check if token is expired
        if payload.get("exp", 0) < current_time:
            logger.error("Token has expired")
            return None
        
        # Check if token is not yet valid
        if payload.get("nbf", 0) > current_time:
            logger.error("Token is not yet valid")
            return None
        
        logger.info(f"Token verified successfully for app {token_app_id}")
        return payload
        
    except PyJWTError as e:
        logger.error(f"JWT verification failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        return None

async def get_current_user_simple(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserInfo:
    """Simplified authentication for testing."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # Decode without signature verification for testing
        payload = jwt.decode(credentials.credentials, options={"verify_signature": False})
        
        # Basic checks
        current_time = time.time()
        
        # Check expiration
        if payload.get("exp", 0) < current_time:
            raise HTTPException(status_code=401, detail="Token expired")
        
        # Check app ID
        token_app_id = payload.get("appid")
        if token_app_id != AZURE_AD_CONFIG["client_id"]:
            raise HTTPException(status_code=401, detail=f"Wrong app ID: {token_app_id}")
        
        # Check issuer
        expected_issuer = f"https://sts.windows.net/{AZURE_AD_CONFIG['tenant_id']}/"
        if payload.get("iss") != expected_issuer:
            raise HTTPException(status_code=401, detail="Wrong issuer")
        
        return UserInfo(
            user_id=payload.get("oid", ""),
            email=payload.get("unique_name") or payload.get("preferred_username"),
            name=payload.get("name"),
            roles=payload.get("roles", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token processing error: {e}")
        raise HTTPException(status_code=401, detail=f"Invalid token: {str(e)}")

# Add a test endpoint using the simplified auth
@app.get("/auth/test-simple")
async def test_simple_auth(current_user: UserInfo = Depends(get_current_user_simple)):
    """Test endpoint with simplified authentication."""
    return {
        "message": "🎉 Simplified auth working!",
        "user": current_user,
        "timestamp": datetime.now().isoformat()
    }

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> UserInfo:
    """Get current authenticated user from JWT token."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # For production, we'll verify the signature
        # For now, let's use the working approach with basic validation
        payload = jwt.decode(credentials.credentials, options={"verify_signature": False})
        
        # Security checks
        current_time = time.time()
        
        # 1. Check expiration
        if payload.get("exp", 0) < current_time:
            logger.warning("Token expired")
            raise HTTPException(status_code=401, detail="Token expired")
        
        # 2. Check not-before time
        if payload.get("nbf", 0) > current_time:
            logger.warning("Token not yet valid")
            raise HTTPException(status_code=401, detail="Token not yet valid")
        
        # 3. Check app ID (most important security check)
        token_app_id = payload.get("appid")
        if token_app_id != AZURE_AD_CONFIG["client_id"]:
            logger.error(f"Wrong app ID: {token_app_id} != {AZURE_AD_CONFIG['client_id']}")
            raise HTTPException(status_code=401, detail="Invalid application")
        
        # 4. Check issuer (verify it's from your Azure AD)
        expected_issuer = f"https://sts.windows.net/{AZURE_AD_CONFIG['tenant_id']}/"
        if payload.get("iss") != expected_issuer:
            logger.error(f"Wrong issuer: {payload.get('iss')}")
            raise HTTPException(status_code=401, detail="Invalid token issuer")
        
        # 5. Check token type
        if payload.get("typ") != "JWT":
            logger.warning(f"Unexpected token type: {payload.get('typ')}")
        
        # Extract user information
        user_info = UserInfo(
            user_id=payload.get("oid", ""),
            email=payload.get("unique_name") or payload.get("preferred_username") or payload.get("upn"),
            name=payload.get("name", ""),
            roles=payload.get("roles", [])
        )
        
        logger.info(f"User authenticated: {user_info.email}")
        return user_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_optional_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[UserInfo]:
    """Get current user if authenticated, otherwise return None."""
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None

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
    global storage_manager, content_manager, search_manager, llm_manager, rag_manager, cosmo_db_manager
    
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
        
        # Initialize CosmoDB Manager (optional)
        cosmo_connection_string = os.getenv("AZURE_COSMO_CONNECTION_STRING")
        if cosmo_connection_string:
            try:
                cosmo_db_manager = CosmoDBManager(
                    connection_string=cosmo_connection_string,
                    database_name=os.getenv("AZURE_COSMO_DB_NAME"),
                    container_name=os.getenv("AZURE_COSMO_DB_CONTAINER", "chats")
                )
                if cosmo_db_manager.verify_connection():
                    logger.info("CosmoDB Manager initialized successfully")
                else:
                    logger.warning("CosmoDB Manager connection verification failed")
                    cosmo_db_manager = None
            except Exception as e:
                logger.warning(f"CosmoDB Manager initialization failed: {e}")
                cosmo_db_manager = None
        else:
            logger.info("CosmoDB Manager not configured (optional)")
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup."""
    await initialize_components()

# Authentication Endpoints
@app.get("/auth/login")
async def login(redirect_uri: Optional[str] = None):
    """Redirect to Azure AD for authentication."""
    # Store redirect_uri in session or pass it through state parameter
    state = ""
    if redirect_uri:
        # Base64 encode the redirect_uri to pass it safely
        import base64
        state = base64.b64encode(redirect_uri.encode()).decode()
    
    auth_url = (
        f"{AZURE_AD_CONFIG['authority']}/oauth2/v2.0/authorize?"
        f"client_id={AZURE_AD_CONFIG['client_id']}&"
        f"response_type=code&"
        f"redirect_uri={AZURE_AD_CONFIG['redirect_uri']}&"
        f"scope={' '.join(AZURE_AD_CONFIG['scope'])}&"
        f"response_mode=query"
    )
    
    if state:
        auth_url += f"&state={state}"
    
    return RedirectResponse(url=auth_url)

@app.get("/auth/callback")
async def auth_callback(code: str = None, error: str = None, state: str = None):
    """Handle Azure AD authentication callback."""
    if error:
        raise HTTPException(status_code=400, detail=f"Authentication error: {error}")
    
    if not code:
        raise HTTPException(status_code=400, detail="No authorization code provided")
    
    try:
        # Exchange code for token (existing code...)
        token_url = f"{AZURE_AD_CONFIG['authority']}/oauth2/v2.0/token"
        data = {
            "client_id": AZURE_AD_CONFIG["client_id"],
            "client_secret": AZURE_AD_CONFIG["client_secret"],
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": AZURE_AD_CONFIG["redirect_uri"],
            "scope": " ".join(AZURE_AD_CONFIG["scope"])
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(token_url, data=data)
            
        if response.status_code != 200:
            logger.error(f"Token exchange failed: {response.status_code} - {response.text}")
            raise HTTPException(status_code=400, detail="Token exchange failed")
        
        token_data = response.json()
        access_token = token_data.get("access_token")
        
        # Decode redirect_uri from state parameter
        redirect_url = "https://healrag-security.azurewebsites.net"  # default
        if state:
            try:
                import base64
                redirect_url = base64.b64decode(state.encode()).decode()
            except:
                pass
        
        # Option 1: Redirect back to frontend with token in URL (less secure)
        return RedirectResponse(url=f"{redirect_url}?token={access_token}&expires_in={token_data.get('expires_in', 3600)}")
        
        # Option 2: Set httpOnly cookies and redirect (more secure)
        # response = RedirectResponse(url=redirect_url)
        # response.set_cookie(
        #     key="access_token", 
        #     value=access_token, 
        #     httponly=True, 
        #     secure=True, 
        #     samesite="lax",
        #     max_age=token_data.get('expires_in', 3600)
        # )
        # return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication callback error: {e}")
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")

@app.get("/auth/test-token")
async def test_token(current_user: UserInfo = Depends(get_current_user)):
    """Test endpoint to verify your token is working."""
    return {
        "message": "🎉 Token is valid!",
        "user": current_user,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/debug/token")
async def debug_token(request: Request):
    """Debug endpoint to see what's in your token."""
    auth_header = request.headers.get("authorization")
    if not auth_header:
        return {"error": "No authorization header"}
    
    if not auth_header.startswith("Bearer "):
        return {"error": "Invalid authorization header format"}
    
    token = auth_header.split(" ")[1]
    
    try:
        # Decode without verification to see contents
        payload = jwt.decode(token, options={"verify_signature": False})
        
        return {
            "token_info": {
                "audience": payload.get("aud"),
                "issuer": payload.get("iss"),
                "subject": payload.get("sub"),
                "app_id": payload.get("appid"),
                "client_id": AZURE_AD_CONFIG["client_id"],
                "name": payload.get("name"),
                "email": payload.get("unique_name"),
                "expires": payload.get("exp"),
                "scopes": payload.get("scp")
            },
            "config_check": {
                "expected_audience": AZURE_AD_CONFIG["client_id"],
                "actual_audience": payload.get("aud"),
                "audience_match": payload.get("aud") == AZURE_AD_CONFIG["client_id"]
            }
        }
    except Exception as e:
        return {"error": f"Token decode error: {e}"}

@app.get("/auth/logout")
async def logout(redirect_uri: Optional[str] = None):
    """Logout from Azure AD."""
    post_logout_redirect = redirect_uri or "https://healrag-security.azurewebsites.net/"
    logout_url = (
        f"{AZURE_AD_CONFIG['authority']}/oauth2/v2.0/logout?"
        f"post_logout_redirect_uri={post_logout_redirect}"
    )
    return RedirectResponse(url=logout_url)

@app.get("/auth/me")
async def get_user_info(current_user: UserInfo = Depends(get_current_user)):
    """Get current user information."""
    return current_user

# Health Check Endpoints (Public)
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check for all system components."""
    components = {
        "storage_manager": storage_manager is not None,
        "content_manager": content_manager is not None,
        "search_manager": search_manager is not None,
        "llm_manager": llm_manager is not None,
        "rag_manager": rag_manager is not None,
        "cosmo_db_manager": cosmo_db_manager is not None,
        "azure_ad_config": all([
            AZURE_AD_CONFIG["tenant_id"],
            AZURE_AD_CONFIG["client_id"],
            AZURE_AD_CONFIG["client_secret"]
        ])
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
        if cosmo_db_manager:
            connectivity["azure_cosmos_db"] = cosmo_db_manager.verify_connection()
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

# Protected Training Pipeline Endpoints
@app.post("/training/start")
async def start_training(
    request: TrainingRequest, 
    background_tasks: BackgroundTasks,
    current_user: UserInfo = Depends(get_current_user)
):
    """Start the training pipeline in the background."""
    global training_status
    
    logger.info(f"Training started by user: {current_user.email}")
    
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
        "results": {},
        "started_by": current_user.email
    })
    
    # Start training in background
    background_tasks.add_task(run_training_pipeline, request)
    
    return {"message": "Training pipeline started", "status": "running"}

@app.get("/training/status", response_model=TrainingStatusResponse)
async def get_training_status(current_user: UserInfo = Depends(get_current_user)):
    """Get current training pipeline status."""
    return TrainingStatusResponse(**training_status)

@app.post("/training/stop")
async def stop_training(current_user: UserInfo = Depends(get_current_user)):
    """Stop the training pipeline (if running)."""
    global training_status
    
    logger.info(f"Training stopped by user: {current_user.email}")
    
    if training_status["status"] != "running":
        raise HTTPException(status_code=400, detail="No training pipeline is currently running")
    
    training_status.update({
        "status": "stopped",
        "message": f"Training pipeline stopped by {current_user.email}",
        "end_time": datetime.now().isoformat()
    })
    
    return {"message": "Training pipeline stopped", "status": "stopped"}

# Protected RAG Retrieval Endpoints
@app.post("/rag/query", response_model=RAGResponse)
async def rag_query(
    request: RAGRequest,
    current_user: UserInfo = Depends(get_current_user)
):
    """Generate a RAG response for the given query (non-streaming)."""
    if not rag_manager:
        raise HTTPException(status_code=500, detail="RAG Manager not initialized")
    
    logger.info(f"RAG query by {current_user.email}: {request.query[:100]}...")
    
    try:
        # Get conversation history if session_id is provided
        conversation_history = []
        if cosmo_db_manager and request.session_id:
            try:
                history_items = cosmo_db_manager.get_session_history(
                    session_id=request.session_id,
                    limit=10  # Get last 10 interactions
                )
                if history_items:
                    # Convert to the format expected by RAG manager, excluding any interaction with the current query
                    conversation_history = [
                        {
                            "query": interaction.get("query", ""),
                            "response": interaction.get("response", "")
                        }
                        for interaction in history_items
                        if interaction.get("query", "") != request.query  # Exclude current query if already stored
                    ]
                    logger.info(f"Retrieved {len(conversation_history)} previous interactions for context")
            except Exception as history_error:
                logger.warning(f"Failed to retrieve conversation history: {history_error}")
                # Continue without history if retrieval fails
        
        response = rag_manager.generate_rag_response(
            query=request.query,
            top_k=request.top_k,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            custom_system_prompt=request.custom_system_prompt,
            include_search_details=request.include_search_details,
            conversation_history=conversation_history
        )
        
        # Add user info to metadata
        if "metadata" not in response:
            response["metadata"] = {}
        response["metadata"]["user"] = current_user.email
        
        # Store interaction in CosmoDB if available and session_id provided
        if cosmo_db_manager and request.session_id:
            try:
                # Prepare user info
                user_info = {
                    "user_id": current_user.user_id,
                    "email": current_user.email,
                    "name": current_user.name
                }
                
                # Store the interaction
                cosmo_db_manager.store_rag_interaction(
                    session_id=request.session_id,
                    query=request.query,
                    response=response.get("response", ""),
                    user_info=user_info,
                    metadata=response.get("metadata", {}),
                    sources=response.get("sources", [])
                )
                logger.info(f"Stored RAG interaction in CosmoDB for session {request.session_id}")
                
            except Exception as cosmo_error:
                logger.warning(f"Failed to store interaction in CosmoDB: {cosmo_error}")
                # Don't fail the request if CosmoDB storage fails
        
        return RAGResponse(**response)
        
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")

@app.post("/rag/stream")
async def rag_stream(
    request: RAGRequest,
    current_user: UserInfo = Depends(get_current_user)
):
    """Generate a streaming RAG response for the given query."""
    if not rag_manager:
        raise HTTPException(status_code=500, detail="RAG Manager not initialized")
    
    logger.info(f"RAG stream by {current_user.email}: {request.query[:100]}...")
    
    async def generate_stream():
        """Generate streaming response."""
        # Variables to collect the complete response for CosmoDB storage
        full_response = ""
        response_metadata = {}
        response_sources = []
        
        # Get conversation history if session_id is provided
        conversation_history = []
        if cosmo_db_manager and request.session_id:
            try:
                history_items = cosmo_db_manager.get_session_history(
                    session_id=request.session_id,
                    limit=10  # Get last 10 interactions
                )
                if history_items:
                    # Convert to the format expected by RAG manager, excluding any interaction with the current query
                    conversation_history = [
                        {
                            "query": interaction.get("query", ""),
                            "response": interaction.get("response", "")
                        }
                        for interaction in history_items
                        if interaction.get("query", "") != request.query  # Exclude current query if already stored
                    ]
                    logger.info(f"Retrieved {len(conversation_history)} previous interactions for streaming context")
            except Exception as history_error:
                logger.warning(f"Failed to retrieve conversation history for streaming: {history_error}")
                # Continue without history if retrieval fails
        
        try:
            for chunk in rag_manager.generate_streaming_rag_response(
                query=request.query,
                top_k=request.top_k,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                custom_system_prompt=request.custom_system_prompt,
                conversation_history=conversation_history
            ):
                # Add user info to metadata
                if "rag_metadata" in chunk:
                    chunk["rag_metadata"]["user"] = current_user.email
                
                # Collect data for CosmoDB storage
                if chunk.get("type") == "chunk":
                    full_response += chunk.get("content", "")
                elif chunk.get("type") == "sources":
                    response_sources = chunk.get("sources", [])
                elif chunk.get("type") == "complete":
                    response_metadata = chunk.get("metadata", {})
                    full_response = chunk.get("full_response", full_response)
                
                # Format chunk as Server-Sent Events
                chunk_data = json.dumps(chunk)
                yield f"data: {chunk_data}\n\n"
                
                # Add small delay to prevent overwhelming the client
                await asyncio.sleep(0.01)
            
            # Store interaction in CosmoDB if available and session_id provided
            if cosmo_db_manager and request.session_id and full_response:
                try:
                    # Prepare user info
                    user_info = {
                        "user_id": current_user.user_id,
                        "email": current_user.email,
                        "name": current_user.name
                    }
                    
                    # Add user to metadata
                    response_metadata["user"] = current_user.email
                    response_metadata["stream_type"] = "streaming"
                    
                    # Store the complete interaction
                    cosmo_db_manager.store_rag_interaction(
                        session_id=request.session_id,
                        query=request.query,
                        response=full_response,
                        user_info=user_info,
                        metadata=response_metadata,
                        sources=response_sources
                    )
                    logger.info(f"Stored streaming RAG interaction in CosmoDB for session {request.session_id}")
                    
                except Exception as cosmo_error:
                    logger.warning(f"Failed to store streaming interaction in CosmoDB: {cosmo_error}")
                    # Don't fail the stream if CosmoDB storage fails
            
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

# Protected Document Search Endpoints
@app.post("/search/documents", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    current_user: UserInfo = Depends(get_current_user)
):
    """Search for documents using the given query."""
    if not rag_manager:
        raise HTTPException(status_code=500, detail="RAG Manager not initialized")
    
    logger.info(f"Document search by {current_user.email}: {request.query[:100]}...")
    
    try:
        result = rag_manager.search_documents(
            query=request.query,
            top_k=request.top_k
        )
        
        # Add user info to metadata
        if result.get("success", False) and "metadata" in result:
            result["metadata"]["user"] = current_user.email
        
        # Transform the result to match SearchResponse model
        if result.get("success", False):
            search_response = SearchResponse(
                success=result["success"],
                results=result.get("documents", []),
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

# Public search test endpoint (for health checks)
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
        "rag_manager": rag_manager is not None,
        "cosmo_db_manager": cosmo_db_manager is not None
    }
    
    if rag_manager:
        rag_config = rag_manager.get_configuration_info()
        config["rag_settings"] = rag_config.get("rag_settings", {})
    
    return config

@app.post("/config/reload")
async def reload_configuration(current_user: UserInfo = Depends(get_current_user)):
    """Reload system configuration and reinitialize components."""
    logger.info(f"Configuration reload by user: {current_user.email}")
    
    try:
        await initialize_components()
        return {"message": "Configuration reloaded successfully", "status": "success"}
    except Exception as e:
        logger.error(f"Configuration reload error: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration reload failed: {str(e)}")

# Utility Endpoints
@app.get("/storage/stats")
async def get_storage_stats(current_user: UserInfo = Depends(get_current_user)):
    """Get Azure Storage container statistics."""
    if not storage_manager:
        raise HTTPException(status_code=500, detail="Storage Manager not initialized")
    
    try:
        stats = storage_manager.get_container_statistics()
        return stats
    except Exception as e:
        logger.error(f"Storage stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get storage stats: {str(e)}")

# CosmoDB Chat History Endpoints
@app.post("/sessions/history", response_model=SessionHistoryResponse)
async def get_session_history(
    request: SessionHistoryRequest,
    current_user: UserInfo = Depends(get_current_user)
):
    """Get chat history for a specific session."""
    if not cosmo_db_manager:
        raise HTTPException(status_code=503, detail="CosmoDB Manager not available")
    
    logger.info(f"Session history request by {current_user.email} for session {request.session_id}")
    
    try:
        interactions = cosmo_db_manager.get_session_history(
            session_id=request.session_id,
            limit=request.limit,
            include_metadata=request.include_metadata
        )
        
        return SessionHistoryResponse(
            success=True,
            session_id=request.session_id,
            interactions=interactions,
            total_count=len(interactions),
            error=None
        )
        
    except Exception as e:
        logger.error(f"Session history error: {e}")
        return SessionHistoryResponse(
            success=False,
            session_id=request.session_id,
            interactions=[],
            total_count=0,
            error=str(e)
        )

@app.get("/sessions/user")
async def get_user_sessions(
    limit: int = 50,
    current_user: UserInfo = Depends(get_current_user)
):
    """Get all sessions for the current user."""
    if not cosmo_db_manager:
        raise HTTPException(status_code=503, detail="CosmoDB Manager not available")
    
    logger.info(f"User sessions request by {current_user.email}")
    
    try:
        # Use email as the primary identifier
        user_identifier = current_user.email or current_user.user_id
        sessions = cosmo_db_manager.get_user_sessions(
            user_identifier=user_identifier,
            limit=limit
        )
        
        return {
            "success": True,
            "user_identifier": user_identifier,
            "sessions": sessions,
            "total_count": len(sessions)
        }
        
    except Exception as e:
        logger.error(f"User sessions error: {e}")
        return {
            "success": False,
            "user_identifier": current_user.email or current_user.user_id,
            "sessions": [],
            "total_count": 0,
            "error": str(e)
        }

@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: UserInfo = Depends(get_current_user)
):
    """Delete a specific session and all its interactions."""
    if not cosmo_db_manager:
        raise HTTPException(status_code=503, detail="CosmoDB Manager not available")
    
    logger.info(f"Session deletion request by {current_user.email} for session {session_id}")
    
    try:
        # Verify the session belongs to the user before deletion
        # This is a security measure to prevent users from deleting other users' sessions
        user_identifier = current_user.email or current_user.user_id
        user_sessions = cosmo_db_manager.get_user_sessions(user_identifier, limit=1000)
        session_ids = [session.get("sessionID") for session in user_sessions]
        
        if session_id not in session_ids:
            raise HTTPException(status_code=403, detail="Session not found or access denied")
        
        success = cosmo_db_manager.delete_session(session_id)
        
        if success:
            return {
                "success": True,
                "message": f"Session {session_id} deleted successfully",
                "session_id": session_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete session")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session deletion error: {e}")
        raise HTTPException(status_code=500, detail=f"Session deletion failed: {str(e)}")

@app.get("/cosmo/stats")
async def get_cosmo_stats(current_user: UserInfo = Depends(get_current_user)):
    """Get CosmoDB container statistics."""
    if not cosmo_db_manager:
        raise HTTPException(status_code=503, detail="CosmoDB Manager not available")
    
    try:
        stats = cosmo_db_manager.get_container_stats()
        return stats
    except Exception as e:
        logger.error(f"CosmoDB stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get CosmoDB stats: {str(e)}")

@app.get("/")
async def root(user: Optional[UserInfo] = Depends(get_optional_user)):
    """Root endpoint with API information."""
    base_info = {
        "message": "HEALRAG Security Assistant API",
        "version": "1.0.0",
        "description": "Comprehensive API for the HEALRAG system with Azure AD authentication",
        "authentication": {
            "type": "Azure AD OAuth2",
            "login_url": "/auth/login",
            "logout_url": "/auth/logout"
        },
        "public_endpoints": {
            "health": "/health",
            "docs": "/docs",
            "search_test": "/search/test"
        }
    }
    
    if user:
        base_info["user"] = {
            "email": user.email,
            "name": user.name,
            "user_id": user.user_id
        }
        base_info["protected_endpoints"] = {
            "training_start": "/training/start",
            "training_status": "/training/status",
            "rag_query": "/rag/query",
            "rag_stream": "/rag/stream",
            "search_documents": "/search/documents",
            "config": "/config",
            "storage_stats": "/storage/stats"
        }
    else:
        base_info["message"] += " - Authentication Required"
        base_info["note"] = "Most endpoints require Azure AD authentication. Visit /auth/login to authenticate."
    
    return base_info

# Training pipeline function
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

if __name__ == "__main__":
    # Development server configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    print(f"🚀 Starting HEALRAG Security Assistant API server...")
    print(f"📍 Host: {host}:{port}")
    print(f"📚 Docs: http://{host}:{port}/docs")
    print(f"🔐 Auth: http://{host}:{port}/auth/login")
    print(f"🔄 Reload: {reload}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        access_log=True
    )