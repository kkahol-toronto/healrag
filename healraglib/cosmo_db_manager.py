"""
CosmoDB Manager for HEALRAG
===========================

Manages Azure Cosmos DB operations for storing RAG request/response data,
session management, and chat history.
"""

import os
import logging
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from azure.cosmos import CosmosClient, PartitionKey, exceptions
from azure.cosmos.container import ContainerProxy
from azure.cosmos.database import DatabaseProxy

logger = logging.getLogger(__name__)

class CosmoDBManager:
    """
    Manages Azure Cosmos DB operations for HEALRAG chat storage.
    
    Features:
    - Store RAG request/response pairs
    - Session-based chat history
    - User information tracking
    - Partition by sessionID for efficient queries
    """
    
    def __init__(
        self,
        connection_string: Optional[str] = None,
        database_name: Optional[str] = None,
        container_name: Optional[str] = None
    ):
        """
        Initialize CosmoDB Manager.
        
        Args:
            connection_string: Azure Cosmos DB connection string
            database_name: Name of the Cosmos DB database
            container_name: Name of the container for storing chats
        """
        self.connection_string = connection_string or os.getenv("AZURE_COSMO_CONNECTION_STRING")
        self.database_name = database_name or os.getenv("AZURE_COSMO_DB_NAME")
        self.container_name = container_name or os.getenv("AZURE_COSMO_DB_CONTAINER", "chats")
        
        if not self.connection_string:
            raise ValueError("Azure Cosmos DB connection string is required")
        if not self.database_name:
            raise ValueError("Azure Cosmos DB database name is required")
        
        self.client = None
        self.database = None
        self.container = None
        
        # Initialize connection
        self._initialize_connection()
    
    def _initialize_connection(self):
        """Initialize connection to Cosmos DB and ensure container exists."""
        try:
            # Create Cosmos client
            self.client = CosmosClient.from_connection_string(self.connection_string)
            logger.info("✅ Connected to Azure Cosmos DB")
            
            # Get or create database
            self.database = self.client.create_database_if_not_exists(id=self.database_name)
            logger.info(f"✅ Database '{self.database_name}' ready")
            
            # Get or create container with sessionID partition key
            self.container = self.database.create_container_if_not_exists(
                id=self.container_name,
                partition_key=PartitionKey(path="/sessionID"),
                offer_throughput=400  # Minimum throughput
            )
            logger.info(f"✅ Container '{self.container_name}' ready with /sessionID partition")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Cosmos DB connection: {e}")
            raise
    
    def verify_connection(self) -> bool:
        """
        Verify connection to Cosmos DB.
        
        Returns:
            bool: True if connection is successful
        """
        try:
            if not self.container:
                return False
            
            # Test connection by reading container properties
            properties = self.container.read()
            logger.info(f"✅ Cosmos DB connection verified - Container: {properties['id']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Cosmos DB connection verification failed: {e}")
            return False
    
    def store_rag_interaction(
        self,
        session_id: str,
        query: str,
        response: str,
        user_info: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Store a RAG request/response interaction.
        
        Args:
            session_id: Unique session identifier (partition key)
            query: User's query/question
            response: AI-generated response
            user_info: Optional user information
            metadata: Optional metadata (response time, tokens, etc.)
            sources: Optional source documents used
            
        Returns:
            Dict containing the stored document
        """
        try:
            # Create document
            document = {
                "id": f"{session_id}_{datetime.utcnow().isoformat()}_{hash(query) % 10000}",
                "sessionID": session_id,  # Partition key
                "type": "rag_interaction",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "query": query,
                "response": response,
                "user_info": user_info or {},
                "metadata": metadata or {},
                "sources": sources or [],
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            # Store in Cosmos DB
            created_item = self.container.create_item(body=document)
            
            logger.info(f"✅ Stored RAG interaction for session {session_id}")
            return created_item
            
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"❌ Failed to store RAG interaction: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error storing RAG interaction: {e}")
            raise
    
    def get_session_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
        include_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve chat history for a specific session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of interactions to return
            include_metadata: Whether to include metadata in response
            
        Returns:
            List of chat interactions ordered by timestamp
        """
        try:
            # Query for session interactions
            query = "SELECT * FROM c WHERE c.sessionID = @session_id AND c.type = 'rag_interaction' ORDER BY c.timestamp ASC"
            parameters = [{"name": "@session_id", "value": session_id}]
            
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                partition_key=session_id
            ))
            
            # Apply limit if specified
            if limit:
                items = items[-limit:]  # Get the most recent items
            
            # Remove metadata if not requested
            if not include_metadata:
                for item in items:
                    item.pop('metadata', None)
                    item.pop('_rid', None)
                    item.pop('_self', None)
                    item.pop('_etag', None)
                    item.pop('_attachments', None)
                    item.pop('_ts', None)
            
            logger.info(f"✅ Retrieved {len(items)} interactions for session {session_id}")
            return items
            
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"❌ Failed to retrieve session history: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Unexpected error retrieving session history: {e}")
            return []
    
    def get_user_sessions(
        self,
        user_identifier: str,
        limit: Optional[int] = 50
    ) -> List[Dict[str, Any]]:
        """
        Get all sessions for a specific user.
        
        Args:
            user_identifier: User identifier (email, user_id, etc.)
            limit: Maximum number of sessions to return
            
        Returns:
            List of session summaries
        """
        try:
            # Query for user sessions
            query = """
            SELECT DISTINCT c.sessionID, 
                   MIN(c.timestamp) as first_interaction,
                   MAX(c.timestamp) as last_interaction,
                   COUNT(1) as interaction_count
            FROM c 
            WHERE c.user_info.email = @user_id OR c.user_info.user_id = @user_id
            GROUP BY c.sessionID
            ORDER BY MAX(c.timestamp) DESC
            """
            parameters = [{"name": "@user_id", "value": user_identifier}]
            
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                enable_cross_partition_query=True
            ))
            
            # Apply limit
            if limit:
                items = items[:limit]
            
            logger.info(f"✅ Retrieved {len(items)} sessions for user {user_identifier}")
            return items
            
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"❌ Failed to retrieve user sessions: {e}")
            return []
        except Exception as e:
            logger.error(f"❌ Unexpected error retrieving user sessions: {e}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete all interactions for a specific session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            bool: True if deletion was successful
        """
        try:
            # Get all items in the session
            query = "SELECT c.id FROM c WHERE c.sessionID = @session_id"
            parameters = [{"name": "@session_id", "value": session_id}]
            
            items = list(self.container.query_items(
                query=query,
                parameters=parameters,
                partition_key=session_id
            ))
            
            # Delete each item
            deleted_count = 0
            for item in items:
                self.container.delete_item(item=item['id'], partition_key=session_id)
                deleted_count += 1
            
            logger.info(f"✅ Deleted {deleted_count} interactions for session {session_id}")
            return True
            
        except exceptions.CosmosHttpResponseError as e:
            logger.error(f"❌ Failed to delete session: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error deleting session: {e}")
            return False
    
    def get_container_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the chat container.
        
        Returns:
            Dict containing container statistics
        """
        try:
            # Get total document count
            query = "SELECT VALUE COUNT(1) FROM c"
            total_count = list(self.container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))[0]
            
            # Get unique sessions count
            query = "SELECT VALUE COUNT(DISTINCT c.sessionID) FROM c"
            session_count = list(self.container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))[0]
            
            # Get unique users count
            query = "SELECT VALUE COUNT(DISTINCT c.user_info.email) FROM c WHERE IS_DEFINED(c.user_info.email)"
            user_count = list(self.container.query_items(
                query=query,
                enable_cross_partition_query=True
            ))[0]
            
            stats = {
                "total_interactions": total_count,
                "unique_sessions": session_count,
                "unique_users": user_count,
                "container_name": self.container_name,
                "database_name": self.database_name,
                "partition_key": "/sessionID"
            }
            
            logger.info(f"✅ Retrieved container stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Failed to retrieve container stats: {e}")
            return {
                "error": str(e),
                "container_name": self.container_name,
                "database_name": self.database_name
            }
    
    def search_interactions(
        self,
        search_query: str,
        session_id: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search for interactions containing specific text.
        
        Args:
            search_query: Text to search for in queries or responses
            session_id: Optional session to limit search to
            limit: Maximum results to return
            
        Returns:
            List of matching interactions
        """
        try:
            # Build query
            if session_id:
                query = """
                SELECT * FROM c 
                WHERE c.sessionID = @session_id 
                AND (CONTAINS(UPPER(c.query), UPPER(@search)) OR CONTAINS(UPPER(c.response), UPPER(@search)))
                ORDER BY c.timestamp DESC
                """
                parameters = [
                    {"name": "@session_id", "value": session_id},
                    {"name": "@search", "value": search_query}
                ]
                
                items = list(self.container.query_items(
                    query=query,
                    parameters=parameters,
                    partition_key=session_id
                ))
            else:
                query = """
                SELECT * FROM c 
                WHERE (CONTAINS(UPPER(c.query), UPPER(@search)) OR CONTAINS(UPPER(c.response), UPPER(@search)))
                ORDER BY c.timestamp DESC
                """
                parameters = [{"name": "@search", "value": search_query}]
                
                items = list(self.container.query_items(
                    query=query,
                    parameters=parameters,
                    enable_cross_partition_query=True
                ))
            
            # Apply limit
            items = items[:limit]
            
            logger.info(f"✅ Found {len(items)} interactions matching '{search_query}'")
            return items
            
        except Exception as e:
            logger.error(f"❌ Failed to search interactions: {e}")
            return [] 