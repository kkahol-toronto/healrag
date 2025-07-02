"""
RAG Manager for HEALRAG

Combines Search Index Manager and LLM Manager to provide 
Retrieval-Augmented Generation capabilities with both streaming and non-streaming responses.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Iterator, Any, Tuple
from datetime import datetime
import time
from pathlib import Path


class RAGManager:
    """
    RAG Manager for HEALRAG library.
    
    Orchestrates the Retrieval-Augmented Generation process by combining
    document search capabilities with language model generation.
    """
    
    def __init__(self, 
                 search_index_manager,
                 llm_manager,
                 default_top_k: int = 5,
                 max_context_tokens: int = 3000,
                 include_source_info: bool = True,
                 relevance_threshold: float = 0.0):
        """
        Initialize the RAG Manager.
        
        Args:
            search_index_manager: SearchIndexManager instance
            llm_manager: LLMManager instance
            default_top_k: Default number of documents to retrieve
            max_context_tokens: Maximum tokens for context
            include_source_info: Whether to include source information in responses
            relevance_threshold: Minimum relevance score for retrieved documents
        """
        self.search_manager = search_index_manager
        self.llm_manager = llm_manager
        self.default_top_k = default_top_k
        self.max_context_tokens = max_context_tokens
        self.include_source_info = include_source_info
        self.relevance_threshold = relevance_threshold
        self.logger = logging.getLogger(__name__)
        
        # Validate dependencies
        self._validate_dependencies()
    
    def _validate_dependencies(self):
        """Validate that required dependencies are properly configured."""
        issues = []
        
        if not self.search_manager:
            issues.append("SearchIndexManager not provided")
        elif not hasattr(self.search_manager, 'search_similar_chunks'):
            issues.append("SearchIndexManager missing search_similar_chunks method")
        
        if not self.llm_manager:
            issues.append("LLMManager not provided")
        elif not hasattr(self.llm_manager, 'generate_response'):
            issues.append("LLMManager missing generate_response method")
        
        if issues:
            error_msg = f"RAG Manager initialization failed: {', '.join(issues)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info("RAG Manager dependencies validated successfully")
    
    def generate_rag_response(self, 
                             query: str,
                             top_k: Optional[int] = None,
                             custom_system_prompt: Optional[str] = None,
                             temperature: Optional[float] = None,
                             max_tokens: Optional[int] = None,
                             include_search_details: bool = False,
                             conversation_history: Optional[List[Dict[str, str]]] = None,
                             **kwargs) -> Dict[str, Any]:
        """
        Generate a non-streaming RAG response.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve (optional)
            custom_system_prompt: Custom system prompt template (optional)
            temperature: LLM temperature (optional)
            max_tokens: Maximum response tokens (optional)
            include_search_details: Whether to include detailed search info in response
            conversation_history: Previous conversation exchanges for context (optional)
            **kwargs: Additional parameters for LLM generation
            
        Returns:
            Dict containing response, sources, and metadata
        """
        try:
            start_time = time.time()
            
            # Step 1: Retrieve relevant documents
            print(f"🔍 Retrieving relevant documents for query: '{query[:100]}...'")
            retrieval_start = time.time()
            
            top_k = top_k or self.default_top_k
            retrieved_docs = self._retrieve_documents(query, top_k)
            
            retrieval_time = time.time() - retrieval_start
            print(f"   ✅ Retrieved {len(retrieved_docs)} documents in {retrieval_time:.2f}s")
            
            if not retrieved_docs:
                # No relevant documents found
                return self._generate_no_context_response(
                    query, temperature, max_tokens, conversation_history, **kwargs
                )
            
            # Step 2: Build context from retrieved documents
            context, sources = self._build_context(retrieved_docs, query)  # REMOVED force_context=True
            
            # Step 3: Create system message with context
            system_message = self._create_rag_system_message(
                context, custom_system_prompt, conversation_history
            )
            
            # Step 4: Generate LLM response
            print(f"🤖 Generating LLM response...")
            generation_start = time.time()
            
            llm_result = self.llm_manager.generate_response(
                query=query,
                system_message=system_message,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            generation_time = time.time() - generation_start
            total_time = time.time() - start_time
            
            if not llm_result.get("success"):
                # LLM generation failed
                return {
                    "success": False,
                    "response": None,
                    "error": f"LLM generation failed: {llm_result.get('error', 'Unknown error')}",
                    "metadata": {
                        "query": query,
                        "retrieved_docs": len(retrieved_docs),
                        "retrieval_time": retrieval_time,
                        "total_time": total_time,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            print(f"   ✅ Generated response in {generation_time:.2f}s")
            
            # Step 5: Prepare final response
            rag_response = {
                "success": True,
                "response": llm_result["response"],
                "sources": sources,
                "query": query,
                "metadata": {
                    "retrieval": {
                        "documents_found": len(retrieved_docs),
                        "documents_used": len([doc for doc in retrieved_docs if doc.get('score', 0) > self.relevance_threshold]),
                        "retrieval_time": retrieval_time,
                        "top_k": top_k
                    },
                    "generation": {
                        "model": llm_result["metadata"]["model"],
                        "temperature": llm_result["metadata"]["temperature"],
                        "generation_time": generation_time,
                        "usage": llm_result.get("usage", {}),
                        "finish_reason": llm_result.get("finish_reason")
                    },
                    "total_time": total_time,
                    "timestamp": datetime.now().isoformat(),
                    "context_length": len(context)
                }
            }
            
            # Add search details if requested
            if include_search_details:
                rag_response["search_details"] = {
                    "retrieved_documents": retrieved_docs,
                    "context_used": context,
                    "system_message": system_message
                }
            
            self.logger.info(f"RAG response generated successfully in {total_time:.2f}s")
            return rag_response
            
        except Exception as e:
            error_response = {
                "success": False,
                "response": None,
                "error": str(e),
                "error_type": type(e).__name__,
                "metadata": {
                    "query": query,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            self.logger.error(f"Error generating RAG response: {e}")
            return error_response
    
    def generate_streaming_rag_response(self, 
                                       query: str,
                                       top_k: Optional[int] = None,
                                       custom_system_prompt: Optional[str] = None,
                                       temperature: Optional[float] = None,
                                       max_tokens: Optional[int] = None,
                                       conversation_history: Optional[List[Dict[str, str]]] = None,
                                       **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Generate a streaming RAG response.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve (optional)
            custom_system_prompt: Custom system prompt template (optional)
            temperature: LLM temperature (optional)
            max_tokens: Maximum response tokens (optional)
            conversation_history: Previous conversation exchanges for context (optional)
            **kwargs: Additional parameters for LLM generation
            
        Yields:
            Dict containing streaming response chunks and metadata
        """
        try:
            start_time = time.time()
            
            # Yield initial status
            yield {
                "type": "rag_start",
                "status": "starting_retrieval",
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
            
            # Step 1: Retrieve relevant documents
            retrieval_start = time.time()
            top_k = top_k or self.default_top_k
            retrieved_docs = self._retrieve_documents(query, top_k)
            retrieval_time = time.time() - retrieval_start
            
            # Yield retrieval results
            yield {
                "type": "retrieval_complete",
                "documents_found": len(retrieved_docs),
                "retrieval_time": retrieval_time,
                "sources": self._extract_sources(retrieved_docs) if retrieved_docs else []
            }
            
            if not retrieved_docs:
                # No relevant documents found
                yield {
                    "type": "no_context",
                    "status": "no_relevant_documents",
                    "message": "No relevant documents found. Generating response with conversation history only."
                }
                
                # Create system message with conversation history even without document context
                system_message = self._create_rag_system_message(
                    context="No relevant documents found for this query.",
                    custom_prompt=custom_system_prompt,
                    conversation_history=conversation_history
                )
                
                # Generate response with conversation history but without document context
                for chunk in self.llm_manager.generate_streaming_response(
                    query=query,
                    system_message=system_message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                ):
                    # Add RAG metadata to chunks
                    chunk["rag_metadata"] = {
                        "has_context": False,
                        "has_conversation_history": conversation_history is not None and len(conversation_history) > 0,
                        "documents_used": 0
                    }
                    yield chunk
                return
            
            # Step 2: Build context
            yield {
                "type": "context_building",
                "status": "building_context",
                "documents_count": len(retrieved_docs)
            }
            
            context, sources = self._build_context(retrieved_docs, query)
            
            # Step 3: Create system message
            system_message = self._create_rag_system_message(
                context, custom_system_prompt, conversation_history
            )
            
            # Yield context ready status
            yield {
                "type": "context_ready",
                "status": "context_built",
                "context_length": len(context),
                "sources": sources
            }
            
            # Step 4: Generate streaming LLM response
            yield {
                "type": "generation_start",
                "status": "starting_generation"
            }
            
            generation_start = time.time()
            
            for chunk in self.llm_manager.generate_streaming_response(
                query=query,
                system_message=system_message,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            ):
                # Add RAG metadata to chunks
                if chunk.get("type") == "complete":
                    generation_time = time.time() - generation_start
                    total_time = time.time() - start_time
                    
                    chunk["rag_metadata"] = {
                        "has_context": True,
                        "documents_used": len(retrieved_docs),
                        "sources": sources,
                        "retrieval_time": retrieval_time,
                        "generation_time": generation_time,
                        "total_time": total_time,
                        "context_length": len(context)
                    }
                else:
                    chunk["rag_metadata"] = {
                        "has_context": True,
                        "documents_used": len(retrieved_docs)
                    }
                
                yield chunk
            
        except Exception as e:
            # Yield error information
            yield {
                "type": "rag_error",
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "metadata": {
                    "query": query,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            self.logger.error(f"Error in streaming RAG response: {e}")
    
    def _retrieve_documents(self, query: str, top_k: int) -> List[Dict]:
        """Retrieve relevant documents using the search index manager."""
        try:
            # Use vector search if available
            retrieved_docs = self.search_manager.search_similar_chunks(query, top_k)
            
            # Filter by relevance threshold
            if self.relevance_threshold > 0:
                retrieved_docs = [
                    doc for doc in retrieved_docs 
                    if doc.get('score', 0) >= self.relevance_threshold
                ]
            
            return retrieved_docs
            
        except Exception as e:
            self.logger.error(f"Error retrieving documents: {e}")
            return []
    
    def _build_context(self, retrieved_docs: List[Dict], query: str) -> Tuple[str, List[Dict]]:
        """Build context from retrieved documents and extract source information."""
        context_parts = []
        sources = []
        current_tokens = 0
        
        for i, doc in enumerate(retrieved_docs):
            # Extract document information
            content = doc.get('content', '')
            source_file = doc.get('source_file', 'Unknown')
            section = doc.get('section', 'Unknown')
            score = doc.get('score', 0)
            
            # Estimate tokens for this document
            doc_tokens = self.llm_manager.estimate_tokens(content)
            
            # Check if adding this document would exceed token limit
            if current_tokens + doc_tokens > self.max_context_tokens:
                self.logger.info(f"Context token limit reached. Using {i} of {len(retrieved_docs)} documents")
                break
            
            # Add document to context
            context_parts.append(f"Document {i+1}:")
            context_parts.append(f"Source: {Path(source_file).name}")
            context_parts.append(f"Section: {section}")
            context_parts.append(f"Relevance Score: {score:.3f}")
            context_parts.append(f"Content: {content}")
            context_parts.append("---")
            
            # Add to sources
            sources.append({
                "document_id": doc.get('id', f'doc_{i+1}'),
                "source_file": source_file,
                "section": section,
                "score": score,
                "chunk_index": doc.get('chunk_index', 0),
                "content_preview": content[:200] + "..." if len(content) > 200 else content
            })
            
            current_tokens += doc_tokens
        
        context = "\n".join(context_parts)
        
        self.logger.info(f"Built context from {len(sources)} documents ({current_tokens} estimated tokens)")
        return context, sources
    
    def _create_rag_system_message(self, context: str, custom_prompt: Optional[str] = None, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Create system message with retrieved context and conversation history."""
        # Build conversation history section
        history_section = ""
        if conversation_history and len(conversation_history) > 0:
            history_section = "\n\nConversation History (for context):\n"
            for i, exchange in enumerate(conversation_history[-10:], 1):  # Last 10 exchanges
                query = exchange.get('query', '')
                response = exchange.get('response', '')
                history_section += f"Exchange {i}:\n"
                history_section += f"User: {query}\n"
                history_section += f"Assistant: {response}\n\n"
        
        if custom_prompt:
            # Use custom prompt template - should include {context} and optionally {history} placeholders
            try:
                if "{history}" in custom_prompt:
                    return custom_prompt.format(context=context, history=history_section)
                else:
                    return custom_prompt.format(context=context) + history_section
            except KeyError:
                self.logger.warning("Custom prompt missing {context} placeholder, appending context and history")
                return f"{custom_prompt}\n\nRelevant Context:\n{context}{history_section}"
        
        # Default RAG system message with conversation history
        if "No relevant documents found" in context:
            # Special case: no document context but potentially conversation history
            if conversation_history and len(conversation_history) > 0:
                default_prompt = """You are a helpful AI assistant with access to conversation history. 
While no relevant documents were found for this specific query, use the conversation history to understand the context and provide a helpful response.

Guidelines:
- Use conversation history to understand what the user is referring to
- Reference previous exchanges to provide contextual continuity
- If the query refers to something discussed earlier, acknowledge and build upon that discussion
- Be helpful and provide relevant information based on the conversation flow
- If you cannot fully answer based on conversation history alone, acknowledge this limitation{history}

Please provide a helpful response based on the conversation history."""
            else:
                default_prompt = """You are a helpful AI assistant. No relevant documents were found for this query, and there is no conversation history available.

Please provide a helpful general response to the user's question: {context}"""
        else:
            # Normal case: document context available
            default_prompt = """You are a helpful AI assistant with access to relevant document context and conversation history. 
Use the provided context to answer the user's question accurately and comprehensively, taking into account the conversation history for better contextual understanding.

Guidelines:
- Base your response primarily on the provided context
- Use conversation history to understand the flow of the discussion and provide contextually relevant answers
- If the context doesn't contain enough information, acknowledge this limitation
- Cite specific sources when making claims
- Provide direct quotes when helpful
- If asked about something not in the context, clearly state this
- Be concise but thorough in your responses
- Reference previous exchanges when relevant to provide continuity{history}

Relevant Context:
{context}

Please provide helpful, accurate responses based on this context and conversation history."""
        
        return default_prompt.format(context=context, history=history_section)
    
    def _extract_sources(self, retrieved_docs: List[Dict]) -> List[Dict]:
        """Extract simplified source information from retrieved documents."""
        sources = []
        for i, doc in enumerate(retrieved_docs):
            sources.append({
                "document_id": doc.get('id', f'doc_{i+1}'),
                "source_file": Path(doc.get('source_file', 'Unknown')).name,
                "section": doc.get('section', 'Unknown'),
                "score": doc.get('score', 0)
            })
        return sources
    
    def _generate_no_context_response(self, query: str, temperature: Optional[float], 
                                    max_tokens: Optional[int], conversation_history: Optional[List[Dict[str, str]]] = None, **kwargs) -> Dict[str, Any]:
        """Generate response when no relevant context is found but include conversation history."""
        # Create system message with conversation history even without document context
        system_message = self._create_rag_system_message(
            context="No relevant documents found for this query.",
            custom_prompt=None,
            conversation_history=conversation_history
        )
        
        llm_result = self.llm_manager.generate_response(
            query=query,
            system_message=system_message,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        if llm_result.get("success"):
            return {
                "success": True,
                "response": llm_result["response"],
                "sources": [],
                "query": query,
                "metadata": {
                    "has_context": False,
                    "has_conversation_history": conversation_history is not None and len(conversation_history) > 0,
                    "conversation_exchanges_used": len(conversation_history) if conversation_history else 0,
                    "retrieval": {
                        "documents_found": 0,
                        "documents_used": 0
                    },
                    "generation": llm_result.get("metadata", {}),
                    "timestamp": datetime.now().isoformat()
                }
            }
        else:
            return {
                "success": False,
                "response": None,
                "error": f"No context found and LLM generation failed: {llm_result.get('error')}",
                "sources": [],
                "query": query,
                "metadata": {
                    "has_context": False,
                    "has_conversation_history": conversation_history is not None and len(conversation_history) > 0,
                    "conversation_exchanges_used": len(conversation_history) if conversation_history else 0,
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    def search_documents(self, query: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Search for documents without generating a response.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            
        Returns:
            Dict containing search results and metadata
        """
        try:
            start_time = time.time()
            top_k = top_k or self.default_top_k
            
            retrieved_docs = self._retrieve_documents(query, top_k)
            search_time = time.time() - start_time
            
            return {
                "success": True,
                "query": query,
                "documents": retrieved_docs,
                "sources": self._extract_sources(retrieved_docs),
                "metadata": {
                    "documents_found": len(retrieved_docs),
                    "search_time": search_time,
                    "top_k": top_k,
                    "relevance_threshold": self.relevance_threshold,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "query": query,
                "metadata": {
                    "timestamp": datetime.now().isoformat()
                }
            }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate RAG Manager configuration.
        
        Returns:
            Dict containing validation results
        """
        validation = {
            "valid": False,
            "issues": [],
            "recommendations": [],
            "component_status": {}
        }
        
        # Validate search manager
        if self.search_manager:
            validation["component_status"]["search_manager"] = "✅ Available"
            # Test search functionality
            try:
                test_results = self.search_manager.search_similar_chunks("test", 1)
                validation["component_status"]["search_functionality"] = "✅ Working"
            except Exception as e:
                validation["component_status"]["search_functionality"] = f"❌ Error: {e}"
                validation["issues"].append(f"Search functionality error: {e}")
        else:
            validation["component_status"]["search_manager"] = "❌ Missing"
            validation["issues"].append("SearchIndexManager not configured")
        
        # Validate LLM manager
        if self.llm_manager:
            validation["component_status"]["llm_manager"] = "✅ Available"
            llm_validation = self.llm_manager.validate_configuration()
            if llm_validation["valid"]:
                validation["component_status"]["llm_functionality"] = "✅ Working"
            else:
                validation["component_status"]["llm_functionality"] = "❌ Configuration issues"
                validation["issues"].extend(llm_validation["issues"])
        else:
            validation["component_status"]["llm_manager"] = "❌ Missing"
            validation["issues"].append("LLMManager not configured")
        
        # Check parameters
        if self.default_top_k <= 0:
            validation["issues"].append(f"Invalid default_top_k: {self.default_top_k}")
        
        if self.max_context_tokens <= 0:
            validation["issues"].append(f"Invalid max_context_tokens: {self.max_context_tokens}")
        
        if not (0.0 <= self.relevance_threshold <= 1.0):
            validation["issues"].append(f"Invalid relevance_threshold: {self.relevance_threshold}")
        
        # Set validity
        validation["valid"] = len(validation["issues"]) == 0
        
        if validation["valid"]:
            validation["status"] = "✅ RAG Manager configuration is valid"
        else:
            validation["status"] = f"❌ Found {len(validation['issues'])} configuration issues"
        
        # Add recommendations
        if validation["valid"]:
            validation["recommendations"] = [
                "Configuration is valid. RAG Manager is ready to use.",
                f"Current settings: top_k={self.default_top_k}, max_tokens={self.max_context_tokens}",
                "Test with a sample query to verify end-to-end functionality."
            ]
        
        return validation
    
    def get_configuration_info(self) -> Dict[str, Any]:
        """
        Get detailed configuration information.
        
        Returns:
            Dict containing configuration details
        """
        config = {
            "rag_settings": {
                "default_top_k": self.default_top_k,
                "max_context_tokens": self.max_context_tokens,
                "include_source_info": self.include_source_info,
                "relevance_threshold": self.relevance_threshold
            },
            "search_manager": {
                "available": self.search_manager is not None,
                "type": type(self.search_manager).__name__ if self.search_manager else None
            },
            "llm_manager": {
                "available": self.llm_manager is not None,
                "type": type(self.llm_manager).__name__ if self.llm_manager else None
            }
        }
        
        # Add detailed info if managers are available
        if self.search_manager and hasattr(self.search_manager, 'azure_search_index_name'):
            config["search_manager"]["index_name"] = self.search_manager.azure_search_index_name
            config["search_manager"]["chunk_size"] = getattr(self.search_manager, 'chunk_size', 'Unknown')
        
        if self.llm_manager:
            llm_info = self.llm_manager.get_model_info()
            config["llm_manager"].update(llm_info)
        
        return config
    
    def test_rag_pipeline(self, test_query: str = "What is cyber security?") -> Dict[str, Any]:
        """
        Test the complete RAG pipeline with a sample query.
        
        Args:
            test_query: Query to test with
            
        Returns:
            Dict containing test results
        """
        print(f"🧪 Testing RAG pipeline with query: '{test_query}'")
        
        try:
            # Test search functionality
            print("   🔍 Testing document retrieval...")
            search_result = self.search_documents(test_query, top_k=3)
            
            if not search_result["success"]:
                return {
                    "success": False,
                    "stage_failed": "document_retrieval",
                    "error": search_result["error"],
                    "test_query": test_query
                }
            
            print(f"   ✅ Retrieved {search_result['metadata']['documents_found']} documents")
            
            # Test RAG response generation
            print("   🤖 Testing RAG response generation...")
            rag_result = self.generate_rag_response(
                query=test_query,
                top_k=3,
                max_tokens=150  # Shorter response for testing
            )
            
            if not rag_result["success"]:
                return {
                    "success": False,
                    "stage_failed": "rag_generation",
                    "error": rag_result["error"],
                    "test_query": test_query,
                    "search_results": search_result
                }
            
            print(f"   ✅ Generated RAG response successfully")
            print(f"   📊 Response length: {len(rag_result['response'])} characters")
            print(f"   📚 Used {len(rag_result['sources'])} sources")
            
            return {
                "success": True,
                "test_query": test_query,
                "search_results": search_result,
                "rag_response": rag_result,
                "summary": {
                    "documents_retrieved": search_result['metadata']['documents_found'],
                    "sources_used": len(rag_result['sources']),
                    "response_length": len(rag_result['response']),
                    "total_time": rag_result['metadata']['total_time']
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "stage_failed": "unknown",
                "error": str(e),
                "error_type": type(e).__name__,
                "test_query": test_query
            }