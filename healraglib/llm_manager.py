"""
LLM Manager for HEALRAG

Handles Large Language Model interactions with Azure OpenAI,
providing both streaming and non-streaming response capabilities.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Iterator, Union, Any
from datetime import datetime
import time

try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False


class LLMManager:
    """
    LLM Manager for HEALRAG library.
    
    Handles interactions with Azure OpenAI language models for chat completions,
    supporting both streaming and non-streaming responses.
    """
    
    def __init__(self, 
                 azure_openai_endpoint: Optional[str] = None,
                 azure_openai_key: Optional[str] = None,
                 azure_openai_deployment: Optional[str] = None,
                 default_temperature: float = 0.7,
                 default_max_tokens: int = 1000,
                 default_system_message: str = "You are a helpful AI assistant."):
        """
        Initialize the LLM Manager.
        
        Args:
            azure_openai_endpoint: Azure OpenAI endpoint
            azure_openai_key: Azure OpenAI API key
            azure_openai_deployment: Azure OpenAI chat deployment name (e.g., gpt-4, gpt-35-turbo)
            default_temperature: Default temperature for responses (0.0-1.0)
            default_max_tokens: Default maximum tokens for responses
            default_system_message: Default system message for conversations
        """
        self.azure_openai_endpoint = azure_openai_endpoint
        self.azure_openai_key = azure_openai_key
        self.azure_openai_deployment = azure_openai_deployment
        self.default_temperature = default_temperature
        self.default_max_tokens = default_max_tokens
        self.default_system_message = default_system_message
        self.logger = logging.getLogger(__name__)
        
        # Initialize Azure OpenAI client
        self.openai_client = None
        if all([azure_openai_endpoint, azure_openai_key, azure_openai_deployment]):
            self._initialize_openai_client()
    
    def _initialize_openai_client(self):
        """Initialize Azure OpenAI client for chat completions."""
        try:
            if not AZURE_OPENAI_AVAILABLE:
                raise ImportError("openai package not installed. Install with: pip install openai")
            
            print(f"🤖 Initializing Azure OpenAI LLM client...")
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
            test_response = self.openai_client.chat.completions.create(
                model=self.azure_openai_deployment,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            print(f"   ✅ Azure OpenAI LLM client initialized successfully")
            print(f"   Test response: {test_response.choices[0].message.content}")
            
        except ImportError as e:
            print(f"   ❌ {e}")
            self.logger.error(f"Failed to initialize Azure OpenAI client: {e}")
        except Exception as e:
            print(f"   ❌ Failed to initialize Azure OpenAI LLM client: {e}")
            self.logger.error(f"Failed to initialize Azure OpenAI client: {e}")
    
    def generate_response(self, 
                         query: str,
                         system_message: Optional[str] = None,
                         conversation_history: Optional[List[Dict]] = None,
                         temperature: Optional[float] = None,
                         max_tokens: Optional[int] = None,
                         **kwargs) -> Dict[str, Any]:
        """
        Generate a non-streaming response from the LLM.
        
        Args:
            query: User query/prompt
            system_message: System message to set context (optional)
            conversation_history: Previous conversation messages (optional)
            temperature: Response creativity (0.0-1.0, optional)
            max_tokens: Maximum response tokens (optional)
            **kwargs: Additional parameters for chat completion
            
        Returns:
            Dict containing response, metadata, and usage statistics
        """
        if not self.openai_client:
            raise ValueError("Azure OpenAI client not initialized. Check your configuration.")
        
        try:
            # Build messages
            messages = self._build_messages(query, system_message, conversation_history)
            
            # Use provided parameters or defaults
            temperature = temperature if temperature is not None else self.default_temperature
            max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
            
            # Log request details
            self.logger.info(f"Generating response for query: {query[:100]}...")
            self.logger.debug(f"Temperature: {temperature}, Max tokens: {max_tokens}")
            
            start_time = time.time()
            
            # Make API call
            response = self.openai_client.chat.completions.create(
                model=self.azure_openai_deployment,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                **kwargs
            )
            
            end_time = time.time()
            
            # Extract response data
            response_content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            
            # Get usage statistics
            usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }
            
            # Prepare result
            result = {
                "success": True,
                "response": response_content,
                "finish_reason": finish_reason,
                "usage": usage,
                "metadata": {
                    "model": self.azure_openai_deployment,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "response_time": round(end_time - start_time, 2),
                    "timestamp": datetime.now().isoformat()
                },
                "messages": messages  # Include for conversation tracking
            }
            
            self.logger.info(f"Response generated successfully in {result['metadata']['response_time']}s")
            self.logger.debug(f"Usage: {usage}")
            
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "response": None,
                "error": str(e),
                "error_type": type(e).__name__,
                "metadata": {
                    "model": self.azure_openai_deployment,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            self.logger.error(f"Error generating response: {e}")
            return error_result
    
    def generate_streaming_response(self, 
                                   query: str,
                                   system_message: Optional[str] = None,
                                   conversation_history: Optional[List[Dict]] = None,
                                   temperature: Optional[float] = None,
                                   max_tokens: Optional[int] = None,
                                   **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Generate a streaming response from the LLM.
        
        Args:
            query: User query/prompt
            system_message: System message to set context (optional)
            conversation_history: Previous conversation messages (optional)
            temperature: Response creativity (0.0-1.0, optional)
            max_tokens: Maximum response tokens (optional)
            **kwargs: Additional parameters for chat completion
            
        Yields:
            Dict containing chunk data, metadata, and final summary
        """
        if not self.openai_client:
            raise ValueError("Azure OpenAI client not initialized. Check your configuration.")
        
        try:
            # Build messages
            messages = self._build_messages(query, system_message, conversation_history)
            
            # Use provided parameters or defaults
            temperature = temperature if temperature is not None else self.default_temperature
            max_tokens = max_tokens if max_tokens is not None else self.default_max_tokens
            
            # Log request details
            self.logger.info(f"Generating streaming response for query: {query[:100]}...")
            self.logger.debug(f"Temperature: {temperature}, Max tokens: {max_tokens}")
            
            start_time = time.time()
            full_response = ""
            chunk_count = 0
            
            # Yield initial status
            yield {
                "type": "start",
                "status": "starting",
                "metadata": {
                    "model": self.azure_openai_deployment,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Make streaming API call
            stream = self.openai_client.chat.completions.create(
                model=self.azure_openai_deployment,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
                **kwargs
            )
            
            # Process streaming chunks
            for chunk in stream:
                chunk_count += 1
                
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    
                    # Check for content
                    if hasattr(delta, 'content') and delta.content:
                        content = delta.content
                        full_response += content
                        
                        yield {
                            "type": "chunk",
                            "content": content,
                            "chunk_index": chunk_count,
                            "accumulated_content": full_response,
                            "finish_reason": None
                        }
                    
                    # Check for finish reason
                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason
                        end_time = time.time()
                        
                        # Yield final summary
                        yield {
                            "type": "complete",
                            "status": "completed",
                            "full_response": full_response,
                            "finish_reason": finish_reason,
                            "metadata": {
                                "model": self.azure_openai_deployment,
                                "temperature": temperature,
                                "max_tokens": max_tokens,
                                "total_chunks": chunk_count,
                                "response_time": round(end_time - start_time, 2),
                                "timestamp": datetime.now().isoformat(),
                                "response_length": len(full_response)
                            },
                            "messages": messages
                        }
                        
                        self.logger.info(f"Streaming response completed in {round(end_time - start_time, 2)}s")
                        break
            
        except Exception as e:
            # Yield error information
            yield {
                "type": "error",
                "status": "failed",
                "error": str(e),
                "error_type": type(e).__name__,
                "metadata": {
                    "model": self.azure_openai_deployment,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            self.logger.error(f"Error in streaming response: {e}")
    
    def _build_messages(self, 
                       query: str,
                       system_message: Optional[str] = None,
                       conversation_history: Optional[List[Dict]] = None) -> List[Dict]:
        """Build messages array for chat completion."""
        messages = []
        
        # Add system message
        system_msg = system_message or self.default_system_message
        messages.append({"role": "system", "content": system_msg})
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    messages.append(msg)
                else:
                    self.logger.warning(f"Invalid message format in conversation history: {msg}")
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        return messages
    
    def create_system_message(self, 
                             context: str,
                             instructions: str = "",
                             persona: str = "helpful AI assistant") -> str:
        """
        Create a system message with context and instructions.
        
        Args:
            context: Relevant context information
            instructions: Specific instructions for the AI
            persona: AI persona/role description
            
        Returns:
            Formatted system message
        """
        system_parts = [f"You are a {persona}."]
        
        if context:
            system_parts.append(f"\nRelevant Context:\n{context}")
        
        if instructions:
            system_parts.append(f"\nInstructions:\n{instructions}")
        
        system_parts.append("\nPlease provide helpful, accurate, and relevant responses based on the context provided.")
        
        return "".join(system_parts)
    
    def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the LLM configuration.
        
        Returns:
            Dict containing validation results
        """
        validation = {
            "valid": False,
            "issues": [],
            "recommendations": []
        }
        
        # Check required configurations
        if not self.azure_openai_endpoint:
            validation["issues"].append("Missing AZURE_OPENAI_ENDPOINT")
        
        if not self.azure_openai_key:
            validation["issues"].append("Missing AZURE_OPENAI_KEY")
        
        if not self.azure_openai_deployment:
            validation["issues"].append("Missing AZURE_OPENAI_DEPLOYMENT")
        
        if not AZURE_OPENAI_AVAILABLE:
            validation["issues"].append("OpenAI package not installed")
            validation["recommendations"].append("Install with: pip install openai")
        
        if not self.openai_client:
            validation["issues"].append("OpenAI client not initialized")
        
        # Check parameter ranges
        if not (0.0 <= self.default_temperature <= 1.0):
            validation["issues"].append(f"Invalid temperature: {self.default_temperature} (should be 0.0-1.0)")
        
        if self.default_max_tokens <= 0:
            validation["issues"].append(f"Invalid max_tokens: {self.default_max_tokens} (should be > 0)")
        
        # Set validity
        validation["valid"] = len(validation["issues"]) == 0
        
        if validation["valid"]:
            validation["status"] = "✅ LLM configuration is valid"
        else:
            validation["status"] = f"❌ Found {len(validation['issues'])} configuration issues"
        
        return validation
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the configured model.
        
        Returns:
            Dict containing model information
        """
        return {
            "deployment_name": self.azure_openai_deployment,
            "endpoint": self.azure_openai_endpoint,
            "default_temperature": self.default_temperature,
            "default_max_tokens": self.default_max_tokens,
            "default_system_message": self.default_system_message,
            "client_initialized": self.openai_client is not None,
            "azure_openai_available": AZURE_OPENAI_AVAILABLE
        }
    
    def estimate_tokens(self, text: str) -> int:
        """
        Rough estimation of token count for text.
        
        Args:
            text: Text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Rough estimation: ~4 characters per token for English text
        # This is an approximation - for precise counts, use tiktoken library
        return len(text) // 4
    
    def truncate_conversation(self, 
                             messages: List[Dict],
                             max_tokens: int = 3000) -> List[Dict]:
        """
        Truncate conversation history to fit within token limits.
        
        Args:
            messages: List of conversation messages
            max_tokens: Maximum token limit
            
        Returns:
            Truncated messages list
        """
        if not messages:
            return messages
        
        # Always keep system message if present
        truncated = []
        system_message = None
        
        if messages[0].get("role") == "system":
            system_message = messages[0]
            messages = messages[1:]
        
        # Estimate tokens and truncate from the beginning
        current_tokens = 0
        for msg in reversed(messages):
            msg_tokens = self.estimate_tokens(msg.get("content", ""))
            if current_tokens + msg_tokens > max_tokens:
                break
            truncated.insert(0, msg)
            current_tokens += msg_tokens
        
        # Add system message back
        if system_message:
            truncated.insert(0, system_message)
        
        return truncated