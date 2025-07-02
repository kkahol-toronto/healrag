#!/usr/bin/env python3
"""
HEALRAG RAG Example Script
=========================

This script demonstrates how to use the new LLM Manager and RAG Manager
components for both streaming and non-streaming RAG responses.

Updated with optimized settings based on debug findings.
"""

import os
import json
import asyncio
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not available, using system environment variables")

# Import HEALRAG components
from healraglib import RAGManager, LLMManager, SearchIndexManager, StorageManager

def main():
    """Main RAG demonstration function."""
    
    print("🎯 HEALRAG RAG System Demo")
    print("=" * 60)
    
    # Configuration from environment variables
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
    azure_openai_chat_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")  # e.g., gpt-4, gpt-35-turbo
    azure_openai_embedding_deployment = os.getenv("AZURE_TEXT_EMBEDDING_MODEL")  # e.g., text-embedding-ada-002
    
    azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    azure_search_key = os.getenv("AZURE_SEARCH_KEY")
    azure_search_index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "security-index")
    
    print(f"🔧 Configuration:")
    print(f"   OpenAI Endpoint: {azure_openai_endpoint}")
    print(f"   Chat Model: {azure_openai_chat_deployment}")
    print(f"   Embedding Model: {azure_openai_embedding_deployment}")
    print(f"   Search Endpoint: {azure_search_endpoint}")
    print(f"   Search Index: {azure_search_index_name}")
    
    # Check configuration
    required_vars = [
        azure_openai_endpoint, azure_openai_key, azure_openai_chat_deployment,
        azure_openai_embedding_deployment, azure_search_endpoint, azure_search_key
    ]
    
    if not all(required_vars):
        print("❌ Missing required environment variables")
        print("💡 Please set all required Azure OpenAI and Search variables")
        return
    
    try:
        # Step 1: Initialize Individual Components
        print(f"\n🔧 Step 1: Initializing Components...")
        
        # Initialize Search Index Manager (for searching existing index)
        print("   🔍 Initializing Search Index Manager...")
        search_manager = SearchIndexManager(
            storage_manager=None,  # Not needed for search-only operations
            azure_openai_endpoint=azure_openai_endpoint,
            azure_openai_key=azure_openai_key,
            azure_openai_deployment=azure_openai_embedding_deployment,
            azure_search_endpoint=azure_search_endpoint,
            azure_search_key=azure_search_key,
            azure_search_index_name=azure_search_index_name
        )
        
        # Initialize LLM Manager
        print("   🤖 Initializing LLM Manager...")
        llm_manager = LLMManager(
            azure_openai_endpoint=azure_openai_endpoint,
            azure_openai_key=azure_openai_key,
            azure_openai_deployment=azure_openai_chat_deployment,
            default_temperature=0.7,
            default_max_tokens=500
        )
        
        # Initialize RAG Manager with optimized settings based on debug findings
        print("   🎯 Initializing RAG Manager with optimized settings...")
        rag_manager = RAGManager(
            search_index_manager=search_manager,
            llm_manager=llm_manager,
            default_top_k=3,  # Reduced from 5 based on debug results
            max_context_tokens=6000,  # Increased from 3000 to handle larger documents
            relevance_threshold=0.02  # Added slight threshold for better quality
        )
        
        print("   ✅ All components initialized successfully!")
        print(f"   📊 RAG Settings: top_k={rag_manager.default_top_k}, max_tokens={rag_manager.max_context_tokens}")
        
        # Step 2: Validate Configuration
        print(f"\n🔍 Step 2: Validating Configuration...")
        
        # Validate LLM Manager
        llm_validation = llm_manager.validate_configuration()
        print(f"   LLM Manager: {llm_validation['status']}")
        
        # Validate RAG Manager
        rag_validation = rag_manager.validate_configuration()
        print(f"   RAG Manager: {rag_validation['status']}")
        
        if not (llm_validation['valid'] and rag_validation['valid']):
            print("❌ Configuration validation failed")
            if not llm_validation['valid']:
                for issue in llm_validation['issues']:
                    print(f"     LLM Issue: {issue}")
            if not rag_validation['valid']:
                for issue in rag_validation['issues']:
                    print(f"     RAG Issue: {issue}")
            return
        
        # Step 3: Test RAG Pipeline
        print(f"\n🧪 Step 3: Testing RAG Pipeline...")
        test_result = rag_manager.test_rag_pipeline("What is our incident management process?")
        
        if test_result['success']:
            print("   ✅ RAG pipeline test successful!")
            print(f"   📊 Retrieved {test_result['summary']['documents_retrieved']} documents")
            print(f"   📚 Used {test_result['summary']['sources_used']} sources")
            print(f"   ⏱️  Total time: {test_result['summary']['total_time']:.2f}s")
            print(f"   📝 Response preview: {test_result['rag_response']['response'][:150]}...")
        else:
            print(f"   ❌ RAG pipeline test failed: {test_result['error']}")
            print(f"   🔍 Failed at stage: {test_result.get('stage_failed', 'unknown')}")
            return
        
        # Step 4: Demonstrate Document Search with Real Queries
        print(f"\n🔍 Step 4: Document Search Demo...")
        
        search_queries = [
            "incident management procedures",
            "What is our incident management process?",  # Same as successful debug query
            "risk assessment framework", 
            "access control policies",
            "third party security requirements"
        ]
        
        for query in search_queries:
            print(f"\n   Searching for: '{query}'")
            search_result = rag_manager.search_documents(query, top_k=3)
            
            if search_result['success']:
                docs_found = search_result['metadata']['documents_found']
                print(f"   ✅ Found {docs_found} documents in {search_result['metadata']['search_time']:.2f}s")
                
                if docs_found > 0:
                    for i, source in enumerate(search_result['sources'][:2], 1):
                        print(f"     {i}. {source['source_file']} - {source['section']} (Score: {source['score']:.3f})")
                else:
                    print("     📝 No documents found - consider adjusting relevance threshold")
            else:
                print(f"   ❌ Search failed: {search_result['error']}")
        
        # Step 5: Non-Streaming RAG Responses with Context-Rich Queries
        print(f"\n🤖 Step 5: Non-Streaming RAG Demo...")
        
        rag_queries = [
            "What are the key components of our cyber security incident management process?",
            "How should we handle security incidents according to our policies?",
            "What are the requirements for third-party security assessments?",
            "Explain our approach to information security risk management."
        ]
        
        for query in rag_queries:
            print(f"\n   Query: '{query}'")
            
            rag_response = rag_manager.generate_rag_response(
                query=query,
                top_k=3,
                temperature=0.7,
                max_tokens=400,  # Increased for more detailed responses
                include_search_details=False  # Set to True for debugging
            )
            
            if rag_response['success']:
                print(f"   ✅ Response generated in {rag_response['metadata']['total_time']:.2f}s")
                print(f"   📚 Sources used: {len(rag_response['sources'])}")
                print(f"   📊 Context length: {rag_response['metadata']['context_length']} chars")
                print(f"   📝 Response preview: {rag_response['response'][:200]}...")
                
                if rag_response['sources']:
                    print(f"   📋 Sources:")
                    for source in rag_response['sources'][:2]:
                        print(f"     - {source['source_file']} ({source['section']}) - Score: {source['score']:.3f}")
                else:
                    print("   ⚠️  No sources used - check relevance threshold or document content")
            else:
                print(f"   ❌ RAG response failed: {rag_response['error']}")
        
        # Step 6: Streaming RAG Response Demo
        print(f"\n🌊 Step 6: Streaming RAG Demo...")
        
        streaming_query = "Explain our incident management process step by step."
        print(f"   Streaming query: '{streaming_query}'")
        print(f"   Response: ", end="", flush=True)
        
        full_response = ""
        sources_used = []
        retrieval_info = {}
        
        try:
            for chunk in rag_manager.generate_streaming_rag_response(
                query=streaming_query,
                top_k=3,
                temperature=0.7,
                max_tokens=500
            ):
                if chunk.get("type") == "retrieval_complete":
                    retrieval_info = {
                        "documents_found": chunk.get("documents_found", 0),
                        "retrieval_time": chunk.get("retrieval_time", 0),
                        "sources": chunk.get("sources", [])
                    }
                    print(f"\n   📊 Retrieved {retrieval_info['documents_found']} documents in {retrieval_info['retrieval_time']:.2f}s")
                    print(f"   💬 Streaming response: ", end="", flush=True)
                
                elif chunk.get("type") == "context_ready":
                    context_length = chunk.get("context_length", 0)
                    sources_count = len(chunk.get("sources", []))
                    print(f"\n   🔧 Context built: {context_length} chars, {sources_count} sources")
                    print(f"   💬 Generated response: ", end="", flush=True)
                
                elif chunk.get("type") == "chunk" and chunk.get("content"):
                    # Print content as it streams
                    print(chunk["content"], end="", flush=True)
                    full_response += chunk["content"]
                
                elif chunk.get("type") == "complete":
                    # Final response data
                    rag_metadata = chunk.get("rag_metadata", {})
                    if "sources" in rag_metadata:
                        sources_used = rag_metadata["sources"]
                    
                    print(f"\n   ✅ Streaming completed!")
                    print(f"   📊 Total time: {rag_metadata.get('total_time', 0):.2f}s")
                    print(f"   📚 Sources used: {len(sources_used)}")
                    print(f"   📏 Response length: {len(full_response)} characters")
                
                elif chunk.get("type") == "no_context":
                    print(f"\n   ⚠️  {chunk.get('message', 'No context available')}")
                    print(f"   💬 Response: ", end="", flush=True)
                
                elif chunk.get("type") == "rag_error":
                    print(f"\n   ❌ Streaming error: {chunk.get('error')}")
                    break
        
        except Exception as e:
            print(f"\n   ❌ Streaming demo error: {e}")
        
        # Step 7: Custom System Prompt Demo
        print(f"\n🎨 Step 7: Custom System Prompt Demo...")
        
        custom_prompt = """You are a cybersecurity expert consultant for Point32Health. 
Based on the provided context from our organization's security documentation, 
provide detailed, actionable advice. Format your response as:

1. Summary of relevant policies
2. Key requirements and procedures
3. Implementation recommendations
4. Compliance considerations

Always reference specific document sections when possible.

Context:
{context}"""
        
        custom_query = "What should we consider when implementing access controls?"
        print(f"   Custom prompt query: '{custom_query}'")
        
        custom_response = rag_manager.generate_rag_response(
            query=custom_query,
            custom_system_prompt=custom_prompt,
            top_k=4,
            temperature=0.6,
            max_tokens=600
        )
        
        if custom_response['success']:
            print(f"   ✅ Custom response generated!")
            print(f"   📝 Response preview: {custom_response['response'][:300]}...")
            print(f"   📚 Sources: {len(custom_response['sources'])}")
            print(f"   📊 Context used: {custom_response['metadata']['context_length']} chars")
        else:
            print(f"   ❌ Custom response failed: {custom_response['error']}")
        
        # Step 8: Direct LLM Usage Demo
        print(f"\n💬 Step 8: Direct LLM Usage Demo...")
        
        # Test direct LLM without RAG
        direct_query = "Explain the importance of cybersecurity in modern organizations."
        print(f"   Direct LLM query: '{direct_query}'")
        
        llm_response = llm_manager.generate_response(
            query=direct_query,
            system_message="You are a cybersecurity expert. Provide concise, professional advice.",
            temperature=0.8,
            max_tokens=200
        )
        
        if llm_response['success']:
            print(f"   ✅ Direct LLM response generated!")
            print(f"   📝 Response: {llm_response['response'][:200]}...")
            print(f"   📊 Tokens used: {llm_response['usage']['total_tokens']}")
            print(f"   ⏱️  Time: {llm_response['metadata']['response_time']:.2f}s")
        else:
            print(f"   ❌ Direct LLM failed: {llm_response['error']}")
        
        # Step 9: Advanced Configuration and Tuning Demo
        print(f"\n🔧 Step 9: Advanced Configuration Demo...")
        
        # Test different relevance thresholds
        print(f"   Testing different relevance thresholds...")
        test_query = "What is our incident management process?"
        
        thresholds_to_test = [0.0, 0.01, 0.02, 0.03, 0.05]
        for threshold in thresholds_to_test:
            rag_manager.relevance_threshold = threshold
            result = rag_manager.search_documents(test_query, top_k=3)
            docs_found = result['metadata']['documents_found'] if result['success'] else 0
            print(f"     Threshold {threshold:.3f}: {docs_found} documents")
        
        # Reset to optimal threshold
        rag_manager.relevance_threshold = 0.02
        
        # Test different top_k values
        print(f"   Testing different top_k values...")
        for top_k in [1, 3, 5, 10]:
            result = rag_manager.search_documents(test_query, top_k=top_k)
            if result['success']:
                search_time = result['metadata']['search_time']
                docs_found = result['metadata']['documents_found']
                print(f"     Top-{top_k}: {docs_found} docs in {search_time:.2f}s")
        
        # Step 10: Configuration Summary
        print(f"\n📋 Step 10: System Configuration Summary...")
        
        config_info = rag_manager.get_configuration_info()
        print(f"   RAG Settings:")
        print(f"     - Top K: {config_info['rag_settings']['default_top_k']}")
        print(f"     - Max Context Tokens: {config_info['rag_settings']['max_context_tokens']}")
        print(f"     - Relevance Threshold: {config_info['rag_settings']['relevance_threshold']}")
        
        if 'llm_manager' in config_info and config_info['llm_manager']['available']:
            print(f"   LLM Settings:")
            print(f"     - Model: {config_info['llm_manager']['deployment_name']}")
            print(f"     - Temperature: {config_info['llm_manager']['default_temperature']}")
            print(f"     - Max Tokens: {config_info['llm_manager']['default_max_tokens']}")
        
        # Step 11: Performance and Usage Tips (Updated)
        print(f"\n💡 Step 11: Updated Usage Tips and Best Practices...")
        
        tips = [
            "🎯 Use specific queries for better retrieval results",
            "📊 Start with top_k=3, increase if you need more context", 
            "🔧 Set max_context_tokens=6000+ for documents with large chunks",
            "🎚️  Use relevance_threshold=0.02-0.05 for better quality",
            "🌡️  Lower temperature (0.3-0.5) for factual responses",
            "🌡️  Higher temperature (0.7-0.9) for creative responses",
            "📝 Use custom system prompts for domain-specific formatting",
            "⚡ Use streaming for better user experience in applications",
            "💾 Monitor context_length in responses to optimize token usage",
            "📏 Check sources count - if 0, adjust relevance threshold",
            "🐛 Use include_search_details=True for debugging"
        ]
        
        for tip in tips:
            print(f"   {tip}")
        
        # Step 12: Debugging Helper
        print(f"\n🐛 Step 12: Quick Debugging Guide...")
        
        debug_tips = [
            "❌ No sources used? → Lower relevance_threshold or increase max_context_tokens",
            "❌ No documents found? → Check index name and search query",
            "❌ Poor quality responses? → Increase relevance_threshold",
            "❌ Truncated responses? → Increase max_tokens in generate_rag_response",
            "❌ Slow responses? → Reduce top_k or max_context_tokens",
            "🔍 Use the debug script: python debug_scripts/debug_rag_context.py"
        ]
        
        for tip in debug_tips:
            print(f"   {tip}")
        
        print(f"\n🎉 HEALRAG RAG Demo Completed Successfully!")
        print(f"   ✅ All components working correctly")
        print(f"   📚 Ready for integration into your applications")
        print(f"   🔗 Use rag_manager.generate_rag_response() for non-streaming")
        print(f"   🌊 Use rag_manager.generate_streaming_rag_response() for streaming")
        print(f"   🎯 Optimized settings: top_k=3, max_tokens=6000, threshold=0.02")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

def demonstrate_streaming_async():
    """Demonstrate async streaming patterns (optional advanced usage)."""
    print(f"\n🌊 Advanced: Async Streaming Pattern Example...")
    
    # This is a conceptual example - you could implement async wrappers
    # around the streaming responses for better integration with async frameworks
    
    async def async_rag_stream(rag_manager, query):
        """Example async wrapper for streaming RAG responses."""
        chunks = []
        
        # Simulate async processing of streaming chunks
        for chunk in rag_manager.generate_streaming_rag_response(query):
            chunks.append(chunk)
            # In real async implementation, you'd yield or await here
            await asyncio.sleep(0.01)  # Simulate async delay
        
        return chunks
    
    print("   💡 You can wrap streaming responses in async functions")
    print("   💡 This enables integration with FastAPI, Django Channels, etc.")
    print("   💡 Consider using asyncio.Queue for async chunk processing")

if __name__ == "__main__":
    main()
    
    # Uncomment to see async streaming example
    # asyncio.run(demonstrate_streaming_async())