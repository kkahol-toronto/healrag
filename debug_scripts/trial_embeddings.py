#!/usr/bin/env python3
"""
Trial Embeddings Test Script

This script tests Azure OpenAI embedding generation with sample text
to verify the configuration is working correctly.
"""

import os
import sys
from pathlib import Path

# Add the healraglib to the path
sys.path.insert(0, str(Path(__file__).parent / "healraglib"))

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")

try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False
    print("❌ openai package not installed. Install with: pip install openai")
    sys.exit(1)

def test_azure_openai_embeddings():
    """Test Azure OpenAI embedding generation"""
    
    # Get environment variables
    azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    azure_openai_key = os.getenv('AZURE_OPENAI_KEY')
    
    # Check both environment variables and show which one is being used
    azure_openai_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')
    azure_text_embedding_model = os.getenv('AZURE_TEXT_EMBEDDING_MODEL')
    
    print(f"🔍 Environment Variable Check:")
    print(f"   AZURE_OPENAI_DEPLOYMENT: {azure_openai_deployment}")
    print(f"   AZURE_TEXT_EMBEDDING_MODEL: {azure_text_embedding_model}")
    
    # Use embedding model if available, otherwise fall back to deployment
    if azure_text_embedding_model:
        azure_openai_deployment = azure_text_embedding_model
        print(f"   ✅ Using AZURE_TEXT_EMBEDDING_MODEL: {azure_text_embedding_model}")
        
        # Validate that it looks like an embedding model
        if "embedding" not in azure_text_embedding_model.lower():
            print(f"   ⚠️  Warning: Model name doesn't contain 'embedding' - this might not be an embedding model")
            
    elif azure_openai_deployment:
        print(f"   ⚠️  Using AZURE_OPENAI_DEPLOYMENT: {azure_openai_deployment}")
        print(f"   💡 Consider using AZURE_TEXT_EMBEDDING_MODEL for embeddings")
        
        # Check if it looks like a chat model (which won't work for embeddings)
        if "gpt" in azure_openai_deployment.lower() and "embedding" not in azure_openai_deployment.lower():
            print(f"   ❌ Warning: This looks like a GPT model, not an embedding model!")
            print(f"   💡 Please use AZURE_TEXT_EMBEDDING_MODEL=text-embedding-ada-002")
    else:
        print(f"   ❌ No embedding model specified")
        azure_openai_deployment = None
    
    print("🧪 Testing Azure OpenAI Embeddings")
    print("=" * 50)
    
    # Check configuration
    print(f"🔧 Configuration Check:")
    print(f"   Azure OpenAI Endpoint: {azure_openai_endpoint}")
    print(f"   Azure OpenAI Key: {'***' if azure_openai_key else 'None'}")
    print(f"   Azure OpenAI Deployment: {azure_openai_deployment}")
    
    if not all([azure_openai_endpoint, azure_openai_key, azure_openai_deployment]):
        print("\n❌ Missing Azure OpenAI configuration!")
        print("Please set the following environment variables:")
        print("   AZURE_OPENAI_ENDPOINT")
        print("   AZURE_OPENAI_KEY")
        print("   AZURE_OPENAI_DEPLOYMENT or AZURE_TEXT_EMBEDDING_MODEL")
        return False
    
    # Initialize Azure OpenAI client
    print(f"\n🔧 Initializing Azure OpenAI client...")
    try:
        client = AzureOpenAI(
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_key,
            api_version="2024-02-15-preview"
        )
        print(f"   ✅ Azure OpenAI client created successfully")
    except Exception as e:
        print(f"   ❌ Failed to create Azure OpenAI client: {e}")
        return False
    
    # Test texts
    test_texts = [
        "This is a simple test sentence for embedding generation.",
        "Cyber security policy and information security standards are critical for organizational protection.",
        "The quick brown fox jumps over the lazy dog. This is a longer text to test embedding generation with more content.",
        "",  # Empty text to test edge case
        "A" * 1000,  # Long text to test token limits
    ]
    
    print(f"\n📝 Testing embedding generation with {len(test_texts)} different texts...")
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n🔍 Test {i}:")
        print(f"   Text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        print(f"   Length: {len(text)} characters")
        
        if not text.strip():
            print(f"   ⚠️  Skipping empty text")
            continue
        
        try:
            # Generate embedding
            print(f"   🧠 Generating embedding...")
            response = client.embeddings.create(
                input=text,
                model=azure_openai_deployment
            )
            
            embedding = response.data[0].embedding
            
            # Print embedding details
            print(f"   ✅ Embedding generated successfully!")
            print(f"   Embedding length: {len(embedding)}")
            print(f"   Embedding type: {type(embedding)}")
            print(f"   First 5 values: {embedding[:5]}")
            print(f"   Last 5 values: {embedding[-5:]}")
            print(f"   Min value: {min(embedding):.6f}")
            print(f"   Max value: {max(embedding):.6f}")
            print(f"   Mean value: {sum(embedding)/len(embedding):.6f}")
            
            # Validate embedding format
            if len(embedding) == 1536:
                print(f"   ✅ Embedding dimension is correct (1536)")
            else:
                print(f"   ⚠️  Unexpected embedding dimension: {len(embedding)} (expected 1536)")
            
            if all(isinstance(x, (int, float)) for x in embedding):
                print(f"   ✅ All embedding values are numeric")
            else:
                print(f"   ⚠️  Some embedding values are not numeric")
                
        except Exception as e:
            print(f"   ❌ Error generating embedding: {e}")
            print(f"   Error type: {type(e)}")
            continue
    
    # Test with a specific security document text
    print(f"\n🔍 Test with Security Document Text:")
    security_text = """
    Cyber & Information Security Policy
    
    This policy establishes the framework for protecting the organization's information assets 
    and technology infrastructure from cyber threats and security breaches. All employees 
    must adhere to these security standards to maintain the confidentiality, integrity, 
    and availability of our systems and data.
    
    Key security principles include:
    - Regular security assessments and vulnerability management
    - Access control and identity management
    - Data protection and encryption
    - Incident response and business continuity
    - Security awareness and training programs
    """
    
    print(f"   Text length: {len(security_text)} characters")
    print(f"   Text preview: {security_text[:100]}...")
    
    try:
        print(f"   🧠 Generating embedding for security text...")
        response = client.embeddings.create(
            input=security_text,
            model=azure_openai_deployment
        )
        
        embedding = response.data[0].embedding
        print(f"   ✅ Security text embedding generated successfully!")
        print(f"   Embedding length: {len(embedding)}")
        print(f"   First 10 values: {embedding[:10]}")
        print(f"   Last 10 values: {embedding[-10:]}")
        
    except Exception as e:
        print(f"   ❌ Error generating security text embedding: {e}")
    
    print(f"\n🎉 Azure OpenAI embedding test completed!")
    return True

def test_embedding_similarity():
    """Test embedding similarity between different texts"""
    
    if not AZURE_OPENAI_AVAILABLE:
        return False
    
    azure_openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    azure_openai_key = os.getenv('AZURE_OPENAI_KEY')
    
    # Use embedding model if available, otherwise fall back to deployment
    azure_text_embedding_model = os.getenv('AZURE_TEXT_EMBEDDING_MODEL')
    azure_openai_deployment = azure_text_embedding_model or os.getenv('AZURE_OPENAI_DEPLOYMENT')
    
    if not all([azure_openai_endpoint, azure_openai_key, azure_openai_deployment]):
        return False
    
    try:
        client = AzureOpenAI(
            azure_endpoint=azure_openai_endpoint,
            api_key=azure_openai_key,
            api_version="2024-02-15-preview"
        )
        
        print(f"\n🔍 Testing Embedding Similarity:")
        print("=" * 40)
        
        # Test texts with different similarities
        texts = [
            "Cyber security policy and information protection",
            "Information security policy and cyber protection",  # Similar meaning
            "The weather is sunny today",  # Different topic
            "Cybersecurity policy and data protection",  # Similar topic
            "Cooking recipes for Italian food"  # Completely different
        ]
        
        embeddings = []
        
        # Generate embeddings
        for i, text in enumerate(texts):
            print(f"   Generating embedding {i+1}/{len(texts)}: '{text[:30]}...'")
            response = client.embeddings.create(input=text, model=azure_openai_deployment)
            embeddings.append(response.data[0].embedding)
        
        # Calculate cosine similarities
        import math
        
        def cosine_similarity(a, b):
            dot_product = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot_product / (norm_a * norm_b)
        
        print(f"\n📊 Similarity Matrix:")
        print("   " + "".join(f"{i+1:>8}" for i in range(len(texts))))
        
        for i, emb1 in enumerate(embeddings):
            row = f"{i+1:>2} "
            for j, emb2 in enumerate(embeddings):
                similarity = cosine_similarity(emb1, emb2)
                row += f"{similarity:>8.3f}"
            print(row)
        
        print(f"\n📝 Text Descriptions:")
        for i, text in enumerate(texts, 1):
            print(f"   {i}. {text}")
        
    except Exception as e:
        print(f"   ❌ Error in similarity test: {e}")

if __name__ == "__main__":
    print("🚀 Starting Azure OpenAI Embedding Trial")
    print("=" * 60)
    
    success = test_azure_openai_embeddings()
    
    if success:
        test_embedding_similarity()
    
    print(f"\n🏁 Trial completed!")
    if success:
        print("✅ Azure OpenAI embeddings are working correctly!")
    else:
        print("❌ Azure OpenAI embeddings test failed!")
        print("Please check your configuration and try again.") 