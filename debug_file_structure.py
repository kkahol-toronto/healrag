#!/usr/bin/env python3
"""
Debug script to understand file structure and content extraction issues
"""

import os
from dotenv import load_dotenv
from healraglib import StorageManager, FileGraphManager

# Load environment variables
load_dotenv()

def debug_file_structure():
    """Debug the file structure and content extraction."""
    
    print("🔍 Debugging File Structure and Content Extraction")
    print("=" * 60)
    
    # Initialize storage manager
    storage_manager = StorageManager(
        connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
        container_name=os.getenv("AZURE_CONTAINER_NAME", "healrag-documents")
    )
    
    # Get all files
    print("\n📁 Getting all files from container...")
    try:
        files = storage_manager.get_file_list(as_json=False)
        print(f"   ✅ Found {len(files)} files in container")
        
        # Show file types
        file_types = {}
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            file_types[ext] = file_types.get(ext, 0) + 1
        
        print(f"   📊 File type distribution: {file_types}")
        
        # Show first 10 files
        print("\n📄 First 10 files:")
        for i, file in enumerate(files[:10]):
            print(f"   {i+1}. {file}")
        
        # Test content extraction for different file types
        print("\n🔍 Testing content extraction...")
        
        # Test C# files
        cs_files = [f for f in files if f.endswith('.cs')]
        if cs_files:
            print(f"\n   C# files found: {len(cs_files)}")
            test_cs_file = cs_files[0]
            print(f"   Testing: {test_cs_file}")
            
            try:
                content = storage_manager.get_file_content_with_markitdown(test_cs_file)
                if content:
                    print(f"   ✅ Content extracted: {len(content)} characters")
                    print(f"   📄 Preview: {content[:300]}...")
                else:
                    print(f"   ❌ No content extracted")
            except Exception as e:
                print(f"   ❌ Error extracting content: {e}")
        
        # Test markdown files
        md_files = [f for f in files if f.endswith('.md')]
        if md_files:
            print(f"\n   Markdown files found: {len(md_files)}")
            test_md_file = md_files[0]
            print(f"   Testing: {test_md_file}")
            
            try:
                content = storage_manager.get_file_content_with_markitdown(test_md_file)
                if content:
                    print(f"   ✅ Content extracted: {len(content)} characters")
                    print(f"   📄 Preview: {content[:300]}...")
                else:
                    print(f"   ❌ No content extracted")
            except Exception as e:
                print(f"   ❌ Error extracting content: {e}")
        
        # Test md_files folder
        md_folder_files = [f for f in files if f.startswith('md_files/')]
        if md_folder_files:
            print(f"\n   md_files folder contains: {len(md_folder_files)} files")
            test_md_folder_file = md_folder_files[0]
            print(f"   Testing: {test_md_folder_file}")
            
            try:
                content = storage_manager.get_file_content_with_markitdown(test_md_folder_file)
                if content:
                    print(f"   ✅ Content extracted: {len(content)} characters")
                    print(f"   📄 Preview: {content[:300]}...")
                else:
                    print(f"   ❌ No content extracted")
            except Exception as e:
                print(f"   ❌ Error extracting content: {e}")
        
        # Test direct blob access
        print(f"\n🔍 Testing direct blob access...")
        if files:
            test_file = files[0]
            print(f"   Testing: {test_file}")
            
            try:
                from azure.storage.blob import BlobServiceClient
                blob_service_client = BlobServiceClient.from_connection_string(
                    storage_manager.connection_string
                )
                blob_client = blob_service_client.get_blob_client(
                    container=storage_manager.container_name,
                    blob=test_file
                )
                blob_data = blob_client.download_blob()
                content = blob_data.readall().decode('utf-8', errors='ignore')
                print(f"   ✅ Direct blob access: {len(content)} characters")
                print(f"   📄 Preview: {content[:300]}...")
            except Exception as e:
                print(f"   ❌ Direct blob access failed: {e}")
        
    except Exception as e:
        print(f"   ❌ Error getting file list: {e}")

def debug_dependency_detection():
    """Debug dependency detection patterns."""
    
    print("\n🔗 Debugging Dependency Detection Patterns")
    print("=" * 60)
    
    # Test C# patterns
    test_csharp_content = """
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.DependencyInjection;

namespace MyProject
{
    public class MyClass
    {
        private readonly ILogger<MyClass> _logger;
        
        public MyClass(ILogger<MyClass> logger)
        {
            _logger = logger;
        }
        
        public void DoSomething()
        {
            var list = new List<string>();
            var result = list.Where(x => x.Length > 0);
        }
    }
    
    public interface IMyInterface
    {
        void DoWork();
    }
}
"""
    
    print("📄 Testing C# dependency detection...")
    print("   Test content:")
    print("   " + "\n   ".join(test_csharp_content.split('\n')[:10]) + "...")
    
    # Initialize file graph manager
    file_graph_manager = FileGraphManager(
        storage_manager=None,
        search_manager=None,
        azure_openai_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_openai_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_openai_deployment=os.getenv("AZURE_TEXT_EMBEDDING_MODEL")
    )
    
    dependencies = file_graph_manager.detect_dependencies(test_csharp_content, 'csharp')
    
    print(f"\n   ✅ Dependencies detected:")
    print(f"   📥 Imports: {dependencies['imports']}")
    print(f"   🔗 References: {dependencies['references']}")
    print(f"   🏗️  Classes: {dependencies['classes']}")
    print(f"   ⚙️  Functions: {dependencies['functions']}")

def main():
    """Main function."""
    print("HEALRAG File Structure Debug")
    print("=" * 40)
    
    debug_file_structure()
    debug_dependency_detection()
    
    print("\n🎉 Debug completed!")
    print("\n📖 Next steps:")
    print("   1. Check if files are accessible")
    print("   2. Verify content extraction is working")
    print("   3. Test with different similarity thresholds")
    print("   4. Run the main test script again")

if __name__ == "__main__":
    main() 