#!/usr/bin/env python3
"""
Debug script to test markitdown functionality
"""

import os
import tempfile

def test_markitdown_basic():
    """Test basic markitdown functionality."""
    try:
        from markitdown import MarkItDown
        print("✅ MarkItDown import successful")
        
        # Create a simple test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document for MarkItDown.\nIt has multiple lines.\nAnd some content.")
            test_file = f.name
        
        try:
            print(f"📄 Test file created: {test_file}")
            
            # Test MarkItDown conversion
            md = MarkItDown()
            print("🔧 MarkItDown instance created")
            
            result = md.convert(test_file)
            print(f"📊 Result type: {type(result)}")
            print(f"📊 Result: {result}")
            
            if result is not None:
                print(f"📊 Result attributes: {dir(result)}")
                
                if hasattr(result, 'text_content'):
                    print(f"✅ text_content available: {len(result.text_content)} characters")
                    print(f"📄 Preview: {result.text_content[:100]}...")
                elif hasattr(result, 'text'):
                    print(f"✅ text available: {len(result.text)} characters")
                    print(f"📄 Preview: {result.text[:100]}...")
                else:
                    print(f"⚠️  No text_content or text attribute found")
                    print(f"📄 String representation: {str(result)[:100]}...")
            else:
                print("❌ Result is None")
                
        finally:
            # Clean up test file
            if os.path.exists(test_file):
                os.unlink(test_file)
                print(f"🗑️  Test file cleaned up")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_markitdown_pdf():
    """Test markitdown with the specific PDF file."""
    print(f"\n🔍 Current working directory: {os.getcwd()}")
    
    # Check available paths
    possible_paths = [
        "../security_document",
        "../../security_document", 
        "security_document",
        "../../../security_document"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Found security_document at: {path}")
            test_pdf = os.path.join(path, "C&IS 001_Cyber & Information Security Policy and Standard Framework.pdf")
            if os.path.exists(test_pdf):
                print(f"✅ Found PDF file: {test_pdf}")
                break
        else:
            print(f"❌ Not found: {path}")
    else:
        print("❌ Could not find security_document folder or PDF file")
        return
    
    if os.path.exists(test_pdf):
        print(f"\n📄 Testing with specific PDF: {test_pdf}")
        
        try:
            from markitdown import MarkItDown
            md = MarkItDown()
            result = md.convert(test_pdf)
            
            print(f"📊 PDF Result type: {type(result)}")
            print(f"📊 PDF Result: {result}")
            
            if result is not None:
                if hasattr(result, 'text_content'):
                    print(f"✅ PDF text_content: {len(result.text_content)} characters")
                    print(f"📄 Preview: {result.text_content[:200]}...")
                else:
                    print(f"⚠️  PDF result has no text_content attribute")
            else:
                print("❌ PDF result is None")
                
        except Exception as e:
            print(f"❌ PDF Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n⚠️  Specific PDF not found: {test_pdf}")
        print("   Please ensure the file exists in the security_document folder")

if __name__ == "__main__":
    print("🔍 Debugging MarkItDown")
    print("=" * 50)
    
    test_markitdown_basic()
    test_markitdown_pdf()
    
    print("\n" + "=" * 50)
    print("�� Debug complete") 