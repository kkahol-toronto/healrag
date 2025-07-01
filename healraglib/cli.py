#!/usr/bin/env python3
"""
Command Line Interface for HEALRAG library
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .storage_manager import StorageManager


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def verify_connection_cmd(args):
    """Verify connection to Azure Blob Storage."""
    try:
        storage_manager = StorageManager(args.connection_string, args.container)
        if storage_manager.verify_connection():
            print("✅ Successfully connected to Azure Blob Storage!")
            return 0
        else:
            print("❌ Failed to connect to Azure Blob Storage")
            return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def statistics_cmd(args):
    """Get container statistics."""
    try:
        storage_manager = StorageManager(args.connection_string, args.container)
        stats = storage_manager.get_container_statistics()
        
        if args.json:
            print(json.dumps(stats, indent=2, default=str))
        else:
            print(f"📊 Container: {stats['container_name']}")
            print(f"📁 Total files: {stats['total_files']}")
            print(f"💾 Total size: {stats['total_size_mb']} MB ({stats['total_size_gb']} GB)")
            
            if stats['file_types']:
                print("📋 File types:")
                for ext, data in stats['file_types'].items():
                    print(f"   {ext}: {data['count']} files ({data['total_size_mb']} MB)")
        
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def list_files_cmd(args):
    """List files in container."""
    try:
        storage_manager = StorageManager(args.connection_string, args.container)
        file_list = storage_manager.get_file_list(as_json=args.json)
        
        if args.json:
            print(file_list)
        else:
            files = json.loads(file_list) if isinstance(file_list, str) else file_list
            print(f"📄 Found {len(files)} files in container:")
            for i, file_name in enumerate(files, 1):
                print(f"   {i}. {file_name}")
        
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def upload_cmd(args):
    """Upload file or folder."""
    try:
        storage_manager = StorageManager(args.connection_string, args.container)
        
        if os.path.isfile(args.source):
            # Upload single file
            success = storage_manager.upload_file(args.source, args.destination)
            if success:
                print(f"✅ Successfully uploaded {args.source}")
                return 0
            else:
                print(f"❌ Failed to upload {args.source}")
                return 1
        elif os.path.isdir(args.source):
            # Upload folder
            results = storage_manager.upload_folder(args.source, prefix=args.destination)
            successful = sum(results.values())
            total = len(results)
            print(f"📤 Uploaded {successful}/{total} files successfully")
            return 0 if successful == total else 1
        else:
            print(f"❌ Source path does not exist: {args.source}")
            return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def download_cmd(args):
    """Download file or folder."""
    try:
        storage_manager = StorageManager(args.connection_string, args.container)
        
        if args.entire_container:
            # Download entire container
            results = storage_manager.download_entire_container(args.destination)
            successful = sum(results.values())
            total = len(results)
            print(f"📥 Downloaded {successful}/{total} files successfully")
            return 0 if successful == total else 1
        elif args.prefix:
            # Download folder by prefix
            results = storage_manager.download_folder(args.prefix, args.destination)
            successful = sum(results.values())
            total = len(results)
            print(f"📥 Downloaded {successful}/{total} files successfully")
            return 0 if successful == total else 1
        else:
            # Download single file
            success = storage_manager.download_file(args.source, args.destination)
            if success:
                print(f"✅ Successfully downloaded {args.source}")
                return 0
            else:
                print(f"❌ Failed to download {args.source}")
                return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def extract_cmd(args):
    """Extract content from file using MarkItDown."""
    try:
        storage_manager = StorageManager(args.connection_string, args.container)
        content = storage_manager.get_file_content_with_markitdown(args.file)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Content extracted and saved to {args.output}")
        else:
            print(f"📄 Extracted content ({len(content)} characters):")
            print("-" * 50)
            print(content)
        
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="HEALRAG - Azure RAG Library CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify connection
  healrag verify --connection-string "your_connection_string" --container "your_container"
  
  # Get statistics
  healrag stats --connection-string "your_connection_string" --container "your_container"
  
  # List files
  healrag list --connection-string "your_connection_string" --container "your_container"
  
  # Upload a file
  healrag upload --connection-string "your_connection_string" --container "your_container" --source "file.pdf" --destination "remote_file.pdf"
  
  # Download entire container
  healrag download --connection-string "your_connection_string" --container "your_container" --destination "local_folder" --entire-container
  
  # Extract content
  healrag extract --connection-string "your_connection_string" --container "your_container" --file "document.pdf" --output "content.txt"
        """
    )
    
    parser.add_argument(
        "--connection-string",
        default=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
        help="Azure Storage connection string (or set AZURE_STORAGE_CONNECTION_STRING env var)"
    )
    parser.add_argument(
        "--container",
        default=os.getenv("AZURE_CONTAINER_NAME", "healrag-documents"),
        help="Container name (or set AZURE_CONTAINER_NAME env var)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify connection to Azure Blob Storage")
    verify_parser.set_defaults(func=verify_connection_cmd)
    
    # Statistics command
    stats_parser = subparsers.add_parser("stats", help="Get container statistics")
    stats_parser.add_argument("--json", action="store_true", help="Output as JSON")
    stats_parser.set_defaults(func=statistics_cmd)
    
    # List files command
    list_parser = subparsers.add_parser("list", help="List files in container")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")
    list_parser.set_defaults(func=list_files_cmd)
    
    # Upload command
    upload_parser = subparsers.add_parser("upload", help="Upload file or folder")
    upload_parser.add_argument("--source", required=True, help="Local file or folder path")
    upload_parser.add_argument("--destination", help="Remote blob name or prefix (optional)")
    upload_parser.set_defaults(func=upload_cmd)
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download file or folder")
    download_parser.add_argument("--source", help="Remote blob name")
    download_parser.add_argument("--destination", required=True, help="Local file or folder path")
    download_parser.add_argument("--prefix", help="Download files with this prefix")
    download_parser.add_argument("--entire-container", action="store_true", help="Download entire container")
    download_parser.set_defaults(func=download_cmd)
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract content from file using MarkItDown")
    extract_parser.add_argument("--file", required=True, help="Remote blob name")
    extract_parser.add_argument("--output", help="Output file path (optional)")
    extract_parser.set_defaults(func=extract_cmd)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if not args.connection_string:
        print("❌ Error: Connection string is required. Set --connection-string or AZURE_STORAGE_CONNECTION_STRING environment variable.")
        return 1
    
    setup_logging(args.verbose)
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main()) 