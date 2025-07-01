"""
Azure Blob Storage Manager for HEALRAG

Provides comprehensive functionality for managing files in Azure Blob Storage
including connection verification, statistics, file operations, and bulk operations.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import mimetypes
from datetime import datetime

from azure.storage.blob import BlobServiceClient, ContainerClient
from azure.core.exceptions import ResourceNotFoundError, ClientAuthenticationError


class StorageManager:
    """
    Azure Blob Storage Manager for HEALRAG library.
    
    Provides functionality for managing files in Azure Blob Storage containers
    with support for file operations, statistics, and bulk operations.
    """
    
    def __init__(self, connection_string: str, container_name: str):
        """
        Initialize the Storage Manager.
        
        Args:
            connection_string (str): Azure Storage connection string
            container_name (str): Name of the blob container
        """
        self.connection_string = connection_string
        self.container_name = container_name
        self.blob_service_client = None
        self.container_client = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize the connection
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize Azure Blob Storage clients."""
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(
                self.connection_string
            )
            self.container_client = self.blob_service_client.get_container_client(
                self.container_name
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure clients: {e}")
            raise
    
    def verify_connection(self) -> bool:
        """
        Verify the connection to Azure Blob Storage.
        
        Returns:
            bool: True if connected successfully, False otherwise
        """
        try:
            # Try to get container properties to verify connection
            self.container_client.get_container_properties()
            self.logger.info("Successfully connected to Azure Blob Storage")
            return True
        except (ResourceNotFoundError, ClientAuthenticationError) as e:
            self.logger.error(f"Connection verification failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during connection verification: {e}")
            return False
    
    def get_container_statistics(self) -> Dict:
        """
        Get comprehensive statistics about the container.
        
        Returns:
            Dict: Statistics including file counts, types, and sizes
        """
        if not self.verify_connection():
            raise ConnectionError("Cannot connect to Azure Blob Storage")
        
        try:
            stats = {
                "total_files": 0,
                "total_size_bytes": 0,
                "file_types": {},
                "file_sizes": {},
                "last_modified": None,
                "container_name": self.container_name
            }
            
            # Iterate through all blobs in the container
            blobs = self.container_client.list_blobs()
            
            for blob in blobs:
                stats["total_files"] += 1
                stats["total_size_bytes"] += blob.size
                
                # Get file extension and type
                file_name = blob.name
                file_extension = Path(file_name).suffix.lower()
                if not file_extension:
                    file_extension = "no_extension"
                
                # Update file type statistics
                if file_extension not in stats["file_types"]:
                    stats["file_types"][file_extension] = {
                        "count": 0,
                        "total_size_bytes": 0
                    }
                stats["file_types"][file_extension]["count"] += 1
                stats["file_types"][file_extension]["total_size_bytes"] += blob.size
                
                # Store individual file size
                stats["file_sizes"][file_name] = blob.size
                
                # Track last modified time
                if blob.last_modified:
                    if not stats["last_modified"] or blob.last_modified > stats["last_modified"]:
                        stats["last_modified"] = blob.last_modified
            
            # Convert bytes to human readable format
            stats["total_size_mb"] = round(stats["total_size_bytes"] / (1024 * 1024), 2)
            stats["total_size_gb"] = round(stats["total_size_bytes"] / (1024 * 1024 * 1024), 2)
            
            # Add human readable sizes to file types
            for ext, data in stats["file_types"].items():
                data["total_size_mb"] = round(data["total_size_bytes"] / (1024 * 1024), 2)
                data["total_size_gb"] = round(data["total_size_bytes"] / (1024 * 1024 * 1024), 2)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting container statistics: {e}")
            raise
    
    def get_file_list(self, as_json: bool = True) -> Union[List[str], str]:
        """
        Get list of all files in the container.
        
        Args:
            as_json (bool): If True, return as JSON string, else return as list
            
        Returns:
            Union[List[str], str]: List of filenames or JSON string
        """
        if not self.verify_connection():
            raise ConnectionError("Cannot connect to Azure Blob Storage")
        
        try:
            file_list = []
            blobs = self.container_client.list_blobs()
            
            for blob in blobs:
                file_list.append(blob.name)
            
            if as_json:
                return json.dumps(file_list, indent=2)
            else:
                return file_list
                
        except Exception as e:
            self.logger.error(f"Error getting file list: {e}")
            raise
    
    def upload_file(self, local_file_path: str, blob_name: Optional[str] = None) -> bool:
        """
        Upload a single file to the container.
        
        Args:
            local_file_path (str): Path to the local file
            blob_name (str, optional): Name for the blob in container. If None, uses filename
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        if not self.verify_connection():
            raise ConnectionError("Cannot connect to Azure Blob Storage")
        
        if not os.path.exists(local_file_path):
            raise FileNotFoundError(f"Local file not found: {local_file_path}")
        
        try:
            if blob_name is None:
                blob_name = os.path.basename(local_file_path)
            
            blob_client = self.container_client.get_blob_client(blob_name)
            
            with open(local_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            
            self.logger.info(f"Successfully uploaded {local_file_path} as {blob_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error uploading file {local_file_path}: {e}")
            return False
    
    def upload_folder(self, local_folder_path: str, prefix: str = "") -> Dict[str, bool]:
        """
        Upload all files from a local folder to the container.
        
        Args:
            local_folder_path (str): Path to the local folder
            prefix (str): Optional prefix for blob names
            
        Returns:
            Dict[str, bool]: Dictionary mapping filenames to upload success status
        """
        if not self.verify_connection():
            raise ConnectionError("Cannot connect to Azure Blob Storage")
        
        if not os.path.exists(local_folder_path):
            raise FileNotFoundError(f"Local folder not found: {local_folder_path}")
        
        results = {}
        folder_path = Path(local_folder_path)
        
        try:
            # Walk through all files in the folder
            for file_path in folder_path.rglob("*"):
                if file_path.is_file():
                    # Calculate relative path for blob name
                    relative_path = file_path.relative_to(folder_path)
                    blob_name = str(relative_path)
                    
                    if prefix:
                        blob_name = f"{prefix}/{blob_name}"
                    
                    # Upload the file
                    success = self.upload_file(str(file_path), blob_name)
                    results[str(file_path)] = success
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error uploading folder {local_folder_path}: {e}")
            raise
    
    def download_file(self, blob_name: str, local_file_path: str) -> bool:
        """
        Download a single file from the container.
        
        Args:
            blob_name (str): Name of the blob in the container
            local_file_path (str): Local path where to save the file
            
        Returns:
            bool: True if download successful, False otherwise
        """
        if not self.verify_connection():
            raise ConnectionError("Cannot connect to Azure Blob Storage")
        
        try:
            blob_client = self.container_client.get_blob_client(blob_name)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            with open(local_file_path, "wb") as download_file:
                download_stream = blob_client.download_blob()
                download_file.write(download_stream.readall())
            
            self.logger.info(f"Successfully downloaded {blob_name} to {local_file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading file {blob_name}: {e}")
            return False
    
    def download_folder(self, blob_prefix: str, local_folder_path: str) -> Dict[str, bool]:
        """
        Download all files with a specific prefix from the container.
        
        Args:
            blob_prefix (str): Prefix of blobs to download
            local_folder_path (str): Local folder where to save the files
            
        Returns:
            Dict[str, bool]: Dictionary mapping blob names to download success status
        """
        if not self.verify_connection():
            raise ConnectionError("Cannot connect to Azure Blob Storage")
        
        results = {}
        
        try:
            # Create local folder if it doesn't exist
            os.makedirs(local_folder_path, exist_ok=True)
            
            # List blobs with the specified prefix
            blobs = self.container_client.list_blobs(name_starts_with=blob_prefix)
            
            for blob in blobs:
                # Calculate local file path
                relative_path = blob.name[len(blob_prefix):].lstrip('/')
                local_file_path = os.path.join(local_folder_path, relative_path)
                
                # Download the file
                success = self.download_file(blob.name, local_file_path)
                results[blob.name] = success
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error downloading folder with prefix {blob_prefix}: {e}")
            raise
    
    def download_entire_container(self, local_folder_path: str) -> Dict[str, bool]:
        """
        Download all files from the entire container.
        
        Args:
            local_folder_path (str): Local folder where to save all files
            
        Returns:
            Dict[str, bool]: Dictionary mapping blob names to download success status
        """
        if not self.verify_connection():
            raise ConnectionError("Cannot connect to Azure Blob Storage")
        
        results = {}
        
        try:
            # Create local folder if it doesn't exist
            os.makedirs(local_folder_path, exist_ok=True)
            
            # List all blobs in the container
            blobs = self.container_client.list_blobs()
            
            for blob in blobs:
                # Calculate local file path
                local_file_path = os.path.join(local_folder_path, blob.name)
                
                # Download the file
                success = self.download_file(blob.name, local_file_path)
                results[blob.name] = success
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error downloading entire container: {e}")
            raise 