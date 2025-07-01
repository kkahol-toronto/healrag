"""
HEALRAG - Azure RAG Library

A comprehensive library for building RAG (Retrieval-Augmented Generation) applications
on Azure with support for Azure Blob Storage and MarkItDown document processing.
"""

__version__ = "0.1.0"
__author__ = "HEALRAG Team"

from .storage_manager import StorageManager

__all__ = ["StorageManager"] 