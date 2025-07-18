"""
File Graph Manager for HEALRAG

Handles file similarity analysis, dependency detection, and graph visualization
for understanding relationships between files in the indexed content.
"""

import os
import json
import logging
import re
import math
from typing import List, Dict, Optional, Tuple, Set, Any
from pathlib import Path
from datetime import datetime
import time
from collections import defaultdict, Counter

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False

try:
    from azure.search.documents import SearchClient
    from azure.core.credentials import AzureKeyCredential
    AZURE_SEARCH_AVAILABLE = True
except ImportError:
    AZURE_SEARCH_AVAILABLE = False
    SearchClient = None
    AzureKeyCredential = None


class FileGraphManager:
    """
    File Graph Manager for HEALRAG library.
    
    Analyzes file similarities and dependencies to create graph visualizations
    showing relationships between files in the indexed content.
    """
    
    def __init__(self, 
                 storage_manager,
                 search_manager,
                 azure_openai_endpoint: Optional[str] = None,
                 azure_openai_key: Optional[str] = None,
                 azure_openai_deployment: Optional[str] = None,
                 azure_search_endpoint: Optional[str] = None,
                 azure_search_key: Optional[str] = None,
                 azure_search_index_name: str = "healrag-index"):
        """
        Initialize the File Graph Manager.
        
        Args:
            storage_manager: HEALRAG StorageManager instance
            search_manager: HEALRAG SearchIndexManager instance
            azure_openai_endpoint: Azure OpenAI endpoint
            azure_openai_key: Azure OpenAI API key
            azure_openai_deployment: Azure OpenAI embedding deployment name
            azure_search_endpoint: Azure Cognitive Search endpoint
            azure_search_key: Azure Cognitive Search API key
            azure_search_index_name: Name of the search index
        """
        self.storage_manager = storage_manager
        self.search_manager = search_manager
        self.azure_openai_endpoint = azure_openai_endpoint
        self.azure_openai_key = azure_openai_key
        self.azure_openai_deployment = azure_openai_deployment
        self.azure_search_endpoint = azure_search_endpoint
        self.azure_search_key = azure_search_key
        self.azure_search_index_name = azure_search_index_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize Azure OpenAI client for embeddings
        self.openai_client = None
        if all([azure_openai_endpoint, azure_openai_key, azure_openai_deployment]):
            self._initialize_openai_client()
        
        # Initialize Azure Search client
        self.search_client = None
        if all([azure_search_endpoint, azure_search_key]):
            self._initialize_search_client()
        
        # Cache for file embeddings and relationships
        self.file_embeddings_cache = {}
        self.file_relationships_cache = {}
        self.graph_data_cache = None
        
        # File type patterns for dependency detection
        self.dependency_patterns = {
            'python': {
                'imports': [
                    r'^import\s+(\w+)',
                    r'^from\s+(\w+)\s+import',
                    r'^from\s+(\w+\.\w+)\s+import'
                ],
                'references': [
                    r'(\w+)\.py',
                    r'from\s+(\w+)',
                    r'import\s+(\w+)'
                ]
            },
            'javascript': {
                'imports': [
                    r'^import\s+.*?from\s+[\'"]([^\'"]+)[\'"]',
                    r'^const\s+\w+\s*=\s*require\s*\(\s*[\'"]([^\'"]+)[\'"]',
                    r'^import\s*\(\s*[\'"]([^\'"]+)[\'"]'
                ],
                'references': [
                    r'[\'"]([^\'"]+\.js)[\'"]',
                    r'[\'"]([^\'"]+\.ts)[\'"]',
                    r'[\'"]([^\'"]+\.jsx)[\'"]',
                    r'[\'"]([^\'"]+\.tsx)[\'"]'
                ]
            },
            'typescript': {
                'imports': [
                    r'^import\s+.*?from\s+[\'"]([^\'"]+)[\'"]',
                    r'^import\s*\(\s*[\'"]([^\'"]+)[\'"]'
                ],
                'references': [
                    r'[\'"]([^\'"]+\.ts)[\'"]',
                    r'[\'"]([^\'"]+\.js)[\'"]',
                    r'[\'"]([^\'"]+\.tsx)[\'"]',
                    r'[\'"]([^\'"]+\.jsx)[\'"]'
                ]
            },
            'java': {
                'imports': [
                    r'^import\s+([\w.]+)',
                    r'^import\s+static\s+([\w.]+)'
                ],
                'references': [
                    r'(\w+)\.java',
                    r'import\s+([\w.]+)'
                ]
            },
            'cpp': {
                'imports': [
                    r'^#include\s*[<"]([^>"]+)[>"]',
                    r'^#include\s+([\w./]+)'
                ],
                'references': [
                    r'[\'"]([^\'"]+\.h)[\'"]',
                    r'[\'"]([^\'"]+\.cpp)[\'"]',
                    r'[\'"]([^\'"]+\.hpp)[\'"]'
                ]
            },
            'csharp': {
                'imports': [
                    r'^using\s+([\w.]+)',
                    r'^using\s+static\s+([\w.]+)',
                    r'^using\s+([\w.]+)\s*;',
                    r'^using\s+static\s+([\w.]+)\s*;'
                ],
                'references': [
                    r'(\w+)\.cs',
                    r'using\s+([\w.]+)',
                    r'class\s+(\w+)',
                    r'interface\s+(\w+)',
                    r'namespace\s+(\w+)',
                    r'public\s+class\s+(\w+)',
                    r'public\s+interface\s+(\w+)',
                    r'private\s+class\s+(\w+)',
                    r'protected\s+class\s+(\w+)'
                ]
            }
        }
    
    def _initialize_openai_client(self):
        """Initialize Azure OpenAI client for embeddings."""
        try:
            self.openai_client = AzureOpenAI(
                azure_endpoint=self.azure_openai_endpoint,
                api_key=self.azure_openai_key,
                api_version="2024-02-15-preview"
            )
            self.logger.info("Azure OpenAI client initialized for file graph analysis")
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure OpenAI client: {e}")
    
    def _initialize_search_client(self):
        """Initialize Azure Search client."""
        try:
            credential = AzureKeyCredential(self.azure_search_key)
            self.search_client = SearchClient(
                endpoint=self.azure_search_endpoint,
                index_name=self.azure_search_index_name,
                credential=credential
            )
            self.logger.info("Azure Search client initialized for file graph analysis")
        except Exception as e:
            self.logger.error(f"Failed to initialize Azure Search client: {e}")
    
    def get_file_type(self, filename: str) -> str:
        """Determine the file type based on extension."""
        ext = Path(filename).suffix.lower()
        type_mapping = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.cxx': 'cpp',
            '.cc': 'cpp',
            '.h': 'cpp',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.swift': 'swift',
            '.php': 'php',
            '.rb': 'ruby',
            '.pl': 'perl',
            '.scala': 'scala',
            '.sh': 'shell',
            '.bash': 'shell',
            '.zsh': 'shell',
            '.r': 'r',
            '.R': 'r',
            '.md': 'markdown',
            '.html': 'html',
            '.htm': 'html',
            '.css': 'css',
            '.json': 'json',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.xml': 'xml',
            '.toml': 'toml',
            '.txt': 'text'
        }
        return type_mapping.get(ext, 'unknown')
    
    def extract_file_content(self, filename: str) -> Optional[str]:
        """Extract content from a file in the storage container."""
        try:
            # Try to get content from markdown files first
            md_filename = f"md_files/{filename}"
            if self.storage_manager:
                try:
                    content = self.storage_manager.get_file_content_with_markitdown(md_filename)
                    if content and len(content.strip()) > 0:
                        return content
                except Exception as e:
                    self.logger.debug(f"Could not extract from md_files/{filename}: {e}")
            
            # Fallback to original file
            if self.storage_manager:
                try:
                    content = self.storage_manager.get_file_content_with_markitdown(filename)
                    if content and len(content.strip()) > 0:
                        return content
                except Exception as e:
                    self.logger.debug(f"Could not extract from {filename}: {e}")
            
            # Try direct blob download as last resort
            if self.storage_manager:
                try:
                    from azure.storage.blob import BlobServiceClient
                    blob_service_client = BlobServiceClient.from_connection_string(
                        self.storage_manager.connection_string
                    )
                    blob_client = blob_service_client.get_blob_client(
                        container=self.storage_manager.container_name,
                        blob=filename
                    )
                    blob_data = blob_client.download_blob()
                    content = blob_data.readall().decode('utf-8', errors='ignore')
                    if content and len(content.strip()) > 0:
                        return content
                except Exception as e:
                    self.logger.debug(f"Could not download blob {filename}: {e}")
            
            return None
        except Exception as e:
            self.logger.error(f"Error extracting content from {filename}: {e}")
            return None
    
    def detect_dependencies(self, content: str, file_type: str) -> Dict[str, List[str]]:
        """Detect dependencies in file content based on file type."""
        dependencies = {
            'imports': [],
            'references': [],
            'functions': [],
            'classes': []
        }
        
        if not content:
            return dependencies
        
        patterns = self.dependency_patterns.get(file_type, {})
        
        # Extract imports
        for pattern in patterns.get('imports', []):
            matches = re.findall(pattern, content, re.MULTILINE)
            dependencies['imports'].extend(matches)
        
        # Extract references
        for pattern in patterns.get('references', []):
            matches = re.findall(pattern, content, re.MULTILINE)
            dependencies['references'].extend(matches)
        
        # Extract function definitions
        if file_type == 'python':
            function_matches = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
            dependencies['functions'].extend(function_matches)
        elif file_type in ['javascript', 'typescript']:
            function_matches = re.findall(r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*\(|let\s+(\w+)\s*=\s*\()', content, re.MULTILINE)
            for match in function_matches:
                dependencies['functions'].extend([f for f in match if f])
        elif file_type == 'java':
            function_matches = re.findall(r'(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(?:[\w<>[\],\s]+)\s+(\w+)\s*\(', content, re.MULTILINE)
            dependencies['functions'].extend(function_matches)
        
        # Extract class definitions
        class_patterns = {
            'python': r'^class\s+(\w+)',
            'javascript': r'^class\s+(\w+)',
            'typescript': r'^class\s+(\w+)',
            'java': r'^class\s+(\w+)',
            'cpp': r'^class\s+(\w+)',
            'csharp': r'^class\s+(\w+)'
        }
        
        if file_type in class_patterns:
            class_matches = re.findall(class_patterns[file_type], content, re.MULTILINE)
            dependencies['classes'].extend(class_matches)
        
        # Remove duplicates and empty strings
        for key in dependencies:
            dependencies[key] = list(set([item for item in dependencies[key] if item.strip()]))
        
        return dependencies
    
    def get_file_embedding(self, filename: str) -> Optional[List[float]]:
        """Get embedding for a file, using cache if available."""
        if filename in self.file_embeddings_cache:
            return self.file_embeddings_cache[filename]
        
        if not self.openai_client:
            return None
        
        try:
            content = self.extract_file_content(filename)
            if not content:
                return None
            
            # Truncate content if too long (OpenAI has limits)
            if len(content) > 8000:
                content = content[:8000]
            
            response = self.openai_client.embeddings.create(
                input=content,
                model=self.azure_openai_deployment
            )
            
            embedding = response.data[0].embedding
            self.file_embeddings_cache[filename] = embedding
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error getting embedding for {filename}: {e}")
            return None
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        if not NUMPY_AVAILABLE:
            # Fallback to manual calculation
            dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
            magnitude1 = math.sqrt(sum(a * a for a in embedding1))
            magnitude2 = math.sqrt(sum(b * b for b in embedding2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
        else:
            # Use numpy for faster calculation
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            dot_product = np.dot(vec1, vec2)
            magnitude1 = np.linalg.norm(vec1)
            magnitude2 = np.linalg.norm(vec2)
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
    
    def get_all_files(self) -> List[str]:
        """Get list of all files in the storage container."""
        try:
            if self.storage_manager:
                file_list = self.storage_manager.get_file_list(as_json=False)
                # Filter for supported file types
                supported_extensions = {
                    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.cxx', 
                    '.cc', '.h', '.hpp', '.cs', '.go', '.rs', '.swift', '.php', 
                    '.rb', '.pl', '.scala', '.sh', '.bash', '.zsh', '.r', '.R',
                    '.md', '.html', '.htm', '.css', '.json', '.yaml', '.yml',
                    '.xml', '.toml', '.txt'
                }
                return [f for f in file_list if Path(f).suffix.lower() in supported_extensions]
            return []
        except Exception as e:
            self.logger.error(f"Error getting file list: {e}")
            return []
    
    def analyze_file_similarities(self, similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """Analyze similarities between all files."""
        files = self.get_all_files()
        if not files:
            return {"error": "No files found"}
        
        similarities = []
        file_embeddings = {}
        
        # Get embeddings for all files
        for filename in files:
            embedding = self.get_file_embedding(filename)
            if embedding:
                file_embeddings[filename] = embedding
        
        # Calculate similarities between all pairs
        file_list = list(file_embeddings.keys())
        for i, file1 in enumerate(file_list):
            for j, file2 in enumerate(file_list[i+1:], i+1):
                similarity = self.calculate_similarity(
                    file_embeddings[file1], 
                    file_embeddings[file2]
                )
                
                if similarity >= similarity_threshold:
                    similarities.append({
                        'file1': file1,
                        'file2': file2,
                        'similarity': similarity,
                        'type': 'semantic'
                    })
        
        return {
            'files_analyzed': len(files),
            'files_with_embeddings': len(file_embeddings),
            'similarities_found': len(similarities),
            'similarities': similarities,
            'file_embeddings': {k: len(v) for k, v in file_embeddings.items()}
        }
    
    def analyze_file_dependencies(self) -> Dict[str, Any]:
        """Analyze dependencies between files."""
        files = self.get_all_files()
        if not files:
            return {"error": "No files found"}
        
        dependencies = {}
        all_imports = defaultdict(list)
        all_references = defaultdict(list)
        
        for filename in files:
            file_type = self.get_file_type(filename)
            content = self.extract_file_content(filename)
            
            if content:
                file_deps = self.detect_dependencies(content, file_type)
                dependencies[filename] = file_deps
                
                # Track all imports and references
                for imp in file_deps['imports']:
                    all_imports[imp].append(filename)
                for ref in file_deps['references']:
                    all_references[ref].append(filename)
        
        # Find actual file dependencies
        file_dependencies = []
        for filename, deps in dependencies.items():
            for imp in deps['imports']:
                # Try to find matching files
                for other_file in files:
                    if other_file != filename:
                        # Check if import matches file name or module name
                        if (imp in other_file or 
                            Path(other_file).stem == imp or
                            Path(other_file).stem.replace('_', '') == imp.replace('_', '')):
                            file_dependencies.append({
                                'from_file': filename,
                                'to_file': other_file,
                                'type': 'import',
                                'import_name': imp
                            })
        
        return {
            'files_analyzed': len(files),
            'files_with_dependencies': len(dependencies),
            'dependencies_found': len(file_dependencies),
            'dependencies': file_dependencies,
            'file_dependency_details': dependencies,
            'import_summary': dict(all_imports),
            'reference_summary': dict(all_references)
        }
    
    def create_graph_data(self, 
                         include_similarities: bool = True,
                         include_dependencies: bool = True,
                         similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """Create graph data for visualization."""
        if self.graph_data_cache:
            return self.graph_data_cache
        
        nodes = []
        edges = []
        
        # Get all files
        files = self.get_all_files()
        
        # Create nodes
        for filename in files:
            file_type = self.get_file_type(filename)
            content = self.extract_file_content(filename)
            
            node = {
                'id': filename,
                'label': Path(filename).name,
                'type': file_type,
                'size': len(content) if content else 0,
                'path': filename
            }
            nodes.append(node)
        
        # Add similarity edges
        if include_similarities:
            similarities = self.analyze_file_similarities(similarity_threshold)
            for sim in similarities.get('similarities', []):
                edge = {
                    'source': sim['file1'],
                    'target': sim['file2'],
                    'type': 'similarity',
                    'weight': sim['similarity'],
                    'label': f"{sim['similarity']:.2f}"
                }
                edges.append(edge)
        
        # Add dependency edges
        if include_dependencies:
            dependencies = self.analyze_file_dependencies()
            for dep in dependencies.get('dependencies', []):
                edge = {
                    'source': dep['from_file'],
                    'target': dep['to_file'],
                    'type': 'dependency',
                    'weight': 1.0,
                    'label': dep['import_name']
                }
                edges.append(edge)
        
        graph_data = {
            'nodes': nodes,
            'edges': edges,
            'metadata': {
                'total_files': len(files),
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'similarity_edges': len([e for e in edges if e['type'] == 'similarity']),
                'dependency_edges': len([e for e in edges if e['type'] == 'dependency']),
                'generated_at': datetime.now().isoformat()
            }
        }
        
        self.graph_data_cache = graph_data
        return graph_data
    
    def get_file_relationships(self, filename: str, max_relationships: int = 10) -> Dict[str, Any]:
        """Get relationships for a specific file."""
        if filename in self.file_relationships_cache:
            return self.file_relationships_cache[filename]
        
        file_type = self.get_file_type(filename)
        content = self.extract_file_content(filename)
        
        if not content:
            return {"error": f"Could not extract content from {filename}"}
        
        # Get dependencies
        dependencies = self.detect_dependencies(content, file_type)
        
        # Get similar files
        similar_files = []
        if self.openai_client:
            file_embedding = self.get_file_embedding(filename)
            if file_embedding:
                all_files = self.get_all_files()
                similarities = []
                
                for other_file in all_files:
                    if other_file != filename:
                        other_embedding = self.get_file_embedding(other_file)
                        if other_embedding:
                            similarity = self.calculate_similarity(file_embedding, other_embedding)
                            similarities.append({
                                'file': other_file,
                                'similarity': similarity
                            })
                
                # Sort by similarity and take top results
                similarities.sort(key=lambda x: x['similarity'], reverse=True)
                similar_files = similarities[:max_relationships]
        
        relationships = {
            'filename': filename,
            'file_type': file_type,
            'content_length': len(content),
            'dependencies': dependencies,
            'similar_files': similar_files,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        self.file_relationships_cache[filename] = relationships
        return relationships
    
    def clear_cache(self):
        """Clear all caches."""
        self.file_embeddings_cache.clear()
        self.file_relationships_cache.clear()
        self.graph_data_cache = None
        self.logger.info("File graph caches cleared")
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the graph data."""
        graph_data = self.create_graph_data()
        
        if 'error' in graph_data:
            return graph_data
        
        # Analyze node types
        node_types = Counter(node['type'] for node in graph_data['nodes'])
        
        # Analyze edge types
        edge_types = Counter(edge['type'] for edge in graph_data['edges'])
        
        # Find most connected nodes
        node_connections = defaultdict(int)
        for edge in graph_data['edges']:
            node_connections[edge['source']] += 1
            node_connections[edge['target']] += 1
        
        most_connected = sorted(node_connections.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'total_nodes': len(graph_data['nodes']),
            'total_edges': len(graph_data['edges']),
            'node_types': dict(node_types),
            'edge_types': dict(edge_types),
            'most_connected_files': [
                {'file': file, 'connections': count} 
                for file, count in most_connected
            ],
            'average_connections': sum(node_connections.values()) / len(node_connections) if node_connections else 0,
            'graph_density': len(graph_data['edges']) / (len(graph_data['nodes']) * (len(graph_data['nodes']) - 1)) if len(graph_data['nodes']) > 1 else 0
        } 