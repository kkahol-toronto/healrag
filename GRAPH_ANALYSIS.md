# File Graph Analysis and Visualization

## Overview

The HEALRAG system now includes comprehensive file graph analysis and visualization capabilities that help you understand relationships between files in your indexed content. This feature analyzes both **semantic similarities** and **structural dependencies** between files to create interactive visualizations.

## 🎯 Key Features

### 🔍 **Semantic Similarity Analysis**
- Uses Azure OpenAI embeddings to find files with similar content
- Calculates cosine similarity between file embeddings
- Configurable similarity thresholds
- Identifies files that work on similar topics or concepts

### 🔗 **Dependency Analysis**
- Detects import statements and file references
- Supports multiple programming languages (Python, JavaScript, TypeScript, Java, C++, C#, etc.)
- Identifies function and class dependencies
- Maps import/export relationships

### 📊 **Interactive Visualizations**
- **Network Graph**: Force-directed graph showing file relationships
- **Similarity Heatmap**: Matrix view of file similarities
- **Statistics Dashboard**: Comprehensive metrics and analytics
- **Export to HTML**: Standalone interactive visualizations

### 🚀 **API Integration**
- RESTful API endpoints for programmatic access
- Real-time graph generation
- Caching for performance optimization
- Authentication and authorization support

## 🏗️ Architecture

### Components

1. **FileGraphManager** (`healraglib/file_graph_manager.py`)
   - Analyzes file similarities using embeddings
   - Detects dependencies using regex patterns
   - Manages graph data and caching
   - Provides graph statistics

2. **GraphVisualizer** (`healraglib/graph_visualizer.py`)
   - Creates interactive visualizations using Plotly
   - Supports multiple layout algorithms
   - Exports to HTML and other formats
   - Provides Dash web application integration

### Supported File Types

| Language | Extensions | Import Detection | Dependency Analysis |
|----------|------------|------------------|-------------------|
| Python | `.py` | ✅ | ✅ |
| JavaScript | `.js`, `.jsx` | ✅ | ✅ |
| TypeScript | `.ts`, `.tsx` | ✅ | ✅ |
| Java | `.java` | ✅ | ✅ |
| C++ | `.cpp`, `.cxx`, `.cc`, `.h`, `.hpp` | ✅ | ✅ |
| C# | `.cs` | ✅ | ✅ |
| Go | `.go` | ✅ | ✅ |
| Rust | `.rs` | ✅ | ✅ |
| Swift | `.swift` | ✅ | ✅ |
| PHP | `.php` | ✅ | ✅ |
| Ruby | `.rb` | ✅ | ✅ |
| Perl | `.pl` | ✅ | ✅ |
| Scala | `.scala` | ✅ | ✅ |
| Shell | `.sh`, `.bash`, `.zsh` | ✅ | ✅ |
| R | `.r`, `.R` | ✅ | ✅ |
| Markdown | `.md` | ✅ | ✅ |
| HTML | `.html`, `.htm` | ✅ | ✅ |
| CSS | `.css` | ✅ | ✅ |
| Config | `.json`, `.yaml`, `.yml`, `.xml`, `.toml` | ✅ | ✅ |
| Text | `.txt` | ✅ | ✅ |

## 🚀 Getting Started

### 1. Installation

Install the required dependencies:

```bash
pip install plotly networkx dash dash-bootstrap-components
```

### 2. Environment Setup

Ensure your `.env` file includes the necessary Azure services:

```bash
# Azure Storage
AZURE_STORAGE_CONNECTION_STRING=your_storage_connection_string
AZURE_CONTAINER_NAME=your_container_name

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=your_openai_endpoint
AZURE_OPENAI_KEY=your_openai_key
AZURE_TEXT_EMBEDDING_MODEL=text-embedding-ada-002

# Azure Search
AZURE_SEARCH_ENDPOINT=your_search_endpoint
AZURE_SEARCH_KEY=your_search_key
AZURE_SEARCH_INDEX_NAME=your_index_name
```

### 3. Basic Usage

#### Using the Library Directly

```python
from healraglib import StorageManager, SearchIndexManager, FileGraphManager, GraphVisualizer

# Initialize components
storage_manager = StorageManager(connection_string, container_name)
search_manager = SearchIndexManager(storage_manager, ...)
file_graph_manager = FileGraphManager(storage_manager, search_manager, ...)
graph_visualizer = GraphVisualizer(file_graph_manager)

# Analyze file similarities
similarities = file_graph_manager.analyze_file_similarities(similarity_threshold=0.7)

# Analyze dependencies
dependencies = file_graph_manager.analyze_file_dependencies()

# Create graph data
graph_data = file_graph_manager.create_graph_data(
    include_similarities=True,
    include_dependencies=True,
    similarity_threshold=0.7
)

# Create visualization
network_fig = graph_visualizer.create_network_graph(graph_data)

# Export to HTML
graph_visualizer.export_graph_as_html(graph_data, "file_graph.html")
```

#### Using the API Endpoints

```bash
# Get graph statistics
curl -H "Authorization: Bearer YOUR_TOKEN" \
  https://your-domain.com/graph/statistics

# Analyze file similarities
curl -X POST -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"similarity_threshold": 0.7}' \
  https://your-domain.com/graph/similarities

# Create visualization
curl -X POST -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"include_similarities": true, "include_dependencies": true}' \
  https://your-domain.com/graph/visualize

# Export to HTML
curl -X POST -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"include_similarities": true, "include_dependencies": true}' \
  https://your-domain.com/graph/export
```

## 📊 API Reference

### Graph Analysis Endpoints

#### `POST /graph/analyze`
Analyze file relationships and create graph data.

**Request Body:**
```json
{
  "include_similarities": true,
  "include_dependencies": true,
  "similarity_threshold": 0.7
}
```

**Response:**
```json
{
  "success": true,
  "graph_data": {
    "nodes": [...],
    "edges": [...],
    "metadata": {...}
  }
}
```

#### `GET /graph/statistics`
Get comprehensive statistics about the file graph.

**Response:**
```json
{
  "success": true,
  "statistics": {
    "total_nodes": 50,
    "total_edges": 120,
    "node_types": {"python": 20, "javascript": 15, ...},
    "edge_types": {"similarity": 80, "dependency": 40},
    "most_connected_files": [...],
    "average_connections": 2.4,
    "graph_density": 0.098
  }
}
```

#### `POST /graph/similarities`
Analyze similarities between files.

**Request Body:**
```json
{
  "similarity_threshold": 0.7
}
```

#### `POST /graph/dependencies`
Analyze dependencies between files.

#### `POST /graph/relationships`
Get relationships for a specific file.

**Request Body:**
```json
{
  "filename": "main.py",
  "max_relationships": 10
}
```

#### `POST /graph/visualize`
Create interactive graph visualization.

**Request Body:**
```json
{
  "include_similarities": true,
  "include_dependencies": true,
  "similarity_threshold": 0.7,
  "layout": "force",
  "node_size_factor": 1.0,
  "edge_threshold": 0.5
}
```

#### `POST /graph/export`
Export graph visualization as HTML file.

#### `GET /graph/download/{filename}`
Download exported graph file.

#### `POST /graph/clear-cache`
Clear graph analysis cache.

## 🎨 Visualization Types

### 1. Network Graph
- **Force-directed layout**: Files are positioned based on their relationships
- **Node colors**: Different colors for different file types
- **Node sizes**: Proportional to file size or connection count
- **Edge types**: 
  - Blue dashed lines for similarities
  - Red solid lines for dependencies
- **Interactive features**: Hover for details, zoom, pan

### 2. Similarity Heatmap
- **Matrix visualization**: Shows similarity scores between all file pairs
- **Color coding**: Darker colors indicate higher similarity
- **Interactive**: Click to see exact similarity scores

### 3. Statistics Dashboard
- **File type distribution**: Pie chart showing file type breakdown
- **Edge type analysis**: Bar chart of relationship types
- **File size distribution**: Histogram of file sizes
- **Most connected files**: Bar chart of files with most relationships

## 🔧 Configuration Options

### Similarity Analysis
- **Similarity threshold**: Minimum similarity score (0.0 - 1.0)
- **Embedding model**: Azure OpenAI embedding deployment
- **Content truncation**: Maximum content length for embedding generation

### Dependency Analysis
- **Language patterns**: Regex patterns for different programming languages
- **Import detection**: Support for various import/export syntaxes
- **Reference matching**: File name and module name matching

### Visualization
- **Layout algorithms**: Force-directed, circular, spring, random
- **Node sizing**: Configurable size scaling factors
- **Edge filtering**: Minimum weight thresholds
- **Color schemes**: Customizable node and edge colors

## 📈 Performance Considerations

### Caching
- **Embedding cache**: File embeddings are cached to avoid regeneration
- **Graph data cache**: Graph structure is cached for repeated access
- **Relationship cache**: File relationships are cached per file

### Optimization
- **Batch processing**: Multiple files processed in batches
- **Parallel processing**: Embedding generation can be parallelized
- **Lazy loading**: Graph data generated on-demand

### Memory Management
- **Content truncation**: Large files are truncated for embedding
- **Garbage collection**: Caches can be cleared to free memory
- **Streaming**: Large graphs can be processed in chunks

## 🧪 Testing

Run the test script to verify functionality:

```bash
python test_graph_analysis.py
```

This will:
1. Initialize all components
2. Analyze file similarities and dependencies
3. Create visualizations
4. Export to HTML
5. Display comprehensive statistics

## 🔍 Troubleshooting

### Common Issues

1. **No files found**
   - Ensure files are uploaded to the Azure Storage container
   - Check container name configuration
   - Verify file extensions are supported

2. **Embedding generation fails**
   - Check Azure OpenAI configuration
   - Verify embedding model deployment name
   - Ensure API key has proper permissions

3. **Visualization not working**
   - Install required dependencies: `pip install plotly networkx dash`
   - Check browser compatibility
   - Verify JavaScript is enabled

4. **Performance issues**
   - Reduce similarity threshold
   - Clear caches periodically
   - Process files in smaller batches

### Debug Information

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check component status:

```python
# Check if components are initialized
print(f"Storage Manager: {storage_manager is not None}")
print(f"Search Manager: {search_manager is not None}")
print(f"Graph Manager: {file_graph_manager is not None}")
print(f"Visualizer: {graph_visualizer is not None}")
```

## 🚀 Advanced Usage

### Custom Dependency Patterns

Add custom patterns for new languages:

```python
# Add custom patterns to FileGraphManager
file_graph_manager.dependency_patterns['custom_lang'] = {
    'imports': [
        r'^import\s+(\w+)',
        r'^require\s+[\'"]([^\'"]+)[\'"]'
    ],
    'references': [
        r'[\'"]([^\'"]+\.custom)[\'"]'
    ]
}
```

### Custom Visualizations

Create custom visualization types:

```python
# Extend GraphVisualizer
class CustomVisualizer(GraphVisualizer):
    def create_custom_chart(self, graph_data):
        # Custom visualization logic
        pass
```

### Integration with Existing Workflows

```python
# Integrate with training pipeline
def enhanced_training_pipeline():
    # ... existing training code ...
    
    # Add graph analysis
    if file_graph_manager:
        graph_data = file_graph_manager.create_graph_data()
        graph_visualizer.export_graph_as_html(graph_data, "training_graph.html")
```

## 📚 Examples

### Example 1: Codebase Analysis

```python
# Analyze a Python codebase
similarities = file_graph_manager.analyze_file_similarities(0.8)
dependencies = file_graph_manager.analyze_file_dependencies()

print("Most similar files:")
for sim in similarities['similarities'][:5]:
    print(f"  {sim['file1']} ↔ {sim['file2']} ({sim['similarity']:.3f})")

print("Dependencies:")
for dep in dependencies['dependencies'][:5]:
    print(f"  {dep['from_file']} → {dep['to_file']}")
```

### Example 2: Documentation Analysis

```python
# Analyze documentation files
graph_data = file_graph_manager.create_graph_data(
    include_similarities=True,
    include_dependencies=False,  # No dependencies for docs
    similarity_threshold=0.6
)

# Create focused visualization
network_fig = graph_visualizer.create_network_graph(
    graph_data,
    layout='circular',
    node_size_factor=2.0
)
```

### Example 3: API Integration

```python
import requests

# Get graph statistics via API
response = requests.get(
    "https://your-domain.com/graph/statistics",
    headers={"Authorization": "Bearer YOUR_TOKEN"}
)

stats = response.json()
print(f"Total files: {stats['statistics']['total_nodes']}")
print(f"Total relationships: {stats['statistics']['total_edges']}")
```

## 🔮 Future Enhancements

### Planned Features
- **Temporal analysis**: Track how relationships change over time
- **Impact analysis**: Identify files that would be affected by changes
- **Code quality metrics**: Integrate with code quality analysis tools
- **Real-time updates**: Live graph updates as files change
- **Advanced layouts**: More sophisticated graph layout algorithms
- **Export formats**: Support for additional export formats (PNG, SVG, PDF)

### Integration Opportunities
- **CI/CD pipelines**: Generate graphs as part of build processes
- **IDE plugins**: Visualize relationships within development environments
- **Documentation generation**: Auto-generate dependency documentation
- **Code review tools**: Highlight affected files during reviews

---

This file graph analysis feature provides powerful insights into your codebase structure and relationships, helping you understand dependencies, identify similar functionality, and maintain better code organization. 