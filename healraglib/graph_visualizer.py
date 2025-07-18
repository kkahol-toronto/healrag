"""
Graph Visualizer for HEALRAG

Provides interactive graph visualizations for file relationships and dependencies
using Plotly and Dash for web-based visualization.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
from datetime import datetime
import base64
import io

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import networkx as nx
    import numpy as np
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from dash import Dash, html, dcc, Input, Output, callback_context
    from dash.exceptions import PreventUpdate
    import dash_bootstrap_components as dbc
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False


class GraphVisualizer:
    """
    Graph Visualizer for HEALRAG library.
    
    Creates interactive visualizations for file relationships and dependencies
    using Plotly and Dash.
    """
    
    def __init__(self, file_graph_manager):
        """
        Initialize the Graph Visualizer.
        
        Args:
            file_graph_manager: FileGraphManager instance
        """
        self.file_graph_manager = file_graph_manager
        self.logger = logging.getLogger(__name__)
        
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available. Install with: pip install plotly networkx")
        if not DASH_AVAILABLE:
            self.logger.warning("Dash not available. Install with: pip install dash dash-bootstrap-components")
    
    def create_network_graph(self, 
                           graph_data: Dict[str, Any],
                           layout: str = 'force',
                           node_size_factor: float = 1.0,
                           edge_threshold: float = 0.5) -> Optional[go.Figure]:
        """
        Create an interactive network graph using Plotly.
        
        Args:
            graph_data: Graph data from FileGraphManager
            layout: Layout algorithm ('force', 'circular', 'spring', 'random')
            node_size_factor: Factor to scale node sizes
            edge_threshold: Minimum similarity threshold for edges
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        try:
            nodes = graph_data.get('nodes', [])
            edges = graph_data.get('edges', [])
            
            if not nodes:
                return None
            
            # Filter edges by threshold
            filtered_edges = [
                edge for edge in edges 
                if edge.get('weight', 0) >= edge_threshold
            ]
            
            # Create NetworkX graph
            G = nx.Graph()
            
            # Add nodes
            for node in nodes:
                G.add_node(node['id'], **node)
            
            # Add edges
            for edge in filtered_edges:
                G.add_edge(edge['source'], edge['target'], **edge)
            
            # If no edges, create a simple layout with just nodes
            if len(filtered_edges) == 0:
                print("   ⚠️  No edges found, creating node-only visualization")
                # Create a simple circular layout for nodes only
                pos = {}
                import math
                for i, node in enumerate(G.nodes()):
                    angle = 2 * math.pi * i / len(G.nodes())
                    radius = 1.0
                    pos[node] = {'x': radius * math.cos(angle), 'y': radius * math.sin(angle)}
            else:
                # Calculate layout
                if layout == 'force':
                    pos = nx.spring_layout(G, k=1, iterations=50)
                elif layout == 'circular':
                    pos = nx.circular_layout(G)
                elif layout == 'spring':
                    pos = nx.spring_layout(G)
                elif layout == 'random':
                    pos = nx.random_layout(G)
                else:
                    pos = nx.spring_layout(G)
            
            # Extract positions
            node_x = []
            node_y = []
            node_labels = []
            node_sizes = []
            node_colors = []
            
            for node in G.nodes():
                if node in pos:
                    node_x.append(pos[node]['x'])
                    node_y.append(pos[node]['y'])
                    node_labels.append(G.nodes[node].get('label', str(node)))
                    node_sizes.append(max(10, G.nodes[node].get('size', 0) / 1000 * node_size_factor))
                    node_colors.append(self._get_node_color(G.nodes[node].get('type', 'unknown')))
            
            # Create node trace
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_labels,
                textposition="middle center",
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                textfont=dict(size=8)
            )
            
            # Create edge traces
            edge_traces = []
            
            # Separate traces for different edge types
            similarity_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'similarity']
            dependency_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'dependency']
            
            # Similarity edges (blue, dashed)
            if similarity_edges:
                edge_x = []
                edge_y = []
                edge_weights = []
                
                for edge in similarity_edges:
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_weights.append(G.edges[edge]['weight'])
                
                similarity_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=1, color='blue', dash='dash'),
                    hoverinfo='text',
                    text=[f"Similarity: {w:.3f}" for w in edge_weights],
                    mode='lines',
                    name='Similarity',
                    opacity=0.6
                )
                edge_traces.append(similarity_trace)
            
            # Dependency edges (red, solid)
            if dependency_edges:
                edge_x = []
                edge_y = []
                edge_labels = []
                
                for edge in dependency_edges:
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_labels.append(G.edges[edge].get('label', 'import'))
                
                dependency_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=2, color='red'),
                    hoverinfo='text',
                    text=[f"Dependency: {label}" for label in edge_labels],
                    mode='lines',
                    name='Dependency',
                    opacity=0.8
                )
                edge_traces.append(dependency_trace)
            
            # Create figure
            fig = go.Figure(data=edge_traces + [node_trace],
                          layout=go.Layout(
                              title=f'File Relationships Graph<br><sub>Nodes: {len(nodes)}, Edges: {len(filtered_edges)}</sub>',
                              showlegend=True,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=40),
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              plot_bgcolor='white'
                          ))
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating network graph: {e}")
            return None
    
    def create_similarity_heatmap(self, graph_data: Dict[str, Any]) -> Optional[go.Figure]:
        """
        Create a similarity heatmap for files.
        
        Args:
            graph_data: Graph data from FileGraphManager
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        try:
            nodes = graph_data.get('nodes', [])
            edges = graph_data.get('edges', [])
            
            if not nodes:
                return None
            
            # Create similarity matrix
            file_names = [node['label'] for node in nodes]
            n_files = len(file_names)
            similarity_matrix = np.zeros((n_files, n_files))
            
            # Fill similarity matrix
            for edge in edges:
                if edge.get('type') == 'similarity':
                    try:
                        i = next(i for i, node in enumerate(nodes) if node['id'] == edge['source'])
                        j = next(j for j, node in enumerate(nodes) if node['id'] == edge['target'])
                        similarity_matrix[i, j] = edge['weight']
                        similarity_matrix[j, i] = edge['weight']  # Symmetric
                    except StopIteration:
                        continue
            
            # Set diagonal to 1.0 (self-similarity)
            np.fill_diagonal(similarity_matrix, 1.0)
            
            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=similarity_matrix,
                x=file_names,
                y=file_names,
                colorscale='Viridis',
                zmin=0,
                zmax=1,
                colorbar=dict(title="Similarity Score")
            ))
            
            fig.update_layout(
                title='File Similarity Heatmap',
                xaxis_title='Files',
                yaxis_title='Files',
                width=800,
                height=600
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating similarity heatmap: {e}")
            return None
    
    def create_dependency_sankey(self, graph_data: Dict[str, Any]) -> Optional[go.Figure]:
        """
        Create a Sankey diagram for file dependencies.
        
        Args:
            graph_data: Graph data from FileGraphManager
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        try:
            edges = graph_data.get('edges', [])
            
            # Filter dependency edges
            dependency_edges = [edge for edge in edges if edge.get('type') == 'dependency']
            
            if not dependency_edges:
                return None
            
            # Create node mapping
            all_nodes = set()
            for edge in dependency_edges:
                all_nodes.add(edge['source'])
                all_nodes.add(edge['target'])
            
            node_list = list(all_nodes)
            node_to_index = {node: i for i, node in enumerate(node_list)}
            
            # Create Sankey data
            source = []
            target = []
            value = []
            label = []
            
            for edge in dependency_edges:
                source.append(node_to_index[edge['source']])
                target.append(node_to_index[edge['target']])
                value.append(1)  # All dependencies have equal weight
                label.append(edge.get('label', 'import'))
            
            # Create Sankey diagram
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=[Path(node).name for node in node_list],
                    color="blue"
                ),
                link=dict(
                    source=source,
                    target=target,
                    value=value,
                    label=label
                )
            )])
            
            fig.update_layout(
                title_text="File Dependencies Flow",
                font_size=10,
                width=1000,
                height=600
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating dependency Sankey: {e}")
            return None
    
    def create_statistics_dashboard(self, graph_data: Dict[str, Any]) -> Optional[go.Figure]:
        """
        Create a statistics dashboard with multiple charts.
        
        Args:
            graph_data: Graph data from FileGraphManager
            
        Returns:
            Plotly figure object
        """
        if not PLOTLY_AVAILABLE:
            return None
        
        try:
            nodes = graph_data.get('nodes', [])
            edges = graph_data.get('edges', [])
            
            if not nodes:
                return None
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('File Types Distribution', 'Edge Types', 'File Sizes', 'Most Connected Files'),
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "histogram"}, {"type": "bar"}]]
            )
            
            # 1. File types distribution (pie chart)
            file_types = [node['type'] for node in nodes]
            type_counts = {}
            for file_type in file_types:
                type_counts[file_type] = type_counts.get(file_type, 0) + 1
            
            fig.add_trace(
                go.Pie(labels=list(type_counts.keys()), values=list(type_counts.values())),
                row=1, col=1
            )
            
            # 2. Edge types (bar chart)
            edge_types = [edge['type'] for edge in edges]
            edge_type_counts = {}
            for edge_type in edge_types:
                edge_type_counts[edge_type] = edge_type_counts.get(edge_type, 0) + 1
            
            fig.add_trace(
                go.Bar(x=list(edge_type_counts.keys()), y=list(edge_type_counts.values())),
                row=1, col=2
            )
            
            # 3. File sizes (histogram)
            file_sizes = [node['size'] for node in nodes if node['size'] > 0]
            if file_sizes:
                fig.add_trace(
                    go.Histogram(x=file_sizes, nbinsx=20),
                    row=2, col=1
                )
            
            # 4. Most connected files (bar chart)
            node_connections = {}
            for edge in edges:
                node_connections[edge['source']] = node_connections.get(edge['source'], 0) + 1
                node_connections[edge['target']] = node_connections.get(edge['target'], 0) + 1
            
            if node_connections:
                sorted_connections = sorted(node_connections.items(), key=lambda x: x[1], reverse=True)[:10]
                top_files = [Path(file).name for file, _ in sorted_connections]
                connection_counts = [count for _, count in sorted_connections]
                
                fig.add_trace(
                    go.Bar(x=top_files, y=connection_counts),
                    row=2, col=2
                )
            
            fig.update_layout(
                title_text="File Graph Statistics Dashboard",
                height=800,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating statistics dashboard: {e}")
            return None
    
    def export_graph_as_html(self, 
                           graph_data: Dict[str, Any],
                           output_path: str = "file_graph.html",
                           include_all_charts: bool = True) -> bool:
        """
        Export the graph visualization as an interactive HTML file.
        
        Args:
            graph_data: Graph data from FileGraphManager
            output_path: Path to save the HTML file
            include_all_charts: Whether to include all chart types
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not PLOTLY_AVAILABLE:
            return False
        
        try:
            # Create main network graph
            network_fig = self.create_network_graph(graph_data)
            
            if not network_fig:
                return False
            
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>File Relationships Graph</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .chart-container {{ margin: 20px 0; }}
                    .controls {{ margin: 20px 0; padding: 10px; background-color: #f5f5f5; }}
                    .stats {{ margin: 20px 0; padding: 10px; background-color: #e8f4f8; }}
                </style>
            </head>
            <body>
                <h1>File Relationships Graph</h1>
                
                <div class="stats">
                    <h3>Graph Statistics</h3>
                    <p><strong>Total Files:</strong> {graph_data.get('metadata', {}).get('total_files', 0)}</p>
                    <p><strong>Total Nodes:</strong> {graph_data.get('metadata', {}).get('total_nodes', 0)}</p>
                    <p><strong>Total Edges:</strong> {graph_data.get('metadata', {}).get('total_edges', 0)}</p>
                    <p><strong>Similarity Edges:</strong> {graph_data.get('metadata', {}).get('similarity_edges', 0)}</p>
                    <p><strong>Dependency Edges:</strong> {graph_data.get('metadata', {}).get('dependency_edges', 0)}</p>
                    <p><strong>Generated:</strong> {graph_data.get('metadata', {}).get('generated_at', 'Unknown')}</p>
                </div>
                
                <div class="chart-container">
                    <h3>Interactive Network Graph</h3>
                    <div id="network-graph"></div>
                </div>
            """
            
            # Add network graph
            html_content += f"""
                <script>
                    var networkData = {network_fig.to_json()};
                    Plotly.newPlot('network-graph', networkData.data, networkData.layout);
                </script>
            """
            
            # Add additional charts if requested
            if include_all_charts:
                # Similarity heatmap
                heatmap_fig = self.create_similarity_heatmap(graph_data)
                if heatmap_fig:
                    html_content += f"""
                        <div class="chart-container">
                            <h3>Similarity Heatmap</h3>
                            <div id="heatmap"></div>
                        </div>
                        <script>
                            var heatmapData = {heatmap_fig.to_json()};
                            Plotly.newPlot('heatmap', heatmapData.data, heatmapData.layout);
                        </script>
                    """
                
                # Statistics dashboard
                stats_fig = self.create_statistics_dashboard(graph_data)
                if stats_fig:
                    html_content += f"""
                        <div class="chart-container">
                            <h3>Statistics Dashboard</h3>
                            <div id="stats-dashboard"></div>
                        </div>
                        <script>
                            var statsData = {stats_fig.to_json()};
                            Plotly.newPlot('stats-dashboard', statsData.data, statsData.layout);
                        </script>
                    """
            
            html_content += """
            </body>
            </html>
            """
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Graph exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting graph as HTML: {e}")
            return False
    
    def _get_node_color(self, file_type: str) -> str:
        """Get color for node based on file type."""
        color_map = {
            'python': '#3776ab',
            'javascript': '#f7df1e',
            'typescript': '#3178c6',
            'java': '#ed8b00',
            'cpp': '#00599c',
            'csharp': '#178600',
            'go': '#00add8',
            'rust': '#ce422b',
            'swift': '#ff6b4a',
            'php': '#777bb4',
            'ruby': '#cc342d',
            'perl': '#39457e',
            'scala': '#dc322f',
            'shell': '#4eaa25',
            'r': '#276dc3',
            'markdown': '#000000',
            'html': '#e34c26',
            'css': '#1572b6',
            'json': '#000000',
            'yaml': '#cb171e',
            'xml': '#ff6600',
            'toml': '#9c4128',
            'text': '#666666',
            'unknown': '#999999'
        }
        return color_map.get(file_type, '#999999')
    
    def create_dash_app(self, graph_data: Dict[str, Any]) -> Optional[Dash]:
        """
        Create a Dash web application for interactive graph visualization.
        
        Args:
            graph_data: Graph data from FileGraphManager
            
        Returns:
            Dash app object
        """
        if not DASH_AVAILABLE:
            return None
        
        try:
            app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
            
            app.layout = dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H1("File Relationships Graph", className="text-center mb-4"),
                        html.Hr()
                    ])
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Graph Statistics"),
                            dbc.CardBody([
                                html.P(f"Total Files: {graph_data.get('metadata', {}).get('total_files', 0)}"),
                                html.P(f"Total Nodes: {graph_data.get('metadata', {}).get('total_nodes', 0)}"),
                                html.P(f"Total Edges: {graph_data.get('metadata', {}).get('total_edges', 0)}"),
                                html.P(f"Similarity Edges: {graph_data.get('metadata', {}).get('similarity_edges', 0)}"),
                                html.P(f"Dependency Edges: {graph_data.get('metadata', {}).get('dependency_edges', 0)}")
                            ])
                        ])
                    ], width=3),
                    
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader("Graph Controls"),
                            dbc.CardBody([
                                dbc.Label("Layout Algorithm:"),
                                dcc.Dropdown(
                                    id='layout-dropdown',
                                    options=[
                                        {'label': 'Force-Directed', 'value': 'force'},
                                        {'label': 'Circular', 'value': 'circular'},
                                        {'label': 'Spring', 'value': 'spring'},
                                        {'label': 'Random', 'value': 'random'}
                                    ],
                                    value='force'
                                ),
                                html.Br(),
                                dbc.Label("Node Size Factor:"),
                                dcc.Slider(
                                    id='node-size-slider',
                                    min=0.1,
                                    max=3.0,
                                    step=0.1,
                                    value=1.0,
                                    marks={i/10: str(i/10) for i in range(1, 31, 5)}
                                ),
                                html.Br(),
                                dbc.Label("Edge Threshold:"),
                                dcc.Slider(
                                    id='edge-threshold-slider',
                                    min=0.0,
                                    max=1.0,
                                    step=0.1,
                                    value=0.5,
                                    marks={i/10: str(i/10) for i in range(0, 11, 2)}
                                )
                            ])
                        ])
                    ], width=3)
                ], className="mb-4"),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='network-graph')
                    ])
                ]),
                
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='similarity-heatmap')
                    ], width=6),
                    dbc.Col([
                        dcc.Graph(id='statistics-dashboard')
                    ], width=6)
                ])
            ], fluid=True)
            
            @app.callback(
                Output('network-graph', 'figure'),
                [Input('layout-dropdown', 'value'),
                 Input('node-size-slider', 'value'),
                 Input('edge-threshold-slider', 'value')]
            )
            def update_network_graph(layout, node_size, edge_threshold):
                return self.create_network_graph(graph_data, layout, node_size, edge_threshold)
            
            @app.callback(
                Output('similarity-heatmap', 'figure'),
                [Input('network-graph', 'figure')]
            )
            def update_heatmap(_):
                return self.create_similarity_heatmap(graph_data)
            
            @app.callback(
                Output('statistics-dashboard', 'figure'),
                [Input('network-graph', 'figure')]
            )
            def update_statistics(_):
                return self.create_statistics_dashboard(graph_data)
            
            return app
            
        except Exception as e:
            self.logger.error(f"Error creating Dash app: {e}")
            return None 