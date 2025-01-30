### app.py
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
import json
import matplotlib.pyplot as plt
from networkx.drawing.layout import spring_layout
from Supplier_Ranking import SupplyChainGraph
from pyvis.network import Network
import streamlit.components.v1 as components
import time
import os
import random
import query_explorer

# Set page configuration
st.set_page_config(
    page_title="Supply Chain Analysis Dashboard",
    page_icon="🏭",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1E88E5;
    }
    h2 {
        color: #424242;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Dashboard'
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'current_ranking' not in st.session_state:
        st.session_state.current_ranking = None
    if 'scg' not in st.session_state:
        st.session_state.scg = SupplyChainGraph()



def dashboard_page():
    """Main dashboard page functionality"""
    st.title("🏭 Supply Chain Analysis Dashboard")

    # Sidebar configuration
    st.sidebar.header("Configuration")
    base_url = st.sidebar.text_input("Base URL", value="http://localhost:8000")
    version = st.sidebar.text_input("Version", value="v2")
    timestamp = st.sidebar.text_input("Timestamp", value="3")

    # Add max nodes slider
    max_nodes = st.sidebar.slider("Maximum nodes to display", 10, 200, 50)

    structural_weight = st.sidebar.slider(
        "Structural Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.1
    )
    attribute_weight = 1 - structural_weight

    if st.sidebar.button("Run Analysis"):
        with st.spinner("Fetching data and performing analysis..."):
            data = st.session_state.scg.fetch_data(base_url, version, timestamp)

            if data:
                # Build graph
                G = st.session_state.scg.build_graph()
                st.session_state.G = G  # Store graph in session state

                # Display network visualization
                st.subheader("Network Visualization")
                display_graph(G, timestamp, max_nodes)

                # Display network statistics
                st.subheader("Network Statistics")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Total Nodes", len(G.nodes()))
                with col2:
                    st.metric("Total Edges", len(G.edges()))
                with col3:
                    st.metric("Network Density", round(nx.density(G), 4))

                # Calculate rankings
                ranking = st.session_state.scg.calculate_final_ranking(
                    structural_weight=structural_weight,
                    attribute_weight=attribute_weight
                )

                # Store the ranking in session state
                st.session_state.current_ranking = ranking
                st.session_state.analysis_complete = True
            else:
                st.error("Failed to fetch data. Please check your connection and parameters.")

    # Display rankings and download button if analysis is complete
    if st.session_state.analysis_complete and st.session_state.current_ranking is not None:
        st.subheader("Supplier Rankings")
        st.dataframe(st.session_state.current_ranking.style.highlight_max(subset=['Final Score']))

        # Create download button
        csv = st.session_state.current_ranking.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Rankings as CSV",
            data=csv,
            file_name="supplier_rankings.csv",
            mime="text/csv",
            key='download_rankings'
        )



def display_graph(G: nx.Graph, timestamp: str = "", max_nodes: int = 50):
    """Display graph using pyvis"""
    # Create a new graph for visualization
    vis_graph = nx.Graph()

    # Randomly sample nodes if there are too many
    nodes_to_show = list(G.nodes())
    if len(nodes_to_show) > max_nodes:
        nodes_to_show = random.sample(nodes_to_show, max_nodes)

    # Add selected nodes and their edges
    for node in nodes_to_show:
        # Add node with its attributes
        vis_graph.add_node(node, **G.nodes[node])

        # Add edges between selected nodes
        for neighbor in G.neighbors(node):
            if neighbor in nodes_to_show:
                vis_graph.add_edge(node, neighbor, **G.edges[node, neighbor])

    # Create and configure the pyvis network
    net = Network(notebook=True, height="500px", width="100%")

    # Define fixed colors for node types
    node_colors = {
        "supplier": "#1E88E5",  # Blue
        "part": "#43A047",  # Green
        "warehouse": "#FDD835"  # Yellow
    }

    # Add nodes with different colors based on type
    for node, attrs in vis_graph.nodes(data=True):
        node_type = attrs.get("type", "default")
        color = node_colors.get(node_type, "#999999")  # Default gray for unknown types
        net.add_node(node, color=color, title=str(attrs))

    # Add edges
    for source, target, attrs in vis_graph.edges(data=True):
        net.add_edge(source, target, title=str(attrs))

    # Set physics options for better layout
    net.set_options("""
        var options = {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -100,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                },
                "minVelocity": 0.75,
                "solver": "forceAtlas2Based"
            },
            "nodes": {
                "font": {
                    "size": 12
                }
            },
            "edges": {
                "color": {
                    "opacity": 0.7
                },
                "smooth": {
                    "type": "continuous"
                }
            }
        }
    """)

    # Ensure cache directory exists
    os.makedirs("cache", exist_ok=True)

    # Generate HTML file
    html_file = f"cache/graph_{timestamp.replace(' ', '_')}.html"
    net.save_graph(html_file)

    # Read the generated HTML
    with open(html_file, "r") as f:
        source_code = f.read()
    components.html(source_code, height=500)


def main():
    """Main application function"""
    initialize_session_state()

    # Create navigation in sidebar
    st.sidebar.title("Navigation")
    pages = {
        "Dashboard": dashboard_page,
        "Query Explorer": query_explorer.app
    }

    # Add page selection
    selected_page = st.sidebar.radio("Go to", list(pages.keys()))

    # Update current page in session state
    st.session_state.current_page = selected_page

    # Display selected page
    pages[selected_page]()

    # Add footer with additional information
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        """
        This dashboard provides comprehensive supply chain analysis tools including:
        * Network visualization
        * Supplier ranking
        * Query exploration
        * Advanced metrics
        """
    )


if __name__ == "__main__":
    main()