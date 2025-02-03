### app.py
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
import json
from networkx.drawing.layout import spring_layout
from Supplier_Ranking import SupplyChainGraph
import os
import random
import query_explorer, query_main

# Set page configuration
st.set_page_config(
    page_title="Supply Chain Analysis Dashboard", page_icon="🏭", layout="wide"
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
    }
    .stPlotlyChart {
        background-color: transparent !important;
        padding: 0 !important;
        border: none !important;
    }
    .stPlotlyChart > div {
        padding: 0 !important;
        background-color: transparent !important;
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
    """,
    unsafe_allow_html=True,
)


def initialize_session_state():
    """Initialize session state variables"""
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"
    if "analysis_complete" not in st.session_state:
        st.session_state.analysis_complete = False
    if "current_ranking" not in st.session_state:
        st.session_state.current_ranking = None
    if "scg" not in st.session_state:
        st.session_state.scg = SupplyChainGraph()


def dashboard_page():
    """Main dashboard page functionality"""
    st.title("Supplier Ranking Analysis")

    # Sidebar configuration
    with st.sidebar:
        st.header("Data Configuration")

        # Use API_HOST environment variable or default to localhost
        default_host = os.getenv("API_HOST", "localhost")
        base_url = st.text_input("Base URL", value=f"http://{default_host}:8000")
        version = st.text_input("Version", value="test-v1")
        timestamp = st.text_input("Timestamp", value="1")

        # Add max nodes slider
        max_nodes = st.slider("Maximum nodes to display", 10, 200, 50)

        structural_weight = st.slider(
            "Structural Weight", min_value=0.0, max_value=1.0, value=0.6, step=0.1
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
                    attribute_weight=attribute_weight,
                )

                # Store the ranking in session state
                st.session_state.current_ranking = ranking
                st.session_state.analysis_complete = True
            else:
                st.error(
                    "Failed to fetch data. Please check your connection and parameters."
                )

    # Display rankings and download button if analysis is complete
    if (
        st.session_state.analysis_complete
        and st.session_state.current_ranking is not None
    ):
        st.subheader("Supplier Rankings")
        st.dataframe(
            st.session_state.current_ranking.style.highlight_max(subset=["Final Score"])
        )

        # Create download button
        csv = st.session_state.current_ranking.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Rankings as CSV",
            data=csv,
            file_name="supplier_rankings.csv",
            mime="text/csv",
            key="download_rankings",
        )


def display_graph(G: nx.Graph, timestamp: str = "", max_nodes: int = 50):
    """Display graph using Plotly"""
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

    # Use Fruchterman-Reingold layout for better node distribution
    pos = nx.spring_layout(vis_graph)

    # Create edges
    edge_x = []
    edge_y = []
    for edge in vis_graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Define fixed colors for node types
    node_colors = {
        "supplier": "#1E88E5",  # Blue
        "part": "#43A047",  # Green
        "warehouse": "#FDD835",  # Yellow
    }

    # Create node traces for each type
    node_traces = []
    for node_type, color in node_colors.items():
        node_x = []
        node_y = []
        node_text = []
        node_info = []

        for node, attrs in vis_graph.nodes(data=True):
            if attrs.get("type") == node_type:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(str(node))
                node_info.append("<br>".join([f"{k}: {v}" for k, v in attrs.items()]))

        if node_x:  # Only create trace if there are nodes of this type
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                hoverinfo="text",
                text=node_text,
                textposition="top center",
                hovertext=node_info,
                name=node_type.capitalize(),
                marker=dict(color=color, size=20, line=dict(width=2, color="white")),
            )
            node_traces.append(node_trace)

    # Create the figure
    fig = go.Figure(
        data=[edge_trace, *node_traces],
        layout=go.Layout(
            title=dict(
                text=f'Network Graph{" (" + timestamp + ")" if timestamp else ""}',
                font=dict(size=16),
            ),
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[dict(text="", showarrow=False, xref="paper", yref="paper")],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
        ),
    )

    # Display the plot in Streamlit with transparent background
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=40, b=0)
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


def main():
    """Main application function"""
    initialize_session_state()

    # Create navigation in sidebar
    st.sidebar.title("Navigation")
    pages = {
        "Supplier Ranking": dashboard_page,
        "Query Explorer": query_explorer.show,
        "Subgraph Analysis": query_main.call,
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
