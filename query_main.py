import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
import json
import matplotlib.pyplot as plt
from networkx.drawing.layout import spring_layout
from pyvis.network import Network
from Supplier_Ranking import SupplyChainGraph
import streamlit.components.v1 as components
import time
import os
import random
import pandas as pd


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
        "warehouse": "#FDD835",  # Yellow
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
    net.set_options(
        """
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
    """
    )

    # Ensure cache directory exists
    os.makedirs("cache", exist_ok=True)

    # Generate HTML file
    html_file = f"cache/graph_{timestamp.replace(' ', '_')}.html"
    net.save_graph(html_file)

    # Read the generated HTML
    with open(html_file, "r") as f:
        source_code = f.read()
    components.html(source_code, height=800)




def create_network_visualization(G):
    """Create an enhanced network visualization using Plotly with Kamada-Kawai layout"""
    # Use Kamada-Kawai layout for better node distribution
    pos = nx.kamada_kawai_layout(G)

    # Create edges
    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        # Get edge type for coloring
        edge_type = edge[2].get("type", "unknown")

        # Set edge color based on type
        if edge_type == "part_to_part":
            edge_color = "rgba(65, 105, 225, 0.5)"  # Royal blue with transparency
        elif edge_type == "part_to_supplier":
            edge_color = "rgba(50, 205, 50, 0.5)"  # Lime green with transparency
        else:  # part_to_warehouse
            edge_color = "rgba(255, 165, 0, 0.5)"  # Orange with transparency

        # Create edge trace
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=1.5, color=edge_color),
            hoverinfo="none",
            mode="lines",
            showlegend=False,
        )
        edge_traces.append(edge_trace)

    # Create node traces for each type
    node_types = {"supplier": [], "part": [], "warehouse": []}

    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_type = node[1].get("type", "unknown")

        if node_type in node_types:
            node_types[node_type].append((x, y, node[0], node[1]))

    # Create separate traces for each node type with different styling
    node_traces = []

    # Styling for each node type
    node_styles = {
        "supplier": {
            "color": "#1E88E5",
            "symbol": "circle",
            "size": 30,
            "name": "Suppliers",
        },
        "part": {"color": "#43A047", "symbol": "diamond", "size": 25, "name": "Parts"},
        "warehouse": {
            "color": "#FDD835",
            "symbol": "square",
            "size": 35,
            "name": "Warehouses",
        },
    }

    for node_type, nodes in node_types.items():
        if nodes:
            x_coords = [node[0] for node in nodes]
            y_coords = [node[1] for node in nodes]

            # Create hover text with node information
            hover_text = []
            for node in nodes:
                info = f"ID: {node[2]}<br>"
                for key, value in node[3].items():
                    if key != "type":
                        info += f"{key}: {value}<br>"
                hover_text.append(info)

            node_trace = go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="markers+text",
                hoverinfo="text",
                text=[
                    n[2].split("_")[1] if "_" in n[2] else n[2] for n in nodes
                ],  # Show only ID numbers
                textposition="top center",
                hovertext=hover_text,
                name=node_styles[node_type]["name"],
                marker=dict(
                    symbol=node_styles[node_type]["symbol"],
                    size=node_styles[node_type]["size"],
                    color=node_styles[node_type]["color"],
                    line=dict(width=2, color="white"),
                ),
            )
            node_traces.append(node_trace)

    # Create the figure with all traces
    fig = go.Figure(data=[*edge_traces, *node_traces])

    # Update layout for better visualization
    fig.update_layout(
        title={
            "text": "Supply Chain Network",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        },
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
        ),
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="rgba(248,248,248,0.9)",
        paper_bgcolor="white",
        height=700,  # Increased height for better visibility
    )

    # Add a slight zoom
    fig.update_xaxes(
        range=[
            min(pos[node][0] for node in G.nodes()) * 1.2,
            max(pos[node][0] for node in G.nodes()) * 1.2,
        ]
    )
    fig.update_yaxes(
        range=[
            min(pos[node][1] for node in G.nodes()) * 1.2,
            max(pos[node][1] for node in G.nodes()) * 1.2,
        ]
    )

    return fig


def create_ranking_table_and_heatmap(ranking_df):
    """Create both a table and heatmap visualization of supplier rankings"""
    # Define the metrics with better labels
    metrics = {
        "degree": "Total Connections",
        "betweenness": "Centrality Score",
        "eigenvector": "Network Influence",
    }

    # Create a copy of the dataframe with formatted column names for display
    display_df = ranking_df.copy()
    display_df.columns = [metrics[col] for col in display_df.columns]
    display_df.index = [f"Supplier {s.split('_')[-1]}" for s in display_df.index]

    # Round the values for better display
    display_df = display_df.round(4)

    # Show the table with formatted values
    st.subheader("Supplier Rankings Table")
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=False,
        column_config={
            "Total Connections": st.column_config.NumberColumn(
                help="Total number of direct connections to other nodes", format="%d"
            ),
            "Centrality Score": st.column_config.NumberColumn(
                help="Supplier's importance in connecting different parts of the network",
                format="%.4f",
            ),
            "Network Influence": st.column_config.NumberColumn(
                help="Overall influence based on connections to other important nodes",
                format="%.4f",
            ),
        },
    )

    # Add download button for the table
    csv = display_df.to_csv()
    st.download_button(
        label="Download Rankings as CSV",
        data=csv,
        file_name="supplier_rankings.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # Create normalized scores for heatmap visualization
    normalized_scores = ranking_df[list(metrics.keys())].copy()
    for col in metrics.keys():
        if normalized_scores[col].max() != normalized_scores[col].min():
            normalized_scores[col] = (
                                             normalized_scores[col] - normalized_scores[col].min()
                                     ) / (normalized_scores[col].max() - normalized_scores[col].min())
        else:
            normalized_scores[col] = 1  # If all values are the same

    # Format supplier IDs to be more readable
    supplier_labels = [f"Supplier {s.split('_')[-1]}" for s in normalized_scores.index]

    st.subheader("Supplier Rankings Heatmap")

    # Add custom CSS to remove padding
    st.markdown(
        """
        <style>
        [data-testid="stPlotlyChart"] {
            margin: 0;
            padding: 0;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=normalized_scores.values.T,
            x=supplier_labels,
            y=list(metrics.values()),
            colorscale="Viridis",
            hoverongaps=False,
            showscale=True,
        )
    )

    fig.update_layout(
        title={
            "text": "Supplier Performance Matrix",
            "y": 0.95,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
            "font": {"size": 20},
        },
        xaxis_title="Suppliers",
        yaxis_title="Performance Metrics",
        height=500,
        xaxis={
            "tickangle": -45,
            "showline": False,
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={
            "side": "left",
            "showline": False,
            "showgrid": False,
            "zeroline": False,
        },
        margin={"l": 150, "r": 50, "t": 80, "b": 100, "pad": 0, "autoexpand": True},
        # paper_bgcolor="rgba(0,0,0,0)",
        # plot_bgcolor="rgba(0,0,0,0)",
        autosize=True,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False, "responsive": True},
    )

    # Add a legend explaining the metrics
    st.markdown(
        """
    ### Metrics Explanation
    - **Total Connections**: Number of direct connections to other nodes (parts, warehouses, etc.)
    - **Centrality Score**: Supplier's importance in connecting different parts of the network
    - **Network Influence**: Overall influence based on connections to other important nodes

    *Values in the heatmap are normalized to a 0-1 scale for easy comparison*
    """
    )


def call():
    st.title("Supply Chain Analysis Dashboard")

    # Initialize session state for graph
    if "G" not in st.session_state:
        st.session_state.G = None
        st.session_state.scg = SupplyChainGraph()

    # Sidebar for data fetching and graph querying
    with st.sidebar:
        st.header("Data Configuration")

        # Data fetching inputs
        base_url = st.text_input("Base URL", value="http://localhost:8000")
        version = st.text_input("Version", value="v1")
        timestamp = st.text_input("Timestamp", value="1")

        if st.button("Fetch Data"):
            with st.spinner("Fetching data..."):
                data = st.session_state.scg.fetch_data(
                    base_url, version, timestamp
                )
                if data:
                    st.session_state.G = st.session_state.scg.build_graph()
                    st.success("Data fetched successfully!")
                else:
                    st.error("Failed to fetch data.")

        # Add separator
        st.markdown("---")

        # Graph querying section
        st.header("Graph Query")

        # Checkbox to enable/disable subgraph querying
        use_subgraph = st.checkbox("Use Query Subgraph", value=False)

        if use_subgraph and st.session_state.G:
            # Get available node types
            node_types = list(
                set(nx.get_node_attributes(st.session_state.G, "type").values())
            )

            # Source node configuration
            st.subheader("Source Nodes")
            source_node_types = st.multiselect(
                "Source Node Types",
                options=node_types,
                default=None,
                help="Select the types of nodes to start from",
            )

            # Get node IDs based on selected types
            source_nodes = []
            if source_node_types:
                source_nodes = [
                    n
                    for n, d in st.session_state.G.nodes(data=True)
                    if d.get("type") in source_node_types
                ]
                # Sort nodes for better readability
                source_nodes.sort()

            source_node_ids = st.multiselect(
                "Source Node IDs (optional)",
                options=source_nodes,
                default=None,
                help="Optionally select specific nodes of the chosen types",
            )

            # Number of hops
            n_hops = st.slider(
                "Number of Hops",
                min_value=1,
                max_value=5,
                value=2,
                help="Number of connections to traverse from source nodes",
            )

            # Create a temporary graph to find reachable nodes
            temp_graph = None
            available_target_types = node_types.copy()
            available_target_nodes = []

            if source_node_types or source_node_ids:
                # Create temporary graph to get reachable nodes
                temp_supply_chain = SupplyChainGraph()
                temp_supply_chain.G = st.session_state.G
                temp_graph = temp_supply_chain.create_query_subgraph(
                    source_node_types=source_node_types,
                    source_node_ids=source_node_ids,
                    n_hops=n_hops,
                )

                # Get available target types from the temporary graph
                available_target_types = list(
                    set(
                        d.get("type")
                        for _, d in temp_graph.nodes(data=True)
                        if d.get("type")
                        not in (source_node_types or [])  # Exclude source types
                    )
                )
                available_target_types.sort()

            # Target node configuration
            st.subheader("Target Nodes")
            target_node_types = st.multiselect(
                "Target Node Types",
                options=available_target_types,
                default=None,
                help="Select the types of nodes to include as targets",
            )

            # Get node IDs based on selected types and reachable nodes
            target_nodes = []
            if target_node_types and temp_graph:
                target_nodes = [
                    n
                    for n, d in temp_graph.nodes(data=True)
                    if d.get("type") in target_node_types
                ]
                # Sort nodes for better readability
                target_nodes.sort()

            target_node_ids = st.multiselect(
                "Target Node IDs (optional)",
                options=target_nodes,
                default=None,
                help="Optionally select specific target nodes",
            )

    # Main content
    if st.session_state.G is not None:
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Network Visualization", "Supplier Rankings"])

        # Get the appropriate graph based on query settings
        current_graph = st.session_state.G
        if use_subgraph:
            current_graph = st.session_state.scg.create_query_subgraph(
                source_node_types=source_node_types,
                source_node_ids=source_node_ids,
                n_hops=n_hops,
                target_node_types=target_node_types,
                target_node_ids=target_node_ids,
            )

        with tab1:
            st.header("Network Visualization")
            # Display graph using existing visualization function
            display_graph(current_graph, timestamp)

        with tab2:
            st.header("Supplier Rankings")

            # Calculate metrics only for suppliers in the current graph
            suppliers_in_graph = [
                n
                for n, d in current_graph.nodes(data=True)
                if d.get("type") == "supplier"
            ]

            if suppliers_in_graph:
                # Create a temporary graph with only the current subgraph for metrics calculation
                temp_supply_chain = SupplyChainGraph()
                temp_supply_chain.G = current_graph

                # Calculate metrics for the subgraph
                metrics = temp_supply_chain.calculate_structural_metrics()

                # Create ranking visualization
                ranking_df = pd.DataFrame.from_dict(metrics, orient="index")
                create_ranking_table_and_heatmap(ranking_df)
            else:
                st.info("No suppliers present in the selected subgraph to rank.")
    else:
        st.info("Please fetch data using the sidebar to begin analysis.")


if __name__ == "__main__":
    call()
