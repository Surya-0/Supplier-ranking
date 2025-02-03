import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
import json
from networkx.drawing.layout import spring_layout
from Supplier_Ranking import SupplyChainGraph
import time
import os
import random
import pandas as pd


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
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
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
                node_info.append('<br>'.join([f'{k}: {v}' for k, v in attrs.items()]))

        if node_x:  # Only create trace if there are nodes of this type
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="top center",
                hovertext=node_info,
                name=node_type.capitalize(),
                marker=dict(
                    color=color,
                    size=20,
                    line=dict(width=2, color='white')
                )
            )
            node_traces.append(node_trace)

    # Create the figure
    fig = go.Figure(data=[edge_trace, *node_traces],
                    layout=go.Layout(
                        title=dict(
                            text=f'Network Graph{" (" + timestamp + ")" if timestamp else ""}',
                            font=dict(size=16)
                        ),
                        showlegend=True,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[
                            dict(
                                text="",
                                showarrow=False,
                                xref="paper", yref="paper"
                            )
                        ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        height=500
                    ))

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


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


def create_ranking_table_and_heatmap(G: nx.Graph):
    """Create both a table and heatmap visualization of supplier rankings"""

    # Add weight slider
    attribute_weight = st.slider(
        "Attribute Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        help="Weight given to supplier attributes. Structural weight will be (1 - attribute_weight)",
    )
    structural_weight = 1 - attribute_weight

    # Create a SupplyChainGraph instance for calculations
    scg = SupplyChainGraph()
    scg.G = G

    # Calculate final ranking using the weights
    ranking_df = scg.calculate_final_ranking(
        structural_weight=structural_weight, attribute_weight=attribute_weight
    )

    # Display the ranking results
    st.subheader("Supplier Rankings")

    # Format the dataframe for display
    display_df = ranking_df.copy()
    display_df["Supplier"] = [
        f"Supplier {s.split('_')[-1]}" for s in display_df["Supplier"]
    ]
    display_df = display_df.round(4)

    # Show the table with formatted values
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Supplier": st.column_config.TextColumn(help="Supplier identifier"),
            "Final Score": st.column_config.NumberColumn(
                help=f"Combined score (Structural: {structural_weight:.1f}, Attribute: {attribute_weight:.1f})",
                format="%.4f",
            ),
            "Structural Score": st.column_config.NumberColumn(
                help="Score based on network position and connectivity", format="%.4f"
            ),
            "Attribute Score": st.column_config.NumberColumn(
                help="Score based on supplier attributes", format="%.4f"
            ),
        },
    )

    # Add download button for the table
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="Download Rankings as CSV",
        data=csv,
        file_name="supplier_rankings.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # Create heatmap visualization
    st.subheader("Supplier Performance Heatmap")

    # Remove default padding around charts
    st.markdown(
        """
        <style>
            [data-testid="stPlotlyChart"] {
                padding: 0;
                border: none;
            }
            [data-testid="stPlotlyChart"] > div {
                padding: 0 !important;
                border: none !important;
            }
            .js-plotly-plot {
                padding: 0 !important;
                border: none !important;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # Prepare data for heatmap
    heatmap_df = display_df.set_index("Supplier")

    # Normalize scores for heatmap
    normalized_scores = heatmap_df.copy()
    for col in normalized_scores.columns:
        if normalized_scores[col].max() != normalized_scores[col].min():
            normalized_scores[col] = (
                normalized_scores[col] - normalized_scores[col].min()
            ) / (normalized_scores[col].max() - normalized_scores[col].min())
        else:
            normalized_scores[col] = 1  # If all values are the same

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=normalized_scores.values.T,
            x=normalized_scores.index,
            y=normalized_scores.columns,
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
            "showline": True,
            "showgrid": True,
            "zeroline": True,
            "gridcolor": "rgba(128, 128, 128, 0.2)",
            "linecolor": "rgba(128, 128, 128, 0.2)",
        },
        yaxis={
            "side": "left",
            "showline": True,
            "showgrid": True,
            "zeroline": True,
            "gridcolor": "rgba(128, 128, 128, 0.2)",
            "linecolor": "rgba(128, 128, 128, 0.2)",
        },
        autosize=True,
        margin=dict(t=50, l=80, r=20, b=80, pad=0),
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
    - **Final Score**: Combined score based on both structural and attribute metrics
    - **Structural Score**: Score based on the supplier's position and connectivity in the network
    - **Attribute Score**: Score based on supplier-specific attributes and performance metrics

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
                data = st.session_state.scg.fetch_data(base_url, version, timestamp)
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

            use_filtering = st.checkbox("Use Filtering", value=False)

            source_node_ids = None
            if not use_filtering:
                source_node_ids = st.multiselect(
                    "Source Node IDs (optional)",
                    options=source_nodes,
                    default=None,
                    help="Optionally select specific nodes of the chosen types",
                )

            else:
                if len(source_node_types) > 1:
                    st.warning(
                        "Please select only one node type for filtering to work."
                    )
                else:
                    selected_node_type = source_node_types[0]  # We know there's only one type due to warning above
                    
                    # Get all possible node attributes for the selected type
                    sample_node = next(n for n, d in st.session_state.G.nodes(data=True)
                                   if d.get('type') == selected_node_type)
                    node_attributes = list(st.session_state.G.nodes[sample_node].keys())
                    node_attributes.remove('type')  # Remove type as it's already selected

                    if node_attributes:
                        # Node attribute filtering
                        selected_attr = st.selectbox("Select Attribute to Filter By", node_attributes)

                        # Get unique values for the selected attribute
                        unique_values = set()
                        for _, attrs in st.session_state.G.nodes(data=True):
                            if attrs.get('type') == selected_node_type and selected_attr in attrs:
                                unique_values.add(attrs[selected_attr])

                        # Create appropriate filter input based on attribute values
                        if all(isinstance(x, (int, float)) for x in unique_values if x is not None):
                            # Numeric filter
                            min_val = min(x for x in unique_values if x is not None)
                            max_val = max(x for x in unique_values if x is not None)
                            filter_value = st.slider(f"Filter by {selected_attr}",
                                                 float(min_val), float(max_val),
                                                 (float(min_val), float(max_val)))

                            source_node_ids = [n for n, d in st.session_state.G.nodes(data=True)
                                          if d.get('type') == selected_node_type
                                          and d.get(selected_attr) is not None
                                          and filter_value[0] <= float(d[selected_attr]) <= filter_value[1]]
                        else:
                            # Categorical filter
                            filter_value = st.multiselect(f"Select {selected_attr} Values", list(unique_values))
                            source_node_ids = [n for n, d in st.session_state.G.nodes(data=True)
                                          if d.get('type') == selected_node_type
                                          and d.get(selected_attr) in filter_value]

                        if source_node_ids:
                            st.write(f"Found {len(source_node_ids)} matching nodes")

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

                create_ranking_table_and_heatmap(current_graph)
            else:
                st.info("No suppliers present in the selected subgraph to rank.")
    else:
        st.info("Please fetch data using the sidebar to begin analysis.")


if __name__ == "__main__":
    call()
