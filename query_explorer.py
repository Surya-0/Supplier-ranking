### query_explorer.py
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
from Supplier_Ranking import SupplyChainGraph
import pandas as pd
from typing import List, Dict, Any


def app():
    st.title("🔍 Supply Chain Query Explorer")

    # Initialize supply chain graph
    if 'scg' not in st.session_state:
        st.session_state.scg = SupplyChainGraph()

    # Sidebar configuration
    st.sidebar.header("Query Configuration")
    base_url = st.sidebar.text_input("Base URL", value="http://localhost:8000")
    version = st.sidebar.text_input("Version", value="v2")
    timestamp = st.sidebar.text_input("Timestamp", value="3")

    if st.sidebar.button("Load Data"):
        with st.spinner("Fetching data..."):
            data = st.session_state.scg.fetch_data(base_url, version, timestamp)
            if data:
                st.session_state.G = st.session_state.scg.build_graph()
                st.success("Data loaded successfully!")
            else:
                st.error("Failed to fetch data. Please check your connection and parameters.")

    if 'G' in st.session_state:
        # Create tabs for different query types
        query_tab, path_tab, metrics_tab = st.tabs(["Node Query", "Path Explorer", "Advanced Metrics"])

        with query_tab:
            display_node_query_interface()

        with path_tab:
            display_path_explorer()

        with metrics_tab:
            display_advanced_metrics()


def display_node_query_interface():
    """Display the node query interface"""
    st.subheader("Node Query Explorer")

    # Get all node types from the graph
    node_types = list(set(nx.get_node_attributes(st.session_state.G, 'type').values()))

    # Node type selection
    selected_type = st.selectbox("Select Node Type", node_types)

    # Get nodes of selected type
    nodes_of_type = [n for n, d in st.session_state.G.nodes(data=True)
                     if d.get('type') == selected_type]

    # Node selection
    if nodes_of_type:
        selected_node = st.selectbox("Select Node", nodes_of_type)

        # Display node attributes
        if selected_node:
            st.subheader("Node Attributes")
            node_attrs = st.session_state.G.nodes[selected_node]
            st.json(node_attrs)

            # Neighborhood exploration
            st.subheader("Neighborhood Explorer")
            max_hops = st.slider("Maximum Hops", 1, 3, 1)

            # Get and filter neighbors
            neighbors = get_n_hop_neighbors(st.session_state.G, selected_node, max_hops)

            # Display neighbor filtering options
            neighbor_types = list(set(nx.get_node_attributes(st.session_state.G.subgraph(neighbors), 'type').values()))
            selected_neighbor_type = st.multiselect("Filter Neighbors by Type", neighbor_types)

            # Filter and display neighbors
            filtered_neighbors = filter_neighbors_by_type(
                st.session_state.G,
                neighbors,
                selected_neighbor_type if selected_neighbor_type else neighbor_types
            )

            if filtered_neighbors:
                st.write(f"Found {len(filtered_neighbors)} matching neighbors:")
                for neighbor in filtered_neighbors:
                    with st.expander(f"Neighbor: {neighbor}"):
                        st.json(st.session_state.G.nodes[neighbor])

                        # Display edge attributes
                        if st.session_state.G.has_edge(selected_node, neighbor):
                            st.subheader("Edge Attributes (Outgoing)")
                            st.json(st.session_state.G.edges[selected_node, neighbor])
                        if st.session_state.G.has_edge(neighbor, selected_node):
                            st.subheader("Edge Attributes (Incoming)")
                            st.json(st.session_state.G.edges[neighbor, selected_node])


def display_path_explorer():
    """Display the path exploration interface"""
    st.subheader("Path Explorer")

    # Source and target node selection
    col1, col2 = st.columns(2)

    with col1:
        source_type = st.selectbox("Source Node Type",
                                   list(set(nx.get_node_attributes(st.session_state.G, 'type').values())),
                                   key="source_type")
        source_nodes = [n for n, d in st.session_state.G.nodes(data=True)
                        if d.get('type') == source_type]
        source_node = st.selectbox("Source Node", source_nodes, key="source_node")

    with col2:
        target_type = st.selectbox("Target Node Type",
                                   list(set(nx.get_node_attributes(st.session_state.G, 'type').values())),
                                   key="target_type")
        target_nodes = [n for n, d in st.session_state.G.nodes(data=True)
                        if d.get('type') == target_type]
        target_node = st.selectbox("Target Node", target_nodes, key="target_node")

    if st.button("Find Paths"):
        try:
            # Find all simple paths between source and target
            paths = list(nx.all_simple_paths(st.session_state.G, source_node, target_node, cutoff=4))

            if paths:
                st.success(f"Found {len(paths)} paths!")

                for i, path in enumerate(paths, 1):
                    with st.expander(f"Path {i} (Length: {len(path) - 1})"):
                        # Display path as a sequence
                        st.write(" → ".join(path))

                        # Display node and edge details along the path
                        for j in range(len(path) - 1):
                            st.write(f"\nEdge from {path[j]} to {path[j + 1]}:")
                            st.json(st.session_state.G.edges[path[j], path[j + 1]])
            else:
                st.warning("No paths found between selected nodes.")
        except nx.NetworkXNoPath:
            st.error("No path exists between selected nodes.")


def display_advanced_metrics():
    """Display advanced supply chain metrics"""
    st.subheader("Advanced Supply Chain Metrics")

    # Calculate and display various network metrics
    col1, col2 = st.columns(2)

    with col1:
        st.write("Network-Level Metrics")
        metrics = calculate_network_metrics(st.session_state.G)
        for metric, value in metrics.items():
            st.metric(metric, value)

    with col2:
        st.write("Node-Type Distribution")
        type_dist = calculate_node_type_distribution(st.session_state.G)
        st.bar_chart(type_dist)

    # Display critical path analysis
    st.subheader("Critical Path Analysis")
    critical_paths = analyze_critical_paths(st.session_state.G)
    if critical_paths:
        for path_type, paths in critical_paths.items():
            with st.expander(f"{path_type} Paths"):
                for path in paths:
                    st.write(" → ".join(path))


def get_n_hop_neighbors(G: nx.Graph, node: str, n: int) -> set:
    """Get all neighbors within n hops"""
    neighbors = set()
    current_neighbors = {node}

    for _ in range(n):
        next_neighbors = set()
        for current_node in current_neighbors:
            next_neighbors.update(G.neighbors(current_node))
        neighbors.update(next_neighbors)
        current_neighbors = next_neighbors

    return neighbors


def filter_neighbors_by_type(G: nx.Graph, neighbors: set, types: List[str]) -> List[str]:
    """Filter neighbors by node type"""
    return [n for n in neighbors if G.nodes[n].get('type') in types]


def calculate_network_metrics(G: nx.Graph) -> Dict[str, float]:
    """Calculate various network-level metrics"""
    return {
        "Average Clustering": round(nx.average_clustering(G.to_undirected()), 3),
        "Graph Density": round(nx.density(G), 3),
        "Average Degree": round(sum(dict(G.degree()).values()) / G.number_of_nodes(), 2),
        "Number of Connected Components": nx.number_connected_components(G.to_undirected())
    }


def calculate_node_type_distribution(G: nx.Graph) -> pd.Series:
    """Calculate distribution of node types"""
    node_types = nx.get_node_attributes(G, 'type')
    return pd.Series(node_types).value_counts()


def analyze_critical_paths(G: nx.Graph) -> Dict[str, List[List[str]]]:
    """Analyze critical paths in the supply chain"""
    critical_paths = {
        "Supplier to Warehouse": [],
        "Critical Parts": []
    }

    # Find paths from suppliers to warehouses
    suppliers = [n for n, d in G.nodes(data=True) if d.get('type') == 'supplier']
    warehouses = [n for n, d in G.nodes(data=True) if d.get('type') == 'warehouse']

    # Sample a few paths for demonstration
    for s in suppliers[:3]:
        for w in warehouses[:3]:
            try:
                path = nx.shortest_path(G, s, w)
                if path:
                    critical_paths["Supplier to Warehouse"].append(path)
            except nx.NetworkXNoPath:
                continue

    return critical_paths


if __name__ == "__main__":
    app()