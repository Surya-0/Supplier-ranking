import streamlit as st
import networkx as nx
import plotly.graph_objects as go
from Supplier_Ranking import SupplyChainGraph
import pandas as pd
from typing import List, Dict, Any


def show():
    st.title("🔍 Supply Chain Query Explorer")

    # Initialize supply chain graph
    if 'scg' not in st.session_state:
        st.session_state.scg = SupplyChainGraph()

    # Sidebar configuration
    st.sidebar.header("Query Configuration")
    base_url = st.sidebar.text_input("Base URL", value="http://localhost:8000")
    version = st.sidebar.text_input("Version", value="test-v1")
    timestamp = st.sidebar.text_input("Timestamp", value="1")

    if st.sidebar.button("Load Data"):
        with st.spinner("Fetching data..."):
            data = st.session_state.scg.fetch_data(base_url, version, timestamp)
            if data:
                st.session_state.G = st.session_state.scg.build_graph()
                st.success("Data loaded successfully!")
            else:
                st.error("Failed to fetch data. Please check your connection and parameters.")

    if 'G' in st.session_state and st.session_state.G is not None:
        # Create tabs for different query types
        query_tab, path_tab, metrics_tab = st.tabs(["Node Query", "Path Explorer", "Advanced Metrics"])

        with query_tab:
            display_node_query_interface()

        with path_tab:
            display_path_explorer()

        with metrics_tab:
            display_advanced_metrics()
    else:
        st.warning("Please load the data to start querying.")


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

            # Neighborhood exploration options
            st.subheader("Neighborhood Explorer Options")
            exploration_mode = st.radio("Choose Neighborhood Exploration Mode",
                                        ["All nodes up to N hops", "Only nodes at exact N hops"])

            # Maximum hops input
            max_hops = st.slider("Select Hop Distance", 1, 3, 1)

            if exploration_mode == "All nodes up to N hops":
                # Option 1: Show all nodes up to `max_hops` using get_n_hop_neighbors
                neighbors = get_n_hop_neighbors(st.session_state.G, selected_node, max_hops)
            else:
                # Option 2: Show only nodes at `max_hops` using get_exact_n_hop_neighbors
                neighbors = get_exact_hop_neighbors(st.session_state.G, selected_node, max_hops)

            # Display neighbor filtering options
            if neighbors:
                neighbor_types = list(
                    set(nx.get_node_attributes(st.session_state.G.subgraph(neighbors), 'type').values()))
                selected_neighbor_type = st.multiselect("Filter Neighbors by Type", neighbor_types)

                # Filter and display neighbors
                filtered_neighbors = filter_neighbors_by_type(
                    st.session_state.G,
                    neighbors,
                    selected_neighbor_type if selected_neighbor_type else neighbor_types
                )

                if filtered_neighbors:
                    st.write(f"Found {len(filtered_neighbors)} matching neighbors at {max_hops} hop(s):")
                    for neighbor in filtered_neighbors:
                        with st.expander(f"Neighbor: {neighbor}"):
                            st.json(st.session_state.G.nodes[neighbor])

                            # Display edge attributes (optimize by combining checks)
                            if st.session_state.G.has_edge(selected_node, neighbor) or st.session_state.G.has_edge(
                                    neighbor, selected_node):
                                st.subheader("Edge Attributes")
                                if st.session_state.G.has_edge(selected_node, neighbor):
                                    st.json(st.session_state.G.edges[selected_node, neighbor])
                                if st.session_state.G.has_edge(neighbor, selected_node):
                                    st.json(st.session_state.G.edges[neighbor, selected_node])
            else:
                st.write(f"No neighbors found for the selected hop configuration.")


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
        except Exception as e:
            st.error(f"An error occurred: {e}")


def display_advanced_metrics():
    st.subheader("Node and Edge Filtering")

    # Node type selection for filtering
    node_types = list(set(nx.get_node_attributes(st.session_state.G, 'type').values()))
    selected_node_type = st.selectbox("Select Node Type to Filter", node_types)

    # Get all possible node attributes for the selected type
    sample_node = next(n for n, d in st.session_state.G.nodes(data=True)
                       if d.get('type') == selected_node_type)
    node_attributes = list(st.session_state.G.nodes[sample_node].keys())
    node_attributes.remove('type')  # Remove type as it's already selected

    if node_attributes:
        # Node attribute filtering
        st.write("Filter Nodes by Attributes")
        selected_attr = st.selectbox("Select Attribute", node_attributes)

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

            filtered_nodes = [n for n, d in st.session_state.G.nodes(data=True)
                              if d.get('type') == selected_node_type
                              and d.get(selected_attr) is not None
                              and filter_value[0] <= float(d[selected_attr]) <= filter_value[1]]
        else:
            # Categorical filter
            filter_value = st.multiselect(f"Select {selected_attr}", list(unique_values))
            filtered_nodes = [n for n, d in st.session_state.G.nodes(data=True)
                              if d.get('type') == selected_node_type
                              and d.get(selected_attr) in filter_value]

        if filtered_nodes:
            st.write(f"Found {len(filtered_nodes)} matching nodes")
            st.write("Sample of filtered nodes:")

            # Create a DataFrame for better visualization
            filtered_data = []
            for node in filtered_nodes[:10]:  # Show first 10 nodes
                node_data = st.session_state.G.nodes[node]
                filtered_data.append({
                    'Node ID': node,
                    **{k: v for k, v in node_data.items() if k != 'type'}
                })

            if filtered_data:
                st.dataframe(pd.DataFrame(filtered_data))


def get_n_hop_neighbors(G: nx.Graph, node: str, n: int) -> set:
    """Get all neighbors within n hops, excluding the node itself."""
    if node not in G:
        st.warning(f"Node {node} does not exist in the graph.")
        return set()

    # Use NetworkX function to get neighbors within n hops
    neighbors = set(nx.single_source_shortest_path_length(G, node, cutoff=n).keys())

    # Exclude the node itself from the neighbors set
    neighbors.discard(node)

    return neighbors


def get_exact_hop_neighbors(G: nx.Graph, node: str, hop: int) -> set:
    """Get all neighbors exactly `hop` distance away from the node."""
    if node not in G:
        st.warning(f"Node {node} does not exist in the graph.")
        return set()

    # Get all nodes within hop distance using NetworkX
    neighbors_with_hops = nx.single_source_shortest_path_length(G, node, cutoff=hop)

    # Filter only the nodes that are exactly `hop` distance away
    exact_hop_neighbors = {n for n, distance in neighbors_with_hops.items() if distance == hop}

    return exact_hop_neighbors


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
    """Analyze critical paths in the supply chain that go through parts"""
    critical_paths = {
        "Supplier to Warehouse": []
    }

    # Find paths from suppliers to warehouses via parts
    suppliers = [n for n, d in G.nodes(data=True) if d.get('type') == 'supplier']
    warehouses = [n for n, d in G.nodes(data=True) if d.get('type') == 'warehouse']

    # Look for paths that go from Supplier -> Part -> Warehouse
    for supplier in suppliers:
        # Find all parts connected to this supplier
        supplier_to_part_paths = []
        for part in G.predecessors(supplier):  # Parts connected to the supplier
            if G.nodes[part].get('type') == 'part':
                # Find all warehouses connected to this part
                for warehouse in G.successors(part):
                    if G.nodes[warehouse].get('type') == 'warehouse':
                        # Record the path Supplier -> Part -> Warehouse
                        critical_paths["Supplier to Warehouse"].append([supplier, part, warehouse])

    return critical_paths


if __name__ == "__main__":
    show()
