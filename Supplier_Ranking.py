import requests
import networkx as nx
import pandas as pd
from datetime import datetime
import json
from typing import Dict, List, Tuple
import numpy as np


class SupplyChainGraph:
    def __init__(self):
        self.G = nx.DiGraph()
        self.raw_data = None

    def fetch_data(self, base_url: str, version: str, timestamp: str) -> Dict:
        """
        Fetch JSON data from the server using the provided API endpoint
        """
        url = f"{base_url}/api/archive/schema/{version}/{timestamp}"
        headers = {"accept": "application/json"}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            self.raw_data = response.json()
            return self.raw_data
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            return None

    def build_graph(self) -> nx.DiGraph:
        """
        Construct NetworkX graph from the JSON data
        """
        if not self.raw_data:
            raise ValueError("No data available. Please fetch data first.")

        # Add nodes
        for node_type, nodes in self.raw_data["node_values"].items():
            for node in nodes:
                # Create a dictionary of node attributes
                node_attrs = dict(zip(self.raw_data["node_types"][node_type], node))
                node_id = node_attrs.pop("id")  # Remove id from attributes

                # Add node type separately to avoid conflicts
                attrs = {
                    "type": node_type,  # Changed from node_type to type
                    **{
                        k: v
                        for k, v in node_attrs.items()
                        if k not in ["id", "node_type", "pk_value", "pk_field"]
                    },
                }
                self.G.add_node(node_id, **attrs)

        # Add edges
        for rel_type, relationships in self.raw_data["link_values"].items():
            for rel in relationships:
                rel_attrs = dict(
                    zip(self.raw_data["relationship_types"][rel_type], rel)
                )
                source = rel_attrs.pop("source")
                target = rel_attrs.pop("target")

                # Add edge with remaining attributes
                self.G.add_edge(
                    source,
                    target,
                    type=rel_type,  # Changed from relationship_type to type
                    **{
                        k: v
                        for k, v in rel_attrs.items()
                        if k not in ["source", "target", "relationship_type"]
                    },
                )

        return self.G


    def create_query_subgraph(
        self,
        source_node_types: List[str] = None,
        source_node_ids: List[str] = None,
        n_hops: int = 2,
        target_node_types: List[str] = None,
        target_node_ids: List[str] = None,
    ) -> nx.Graph:
        """
        Create a subgraph based on query parameters.
        """
        if not any(
            [source_node_types, source_node_ids, target_node_types, target_node_ids]
        ):
            return self.G  # Return full graph if no query parameters

        # Initialize result graph
        result_graph = nx.Graph()

        # Get source nodes
        source_nodes = set()
        if source_node_ids:
            # If specific nodes are selected, only use those
            source_nodes.update(source_node_ids)
        elif source_node_types:
            # Only use all nodes of a type if no specific nodes were selected
            source_nodes.update(
                [
                    n
                    for n, d in self.G.nodes(data=True)
                    if d.get("type") in source_node_types
                ]
            )

        # Add source nodes to the result graph with their attributes
        for node in source_nodes:
            if node in self.G:
                result_graph.add_node(node, **self.G.nodes[node])

        # Keep track of nodes at each hop level and all visited nodes
        current_level_nodes = source_nodes
        visited_nodes = set(source_nodes)

        # For each hop
        for _ in range(n_hops):
            next_level_nodes = set()

            # Process current level nodes
            for node in current_level_nodes:
                if node not in self.G:
                    continue

                # Get neighbors (undirected)
                neighbors = set(self.G.neighbors(node))
                new_neighbors = neighbors - visited_nodes

                # Add new neighbors and their edges
                for neighbor in new_neighbors:
                    # Add the neighbor node
                    result_graph.add_node(neighbor, **self.G.nodes[neighbor])
                    # Add the edge (undirected)
                    result_graph.add_edge(
                        node, neighbor, **self.G.edges[node, neighbor]
                    )

                next_level_nodes.update(new_neighbors)

            # Update visited nodes and prepare for next hop
            visited_nodes.update(next_level_nodes)
            current_level_nodes = next_level_nodes

            # If no new nodes were found, stop early
            if not next_level_nodes:
                break

        # Filter nodes based on target criteria if specified
        if target_node_ids or target_node_types:
            nodes_to_keep = set(source_nodes)  # Always keep source nodes

            if target_node_ids:
                # Keep only specified target nodes that we've reached
                nodes_to_keep.update(
                    n for n in result_graph.nodes() if n in target_node_ids
                )
            elif target_node_types:
                # Keep only nodes of target types that we've reached
                nodes_to_keep.update(
                    n
                    for n in result_graph.nodes()
                    if result_graph.nodes[n].get("type") in target_node_types
                )

            # Create new graph with only the filtered nodes and their edges
            result_graph = result_graph.subgraph(nodes_to_keep).copy()

        return result_graph

    def calculate_structural_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate structural metrics for supplier ranking
        """
        metrics = {}

        # Get suppliers from the current graph
        suppliers = [
            n for n, d in self.G.nodes(data=True) if d.get("type") == "supplier"
        ]

        if not suppliers:
            return metrics

        # Calculate degree for undirected graph
        degree = dict(self.G.degree())

        try:
            # Calculate betweenness centrality
            betweenness_dict = nx.betweenness_centrality(self.G, normalized=True)

            # Calculate eigenvector centrality for each component
            eigenvector_dict = {}
            components = list(nx.connected_components(self.G))

            for component in components:
                subgraph = self.G.subgraph(component)
                try:
                    # Try to calculate eigenvector centrality for the component
                    component_eigenvector = nx.eigenvector_centrality(
                        subgraph, max_iter=1000
                    )
                    # Scale the values by component size
                    scale_factor = len(component) / len(self.G)
                    for node, value in component_eigenvector.items():
                        eigenvector_dict[node] = value * scale_factor
                except:
                    # If calculation fails, use degree centrality as fallback
                    component_degree = nx.degree_centrality(subgraph)
                    for node, value in component_degree.items():
                        eigenvector_dict[node] = value * scale_factor

        except Exception as e:
            print(f"Error calculating centrality metrics: {e}")
            # Fallback to simpler metrics if calculation fails
            betweenness_dict = nx.degree_centrality(self.G)
            eigenvector_dict = nx.degree_centrality(self.G)

        # Combine all metrics for each supplier
        for supplier in suppliers:
            metrics[supplier] = {
                "degree": degree.get(supplier, 0),  # Single degree for undirected graph
                "betweenness": betweenness_dict.get(supplier, 0),
                "eigenvector": eigenvector_dict.get(supplier, 0),
            }

        return metrics

    def calculate_attribute_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics based on node and edge attributes
        """
        suppliers = [
            n for n, d in self.G.nodes(data=True) if d.get("type") == "supplier"
        ]

        metrics = {}
        for supplier in suppliers:
            # Get all incoming edges (part to supplier relationships)
            supplier_edges = self.G.in_edges(supplier, data=True)

            # Calculate metrics based on edge attributes
            po_count = sum(edge.get("#PO", 0) for _, _, edge in supplier_edges)
            preferred_supplier_count = sum(
                1
                for _, _, edge in supplier_edges
                if edge.get("Preferred Supplier") == "X"
            )

            # Get connected parts
            connected_parts = [edge[0] for edge in supplier_edges]
            critical_parts = sum(
                1
                for part in connected_parts
                if self.G.nodes[part].get("Part Critical") == "C"
            )

            metrics[supplier] = {
                "po_count": po_count,
                "preferred_supplier_ratio": (
                    preferred_supplier_count / len(supplier_edges)
                    if supplier_edges
                    else 0
                ),
                "critical_parts_count": critical_parts,
            }

        return metrics

    def calculate_final_ranking(
        self, structural_weight: float = 0.5, attribute_weight: float = 0.5
    ) -> pd.DataFrame:
        """
        Calculate final supplier ranking by combining structural and attribute metrics
        """
        structural_metrics = self.calculate_structural_metrics()

        # print(structural_metrics)
        attribute_metrics = self.calculate_attribute_metrics()

        # Create DataFrame for easier manipulation
        structural_df = pd.DataFrame.from_dict(structural_metrics, orient="index")
        attribute_df = pd.DataFrame.from_dict(attribute_metrics, orient="index")

        # Normalize all metrics to 0-1 scale
        structural_df_norm = (structural_df - structural_df.min()) / (
            structural_df.max() - structural_df.min()
        )
        attribute_df_norm = (attribute_df - attribute_df.min()) / (
            attribute_df.max() - attribute_df.min()
        )

        # Calculate weighted scores
        structural_score = structural_df_norm.mean(axis=1) * structural_weight
        attribute_score = attribute_df_norm.mean(axis=1) * attribute_weight

        structural_score = structural_df.mean(axis=1) * structural_weight
        attribute_score = attribute_df.mean(axis=1) * attribute_weight

        # Combine scores
        final_scores = structural_score + attribute_score

        # Create final ranking DataFrame
        ranking_df = pd.DataFrame(
            {
                "Supplier": final_scores.index,
                "Final Score": final_scores.values,
                "Structural Score": structural_score.values,
                "Attribute Score": attribute_score.values,
            }
        )

        # Sort by final score
        ranking_df = ranking_df.sort_values("Final Score", ascending=False).reset_index(
            drop=True
        )

        return ranking_df


# Example usage

