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
        headers = {'accept': 'application/json'}

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
                    **{k: v for k, v in node_attrs.items()
                       if k not in ["id", "node_type", "pk_value", "pk_field"]}
                }
                self.G.add_node(node_id, **attrs)

        # Add edges
        for rel_type, relationships in self.raw_data["link_values"].items():
            for rel in relationships:
                rel_attrs = dict(zip(self.raw_data["relationship_types"][rel_type], rel))
                source = rel_attrs.pop("source")
                target = rel_attrs.pop("target")

                # Add edge with remaining attributes
                self.G.add_edge(
                    source,
                    target,
                    type=rel_type,  # Changed from relationship_type to type
                    **{k: v for k, v in rel_attrs.items()
                       if k not in ["source", "target", "relationship_type"]}
                )

        return self.G

    def calculate_structural_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate structural metrics for supplier ranking
        """
        suppliers = [n for n, d in self.G.nodes(data=True) if d.get('type') == 'supplier']

        metrics = {}
        for supplier in suppliers:
            # Calculate various centrality measures
            in_degree = self.G.in_degree(supplier)
            out_degree = self.G.out_degree(supplier)
            betweenness = nx.betweenness_centrality(self.G)[supplier]
            eigenvector = nx.eigenvector_centrality_numpy(self.G)[supplier]

            metrics[supplier] = {
                'in_degree': in_degree,
                'out_degree': out_degree,
                'betweenness': betweenness,
                'eigenvector': eigenvector
            }
            #
            # if supplier == 'supplier_1002656':
            #     print("The metrics for the supplier is : ",metrics)


        return metrics

    def calculate_attribute_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics based on node and edge attributes
        """
        suppliers = [n for n, d in self.G.nodes(data=True) if d.get('type') == 'supplier']

        metrics = {}
        for supplier in suppliers:
            # Get all incoming edges (part to supplier relationships)
            supplier_edges = self.G.in_edges(supplier, data=True)

            # Calculate metrics based on edge attributes
            po_count = sum(edge.get('#PO', 0) for _, _, edge in supplier_edges)
            preferred_supplier_count = sum(1 for _, _, edge in supplier_edges
                                           if edge.get('Preferred Supplier') == 'X')

            # Get connected parts
            connected_parts = [edge[0] for edge in supplier_edges]
            critical_parts = sum(1 for part in connected_parts
                                 if self.G.nodes[part].get('Part Critical') == 'C')

            metrics[supplier] = {
                'po_count': po_count,
                'preferred_supplier_ratio': preferred_supplier_count / len(supplier_edges) if supplier_edges else 0,
                'critical_parts_count': critical_parts
            }

        return metrics

    def calculate_final_ranking(self,
                                structural_weight: float = 0.5,
                                attribute_weight: float = 0.5) -> pd.DataFrame:
        """
        Calculate final supplier ranking by combining structural and attribute metrics
        """
        structural_metrics = self.calculate_structural_metrics()

        # print(structural_metrics)
        attribute_metrics = self.calculate_attribute_metrics()



        # Create DataFrame for easier manipulation
        structural_df = pd.DataFrame.from_dict(structural_metrics, orient='index')
        attribute_df = pd.DataFrame.from_dict(attribute_metrics, orient='index')

        # Normalize all metrics to 0-1 scale
        structural_df_norm = (structural_df - structural_df.min()) / (structural_df.max() - structural_df.min())
        attribute_df_norm = (attribute_df - attribute_df.min()) / (attribute_df.max() - attribute_df.min())


        # Calculate weighted scores
        structural_score = structural_df_norm.mean(axis=1) * structural_weight
        attribute_score = attribute_df_norm.mean(axis=1) * attribute_weight

        structural_score = structural_df.mean(axis=1) * structural_weight
        attribute_score = attribute_df.mean(axis=1) * attribute_weight

        # Combine scores
        final_scores = structural_score + attribute_score

        # Create final ranking DataFrame
        ranking_df = pd.DataFrame({
            'Supplier': final_scores.index,
            'Final Score': final_scores.values,
            'Structural Score': structural_score.values,
            'Attribute Score': attribute_score.values
        })

        # Sort by final score
        ranking_df = ranking_df.sort_values('Final Score', ascending=False).reset_index(drop=True)

        return ranking_df


# Example usage
def main():
    # Initialize the supply chain graph
    scg = SupplyChainGraph()

    # Fetch data (replace with your actual base URL, version, and timestamp)
    data = scg.fetch_data(
        base_url="http://localhost:8000",
        version="v2",
        timestamp="3"
    )

    if data:
        # Build the graph
        G = scg.build_graph()

        # Check if the node exists in the graph
        if 'supplier_1002656' in G:
            print(G.nodes['supplier_1002656'])
        else:
            print("Node 'supplier_1002656' does not exist in the graph.")
        # print(G.nodes," ",len(G.nodes))

        # Check if the node exists in the graph
        node_id = 'supplier_1002656'
        if node_id in G:
            # Get incoming edges
            incoming_edges = G.in_edges(node_id, data=True)
            print(f"Incoming edges for {node_id}:")
            for edge in incoming_edges:
                print(edge)

            # Get outgoing edges
            outgoing_edges = G.out_edges(node_id, data=True)
            print(f"Outgoing edges for {node_id}:")
            for edge in outgoing_edges:
                print(edge)
        else:
            print(f"Node '{node_id}' does not exist in the graph.")

        # Calculate final ranking with custom weights
        ranking = scg.calculate_final_ranking(
            structural_weight=0.6,  # Give more weight to structural metrics
            attribute_weight=0.4  # Give less weight to attribute metrics
        )

        # Print the results
        print("\nSupplier Rankings:")
        print(ranking.to_string(index=False))

        # Optional: Save rankings to CSV
        ranking.to_csv('supplier_rankings.csv', index=False)


if __name__ == "__main__":
    main()