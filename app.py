import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import requests
import json
import matplotlib.pyplot as plt
from networkx.drawing.layout import spring_layout
from Supplier_Ranking import SupplyChainGraph
import time

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
        edge_type = edge[2].get('type', 'unknown')

        # Set edge color based on type
        if edge_type == 'part_to_part':
            edge_color = 'rgba(65, 105, 225, 0.5)'  # Royal blue with transparency
        elif edge_type == 'part_to_supplier':
            edge_color = 'rgba(50, 205, 50, 0.5)'  # Lime green with transparency
        else:  # part_to_warehouse
            edge_color = 'rgba(255, 165, 0, 0.5)'  # Orange with transparency

        # Create edge trace
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(
                width=1.5,
                color=edge_color
            ),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        )
        edge_traces.append(edge_trace)

    # Create node traces for each type
    node_types = {'supplier': [], 'part': [], 'warehouse': []}

    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_type = node[1].get('type', 'unknown')

        if node_type in node_types:
            node_types[node_type].append((x, y, node[0], node[1]))

    # Create separate traces for each node type with different styling
    node_traces = []

    # Styling for each node type
    node_styles = {
        'supplier': {
            'color': '#1E88E5',
            'symbol': 'circle',
            'size': 30,
            'name': 'Suppliers'
        },
        'part': {
            'color': '#43A047',
            'symbol': 'diamond',
            'size': 25,
            'name': 'Parts'
        },
        'warehouse': {
            'color': '#FDD835',
            'symbol': 'square',
            'size': 35,
            'name': 'Warehouses'
        }
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
                    if key != 'type':
                        info += f"{key}: {value}<br>"
                hover_text.append(info)

            node_trace = go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers+text',
                hoverinfo='text',
                text=[n[2].split('_')[1] if '_' in n[2] else n[2] for n in nodes],  # Show only ID numbers
                textposition="top center",
                hovertext=hover_text,
                name=node_styles[node_type]['name'],
                marker=dict(
                    symbol=node_styles[node_type]['symbol'],
                    size=node_styles[node_type]['size'],
                    color=node_styles[node_type]['color'],
                    line=dict(
                        width=2,
                        color='white'
                    )
                )
            )
            node_traces.append(node_trace)

    # Create the figure with all traces
    fig = go.Figure(data=[*edge_traces, *node_traces])

    # Update layout for better visualization
    fig.update_layout(
        title={
            'text': "Supply Chain Network",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(248,248,248,0.9)',
        paper_bgcolor='white',
        height=700  # Increased height for better visibility
    )

    # Add a slight zoom
    fig.update_xaxes(range=[min(pos[node][0] for node in G.nodes()) * 1.2,
                            max(pos[node][0] for node in G.nodes()) * 1.2])
    fig.update_yaxes(range=[min(pos[node][1] for node in G.nodes()) * 1.2,
                            max(pos[node][1] for node in G.nodes()) * 1.2])

    return fig


def create_ranking_heatmap(ranking_df):
    """Create a heatmap of supplier rankings"""
    # Normalize scores for better visualization
    score_columns = ['Structural Score', 'Attribute Score', 'Final Score']
    normalized_scores = ranking_df[score_columns].copy()
    for col in score_columns:
        normalized_scores[col] = (normalized_scores[col] - normalized_scores[col].min()) / \
                                 (normalized_scores[col].max() - normalized_scores[col].min())

    fig = go.Figure(data=go.Heatmap(
        z=normalized_scores.values.T,
        x=ranking_df['Supplier'],
        y=score_columns,
        colorscale='Viridis',
        hoverongaps=False
    ))

    fig.update_layout(
        title='Supplier Ranking Scores Heatmap',
        xaxis_title='Suppliers',
        yaxis_title='Metrics',
        height=400
    )

    return fig


def main():
    st.title("🏭 Supply Chain Analysis Dashboard")

    # Sidebar configuration
    st.sidebar.header("Configuration")
    base_url = st.sidebar.text_input("Base URL", value="http://localhost:8000")
    version = st.sidebar.text_input("Version", value="v2")
    timestamp = st.sidebar.text_input("Timestamp", value="3")

    structural_weight = st.sidebar.slider(
        "Structural Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.1
    )
    attribute_weight = 1 - structural_weight

    # Initialize supply chain graph
    scg = SupplyChainGraph()

    # Add a "Run Analysis" button
    if st.sidebar.button("Run Analysis"):
        with st.spinner("Fetching data and performing analysis..."):
            # Fetch and process data
            data = scg.fetch_data(base_url, version, timestamp)

            if data:
                # Create two columns for the layout
                col1, col2 = st.columns(2)

                # Build graph and create visualization
                G = scg.build_graph()

                st.subheader("Network Statistics")
                stats_col1, stats_col2, stats_col3 = st.columns(3)

                with stats_col1:
                    st.metric("Total Nodes", len(G.nodes()))
                with stats_col2:
                    st.metric("Total Edges", len(G.edges()))
                with stats_col3:
                    st.metric("Network Density",
                              round(nx.density(G), 4))

                st.subheader("Network Visualization")
                network_fig = create_network_visualization(G)
                st.plotly_chart(network_fig, use_container_width=True)


                # Calculate rankings
                ranking = scg.calculate_final_ranking(
                    structural_weight=structural_weight,
                    attribute_weight=attribute_weight
                )

                # Display rankings
                st.subheader("Supplier Rankings")
                col1, col2 = st.columns(2)

                with col1:
                    st.dataframe(ranking.style.highlight_max(subset=['Final Score']))

                with col2:
                    heatmap_fig = create_ranking_heatmap(ranking)
                    st.plotly_chart(heatmap_fig, use_container_width=True)

                # Download button for rankings
                csv = ranking.to_csv(index=False)
                st.download_button(
                    label="Download Rankings as CSV",
                    data=csv,
                    file_name="supplier_rankings.csv",
                    mime="text/csv",
                )
            else:
                st.error("Failed to fetch data. Please check your connection and parameters.")


if __name__ == "__main__":
    main()