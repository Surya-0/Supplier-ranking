# Supplier Ranking System

This repository contains a sophisticated supplier ranking system that analyzes supply chain networks using both structural (network-based) and attribute-based metrics to evaluate supplier importance and performance.

## Ranking Algorithms

The system uses a comprehensive dual-metric approach combining both structural and attribute-based measurements:

### Structural Metrics

The following network centrality measures are used to evaluate a supplier's position and importance in the supply chain network:

1. **Degree Centrality**
   - *In-Degree*: Measures the number of direct incoming connections (parts supplied)
   - *Out-Degree*: Measures the number of direct outgoing connections

2. **Betweenness Centrality**
   - Measures how often a supplier acts as a bridge between other nodes in the network
   - Identifies suppliers that are crucial for connecting different parts of the supply chain
   - Higher values indicate suppliers that are more critical for supply chain flow

3. **Eigenvector Centrality**
   - Measures the influence of a supplier in the network
   - Takes into account not just the number of connections, but also the importance of those connections
   - Handles disconnected graphs by defaulting to 0 for isolated components

### Attribute-Based Metrics

The system also considers various business and performance attributes:

1. **Purchase Order Metrics**
   - Number of purchase orders (PO count)
   - Preferred supplier status

2. **Part Criticality**
   - Number of critical parts supplied
   - Relationship with critical components in the supply chain

### Final Ranking Calculation

The final ranking is computed using a weighted combination of structural and attribute metrics:
- Configurable weights for structural vs. attribute importance (adjustable through the UI)
- Default weight distribution: 50% structural, 50% attribute-based
- Normalized scores to ensure fair comparison across different metrics

## Visualization

The system includes interactive visualizations:
- Network graph showing supplier relationships
- Heatmaps of ranking distributions
- Configurable maximum nodes display for better visualization (10-200 nodes)

## Usage

The application provides a web-based dashboard where you can:
1. Configure the base URL, version, and timestamp for data fetching
2. Adjust the structural vs attribute weights for ranking
3. Set the maximum number of nodes to display
4. View interactive network visualizations and ranking results
