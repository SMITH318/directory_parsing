#!/usr/bin/env python3
"""
Enhanced Entity Extraction and Network Analysis
- Configurable entity extraction with normalization
- Multiple network types (co-occurrence, temporal, bipartite, tag-based)
- Network metrics and community detection
- Multiple export formats (JSON, GraphML, GEXF, CSV, D3.js)
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from config_loader import load_config
from entity_extraction import EntityExtractor, EntityNormalizer
from network_analysis import GraphBuilder, MetricsCalculator, CommunityDetector
from output import JSONExporter, GraphExporter, CSVExporter, D3Exporter


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Enhanced entity extraction and network analysis"
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (default: use built-in defaults)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: data/processed)'
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    config.validate_config()

    project_name = config.get_project_metadata()['name']
    print(f"=" * 60)
    print(f"Enhanced Entity Extraction & Network Analysis")
    print(f"Project: {project_name}")
    print(f"=" * 60)

    # Setup paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    input_file = project_root / "data" / "processed" / "tagged_articles.json"

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "data" / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        print(f"Error: {input_file} not found. Run tagging first.")
        return

    # Load articles
    print("\nLoading articles...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    articles = data.get("articles", [])
    print(f"Loaded {len(articles)} articles")

    # Step 1: Entity Extraction
    print("\n" + "=" * 60)
    print("Step 1: Entity Extraction")
    print("=" * 60)

    entity_config = config.get_entity_config()
    extractor = EntityExtractor(entity_config)

    raw_entities = extractor.extract(articles)
    filtered_entities = extractor.filter_by_mentions(raw_entities)

    # Step 2: Entity Normalization
    print("\n" + "=" * 60)
    print("Step 2: Entity Normalization")
    print("=" * 60)

    normalizer = EntityNormalizer(entity_config)
    normalized_entities = normalizer.normalize(filtered_entities)

    # Print normalization stats
    if normalizer.enabled:
        stats = normalizer.get_normalization_stats(filtered_entities, normalized_entities)
        print(f"Normalization reduced entity count:")
        print(f"  People: {stats['people']['original_count']} -> {stats['people']['normalized_count']} "
              f"(-{stats['people']['reduction']})")
        print(f"  Places: {stats['places']['original_count']} -> {stats['places']['normalized_count']} "
              f"(-{stats['places']['reduction']})")

    # Step 3: Network Construction
    print("\n" + "=" * 60)
    print("Step 3: Network Construction")
    print("=" * 60)

    network_config = config.get_network_config()
    graph_builder = GraphBuilder(network_config)

    networks = graph_builder.build_all_networks(normalized_entities, articles)

    # Apply filtering
    filtering_config = network_config.get('filtering', {})
    if filtering_config.get('min_degree', 1) > 1 or filtering_config.get('min_edge_weight', 0) > 0:
        print("\nApplying network filters...")
        for network_name, network_data in networks.items():
            if hasattr(network_data, 'number_of_nodes'):  # Single graph
                networks[network_name] = graph_builder.filter_network(network_data, filtering_config)

    # Step 4: Network Metrics
    print("\n" + "=" * 60)
    print("Step 4: Network Metrics")
    print("=" * 60)

    metrics_calculator = MetricsCalculator(network_config)
    metrics = metrics_calculator.calculate_all(networks)

    # Step 5: Community Detection
    print("\n" + "=" * 60)
    print("Step 5: Community Detection")
    print("=" * 60)

    community_detector = CommunityDetector(network_config)
    communities = {}

    for network_name, network_data in networks.items():
        if hasattr(network_data, 'number_of_nodes'):  # Single graph
            comm_result = community_detector.detect_communities(network_data)
            if comm_result:
                communities[network_name] = comm_result

                # Add communities to graph as node attributes
                for algo_name, comm_data in comm_result.items():
                    if 'partition' in comm_data:
                        community_detector.add_communities_to_graph(
                            network_data,
                            comm_data['partition']
                        )

                # Get community info
                if 'louvain' in comm_result and 'partition' in comm_result['louvain']:
                    comm_info = community_detector.get_community_info(
                        network_data,
                        comm_result['louvain']['partition'],
                        normalized_entities
                    )
                    communities[network_name]['louvain']['community_info'] = comm_info

    # Step 6: Export Results
    print("\n" + "=" * 60)
    print("Step 6: Exporting Results")
    print("=" * 60)

    # JSON Export
    json_exporter = JSONExporter(config.config)
    json_exporter.export_full_dataset(
        normalized_entities,
        networks,
        metrics,
        communities,
        config.get_project_metadata(),
        str(output_dir / "entity_network.json")
    )

    # Export individual networks
    graph_exporter = GraphExporter(config.config)
    csv_exporter = CSVExporter(config.config)
    d3_exporter = D3Exporter(config.config)

    for network_name, network_data in networks.items():
        if hasattr(network_data, 'number_of_nodes'):  # Single graph
            safe_name = network_name.replace('_', '-')

            # GraphML
            graph_exporter.export_graphml(
                network_data,
                str(output_dir / f"{safe_name}.graphml")
            )

            # GEXF
            graph_exporter.export_gexf(
                network_data,
                str(output_dir / f"{safe_name}.gexf")
            )

            # CSV edge list
            csv_exporter.export_edge_list(
                network_data,
                str(output_dir / f"{safe_name}_edges.csv")
            )

            # CSV node list
            csv_exporter.export_node_list(
                network_data,
                str(output_dir / f"{safe_name}_nodes.csv")
            )

            # D3.js format
            d3_exporter.export(
                network_data,
                str(output_dir / f"{safe_name}_d3.json")
            )

        elif isinstance(network_data, list):  # Temporal network
            safe_name = network_name.replace('_', '-')
            d3_exporter.export_timeline_format(
                network_data,
                str(output_dir / f"{safe_name}_timeline.json")
            )

    # Export backward-compatible format (for existing scripts)
    legacy_output = {
        'people': normalized_entities.get('people', {}),
        'places': normalized_entities.get('places', {}),
        'organizations': normalized_entities.get('organizations', {}),
        'article_entities': normalized_entities.get('article_entities', [])
    }

    with open(output_dir / "entities.json", 'w', encoding='utf-8') as f:
        json.dump(legacy_output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Complete!")
    print("=" * 60)
    print(f"\nGenerated files in: {output_dir}")
    print("  - entity_network.json         (Complete dataset)")
    print("  - entities.json               (Legacy format)")
    print("  - *cooccurrence*.graphml/gexf (Network files for Gephi)")
    print("  - *cooccurrence*_edges.csv    (Edge lists)")
    print("  - *cooccurrence*_d3.json      (D3.js visualization format)")

    if 'temporal_network' in networks:
        print("  - temporal-network_timeline.json (Temporal evolution data)")

    print("\nEntity Summary:")
    print(f"  People: {len(normalized_entities.get('people', {}))}")
    print(f"  Places: {len(normalized_entities.get('places', {}))}")
    print(f"  Organizations: {len(normalized_entities.get('organizations', {}))}")

    if communities:
        print("\nCommunity Detection:")
        for network_name, comm_data in communities.items():
            for algo_name, results in comm_data.items():
                num_comm = results.get('num_communities', 0)
                print(f"  {network_name} ({algo_name}): {num_comm} communities")


if __name__ == "__main__":
    main()
