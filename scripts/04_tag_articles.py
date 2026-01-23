#!/usr/bin/env python3
"""
Step 4: Automated Article Tagging
- Classifies articles based on configurable keyword sets.
- Optimized with pre-compiled regex for speed.
- Now supports configuration files for flexible tagging schemes.
"""

import json
import re
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from config_loader import load_config


def build_tag_patterns(tags_config):
    """
    Build compiled regex patterns from tag configuration.

    Args:
        tags_config: List of tag dictionaries from config

    Returns:
        Dictionary with tag info and compiled patterns
    """
    tag_patterns = {}

    for tag in tags_config:
        tag_id = tag['id']
        keywords = tag.get('keywords', [])
        weight = tag.get('weight', 1)

        # Compile regex for each keyword
        compiled_patterns = [
            re.compile(rf'\b{re.escape(kw.lower())}\b')
            for kw in keywords
        ]

        tag_patterns[tag_id] = {
            'label': tag.get('label', tag_id),
            'keywords': keywords,
            'weight': weight,
            'color': tag.get('color', '#808080'),
            'compiled': compiled_patterns
        }

    return tag_patterns


def calculate_tag_scores(text, headline, tag_patterns):
    """
    Calculate tag scores for an article.

    Args:
        text: Article text
        headline: Article headline
        tag_patterns: Dictionary of compiled tag patterns

    Returns:
        Dictionary of tag scores
    """
    combined = (headline + " " + text).lower()
    headline_lower = headline.lower()
    scores = defaultdict(float)

    for tag_id, config in tag_patterns.items():
        score = 0
        for pattern in config["compiled"]:
            # Count occurrences in full text
            matches = len(pattern.findall(combined))

            if matches > 0:
                # Check if keyword is specifically in the headline (bonus weight)
                in_headline = len(pattern.findall(headline_lower))
                # Headline matches are worth significantly more
                score += (in_headline * 5) + (matches - in_headline)

        if score > 0:
            scores[tag_id] = score * config["weight"]

    return scores


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Tag articles based on keyword configuration"
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (default: use built-in defaults)'
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    config.validate_config()

    # Get tag configuration
    tags_config = config.get_tags()
    tag_patterns = build_tag_patterns(tags_config)

    print(f"Loaded configuration: {config.get_project_metadata()['name']}")
    print(f"Using {len(tags_config)} tag categories")

    # Setup paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    input_file = project_root / "data" / "processed" / "articles.json"
    output_file = project_root / "data" / "processed" / "tagged_articles.json"

    if not input_file.exists():
        print(f"Error: {input_file} not found. Run the segmenter first.")
        return

    # Load articles
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    articles = data.get("articles", [])
    print(f"Tagging {len(articles)} articles...")

    # Track tag statistics
    tag_counts = defaultdict(int)

    for art in articles:
        # Calculate scores based on the text and headline
        text = art.get("full_text", art.get("text", ""))
        headline = art.get("headline", "")
        scores = calculate_tag_scores(text, headline, tag_patterns)

        # Sort tags by score
        assigned = []
        for tag_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            assigned.append({
                "tag": tag_id,
                "score": round(score, 2)
            })
            tag_counts[tag_id] += 1

        # Assign tags to article
        art["tags"] = assigned
        art["primary_tag"] = assigned[0]["tag"] if assigned else "general"

        # For backward compatibility, set specific flags
        # Check if any tag ID contains "whitechapel" or "ripper"
        has_ripper_tag = any(
            'whitechapel' in t["tag"] or 'ripper' in t["tag"]
            for t in assigned
        )
        art["is_whitechapel"] = has_ripper_tag
        art["is_ripper_related"] = has_ripper_tag

    # Add metadata to output
    output_data = {
        "metadata": {
            "project_name": config.get_project_metadata()['name'],
            "generated_at": datetime.now().isoformat(),
            "config_file": args.config if args.config else "default",
            "total_articles": len(articles),
            "tags_used": len(tags_config)
        },
        "articles": articles
    }

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    # Summary Stats
    print("-" * 50)
    print(f"Tagging Complete!")
    print(f"Total articles processed: {len(articles)}")
    print(f"\nTag distribution:")
    for tag_id, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True):
        tag_label = tag_patterns[tag_id]['label']
        print(f"  {tag_label} ({tag_id}): {count} articles")
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
