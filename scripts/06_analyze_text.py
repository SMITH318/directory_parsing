#!/usr/bin/env python3
"""
Step 6: Text Analysis & Comparative Linguistics
- Compares text characteristics across configurable article groups.
- Now supports N-way comparison with configurable groups and keywords.
"""

import json
import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from config_loader import load_config

# Default stop words (can be extended by config)
DEFAULT_STOP_WORDS = {
    'the', 'and', 'that', 'with', 'for', 'was', 'his', 'had', 'they', 'from',
    'said', 'were', 'been', 'which', 'there', 'this', 'would', 'their', 'will',
    'upon', 'then', 'into', 'them', 'when', 'more', 'some', 'than'
}


def clean_word(word):
    """Remove non-word characters from a word."""
    return re.sub(r'[^\w]', '', word.lower())


def get_words(text, stop_words):
    """Extract meaningful words from text."""
    words = [clean_word(w) for w in text.split()]
    return [w for w in words if len(w) > 2 and w not in stop_words]


def filter_articles(articles, filter_config):
    """
    Filter articles based on filter configuration.

    Args:
        articles: List of articles
        filter_config: Dict with 'tags' (include) and/or 'exclude_tags'

    Returns:
        Filtered list of articles
    """
    if not filter_config:
        return articles

    include_tags = filter_config.get('tags', [])
    exclude_tags = filter_config.get('exclude_tags', [])

    filtered = []
    for art in articles:
        art_tags = [t["tag"] for t in art.get("tags", [])]

        # Check exclusions first
        if exclude_tags and any(tag in exclude_tags for tag in art_tags):
            continue

        # Check inclusions
        if include_tags:
            if any(tag in include_tags for tag in art_tags):
                filtered.append(art)
        else:
            # No include filter, so include (unless excluded above)
            filtered.append(art)

    return filtered


def analyze_group(articles, sensational_words, stop_words):
    """
    Generate linguistic statistics for a group of articles.

    Args:
        articles: List of article dictionaries
        sensational_words: Set of sensational keywords
        stop_words: Set of stop words

    Returns:
        Dictionary with statistics
    """
    results = {
        "article_count": len(articles),
        "total_words": 0,
        "avg_sentence_len": 0,
        "exclamation_intensity": 0,
        "sensational_score": 0,
        "top_keywords": [],
        "top_phrases": [],
        "sensational_word_details": Counter()
    }

    if not articles:
        return results

    word_counts = Counter()
    bigram_counts = Counter()
    sensational_word_counter = Counter()
    total_excl = 0
    sentence_lengths = []

    for art in articles:
        text = art.get("full_text", "") or art.get("text", "")
        clean_text = text.lower()

        words = get_words(text, stop_words)
        word_counts.update(words)

        # Count bigrams
        for i in range(len(words) - 1):
            bigram_counts.update([f"{words[i]} {words[i+1]}"])

        # Count sensational words
        for word in sensational_words:
            matches = len(re.findall(rf'\b{word}\b', clean_text))
            if matches > 0:
                sensational_word_counter[word] += matches

        # Count exclamations
        total_excl += text.count("!")

        # Calculate sentence lengths
        sentences = re.split(r'[.!?]+', text)
        lengths = [len(s.split()) for s in sentences if len(s.split()) > 2]
        if lengths:
            sentence_lengths.extend(lengths)

    # Update results
    results["total_words"] = sum(word_counts.values())
    if sentence_lengths:
        results["avg_sentence_len"] = round(sum(sentence_lengths)/len(sentence_lengths), 1)

    results["exclamation_intensity"] = round(total_excl / len(articles), 2)
    results["sensational_score"] = round(sum(sensational_word_counter.values()) / len(articles), 2) if articles else 0
    results["top_keywords"] = word_counts.most_common(50)
    results["top_phrases"] = bigram_counts.most_common(20)
    results["sensational_word_details"] = sensational_word_counter

    return results


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Analyze text characteristics across article groups"
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

    # Get text analysis configuration
    analysis_config = config.get_text_analysis_config()
    comparison_groups = analysis_config.get('comparison_groups', [])
    sensational_keywords = set(analysis_config.get('sensational_keywords', []))
    custom_stopwords = set(analysis_config.get('custom_stopwords', []))
    wordcloud_config = analysis_config.get('wordcloud', {})

    # Combine default and custom stop words
    stop_words = DEFAULT_STOP_WORDS | custom_stopwords

    print(f"Loaded configuration: {config.get_project_metadata()['name']}")
    print(f"Comparison groups: {len(comparison_groups)}")
    print(f"Sensational keywords: {len(sensational_keywords)}")

    # Setup paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    input_file = project_root / "data" / "processed" / "tagged_articles.json"
    output_file = project_root / "data" / "processed" / "text_analysis.json"

    if not input_file.exists():
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    all_articles = data.get("articles", [])
    print(f"Analyzing {len(all_articles)} total articles...")

    # Analyze each comparison group
    group_results = []

    for group_config in comparison_groups:
        group_id = group_config['id']
        group_label = group_config.get('label', group_id)
        group_description = group_config.get('description', '')
        filter_config = group_config.get('filter', {})

        # Filter articles for this group
        group_articles = filter_articles(all_articles, filter_config)
        print(f"  {group_label}: {len(group_articles)} articles")

        # Analyze the group
        stats = analyze_group(group_articles, sensational_keywords, stop_words)

        group_results.append({
            "group_id": group_id,
            "label": group_label,
            "description": group_description,
            "stats": {
                "article_count": stats["article_count"],
                "total_words": stats["total_words"],
                "avg_sentence_length": stats["avg_sentence_len"],
                "exclamations_per_article": stats["exclamation_intensity"],
                "sensational_words_per_article": stats["sensational_score"]
            },
            "top_keywords": [{"word": w, "count": c} for w, c in stats["top_keywords"][:20]],
            "top_phrases": [{"phrase": p, "count": c} for p, c in stats["top_phrases"][:20]],
            "sensational_word_details": dict(stats["sensational_word_details"])
        })

    # Generate word cloud data from first group's keywords (usually primary topic)
    if group_results:
        primary_group = group_results[0]
        max_words = wordcloud_config.get('max_words', 50)

        # Re-analyze first group for full keyword list
        filter_config = comparison_groups[0].get('filter', {})
        primary_articles = filter_articles(all_articles, filter_config)
        primary_stats = analyze_group(primary_articles, sensational_keywords, stop_words)

        word_cloud_data = [
            {"text": word, "size": count}
            for word, count in primary_stats["top_keywords"][:max_words]
        ]
    else:
        word_cloud_data = []

    # Generate sensational language details (aggregate across all groups)
    all_sensational = Counter()
    for result in group_results:
        for word, count in result["sensational_word_details"].items():
            all_sensational[word] += count

    sensational_language = {
        word: {
            "total_uses": count,
            "contexts": []  # Could be expanded later
        }
        for word, count in all_sensational.items()
    }

    # Compile final report
    report = {
        "metadata": {
            "project_name": config.get_project_metadata()['name'],
            "generated_at": datetime.now().isoformat(),
            "config_file": args.config if args.config else "default",
            "comparison_groups_count": len(comparison_groups)
        },
        "summary": {
            "total_articles": len(all_articles),
            "groups_analyzed": len(group_results)
        },
        "comparison_groups": group_results,
        "word_cloud_data": word_cloud_data,
        "sensational_language": sensational_language
    }

    # For backward compatibility, add old field names if exactly 2 groups
    if len(group_results) == 2:
        report["whitechapel_analysis"] = {
            "stats": group_results[0]["stats"],
            "top_keywords": group_results[0]["top_keywords"],
            "top_phrases": group_results[0]["top_phrases"]
        }
        report["general_news_analysis"] = {
            "stats": group_results[1]["stats"],
            "top_keywords": group_results[1]["top_keywords"],
            "top_phrases": group_results[1]["top_phrases"]
        }

    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Print summary
    print("-" * 50)
    print(f"Analysis Complete! Saved to {output_file}")
    print(f"\nResults by group:")
    for result in group_results:
        print(f"  {result['label']}: {result['stats']['article_count']} articles")
        print(f"    Avg sentence length: {result['stats']['avg_sentence_length']}")
        print(f"    Sensational words/article: {result['stats']['sensational_words_per_article']}")


if __name__ == "__main__":
    main()
