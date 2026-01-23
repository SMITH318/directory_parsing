#!/usr/bin/env python3
"""
Step 5: Timeline Generation
- Correlates reference events with local newspaper reporting.
- Calculates news "lag" (information delay).
- Now supports configurable timeline events and correlation tags.
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
from config_loader import load_config


def calculate_time_lag(event_date_str, publication_date_str):
    """Calculate days between reference event and local publication."""
    try:
        event_date = datetime.strptime(event_date_str, "%Y-%m-%d")
        pub_date = datetime.strptime(publication_date_str, "%Y-%m-%d")
        return (pub_date - event_date).days
    except Exception:
        return None


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate timeline correlating reference events with local publications"
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

    # Get timeline configuration
    timeline_config = config.get_timeline_config()
    reference_events = timeline_config.get('reference_events', [])
    correlation_tags = timeline_config.get('correlation_tags', [])
    local_label = timeline_config.get('local_label', 'Local Publications')
    reference_label = timeline_config.get('reference_label', 'Reference Events')

    print(f"Loaded configuration: {config.get_project_metadata()['name']}")
    print(f"Reference events: {len(reference_events)}")
    print(f"Correlation tags: {correlation_tags if correlation_tags else 'all articles'}")

    # Setup paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    input_file = project_root / "data" / "processed" / "tagged_articles.json"
    output_file = project_root / "data" / "processed" / "timeline.json"

    if not input_file.exists():
        print(f"Error: {input_file} not found. Run tagging first.")
        return

    # Load tagged articles
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    articles = data.get("articles", [])

    # Filter articles based on correlation tags
    if correlation_tags:
        # Filter to articles with any of the correlation tags
        filtered_articles = []
        for art in articles:
            art_tags = [t["tag"] for t in art.get("tags", [])]
            if any(tag in correlation_tags for tag in art_tags):
                filtered_articles.append(art)
        print(f"Filtered to {len(filtered_articles)} articles matching correlation tags")
    else:
        # Use all articles if no correlation tags specified
        filtered_articles = articles
        print(f"Using all {len(filtered_articles)} articles for timeline")

    local_publications = []

    for art in filtered_articles:
        date_iso = art.get("date")
        if not date_iso:
            continue

        # Display formatting (e.g., Oct 25, 1888)
        try:
            dt = datetime.strptime(date_iso, "%Y-%m-%d")
            date_display = dt.strftime("%b %d, %Y")
        except:
            date_display = date_iso

        # Get the text using new or old field names
        text = art.get("full_text", art.get("text", ""))

        event = {
            "article_id": art["article_id"],
            "pub": art.get("source_pdf", art.get("pub")),
            "date": date_iso,
            "date_display": date_display,
            "headline": art.get("headline", "Untitled Report"),
            "snippet": text[:200] + "..." if len(text) > 200 else text,
            "page": art.get("page_number", art.get("page")),
            "column": art.get("column"),
            "tags": [t["tag"] for t in art.get("tags", [])],
            "type": "publication"
        }

        # Calculate lag if reference events exist
        if reference_events:
            # Find the most recent reference event preceding this publication
            preceding_events = [
                e for e in reference_events
                if e["date"] <= date_iso
            ]

            if preceding_events:
                # The "trigger" event is the one closest to the publication date
                trigger = sorted(preceding_events, key=lambda x: x["date"])[-1]
                lag = calculate_time_lag(trigger["date"], date_iso)

                event["correlated_event"] = trigger["title"]
                event["correlated_event_id"] = trigger.get("id")
                event["days_since_event"] = lag
                event["historical_context"] = f"Reported {lag} days after {trigger['title']}."

        local_publications.append(event)

    # Compile Final Timeline
    timeline_data = {
        "metadata": {
            "project_name": config.get_project_metadata()['name'],
            "generated_at": datetime.now().isoformat(),
            "config_file": args.config if args.config else "default",
            "has_reference_events": len(reference_events) > 0,
            "reference_label": reference_label,
            "local_label": local_label
        },
        "reference_events": reference_events,
        "local_publications": sorted(local_publications, key=lambda x: x['date']),
        "statistics": {
            "total_publications": len(local_publications),
            "total_reference_events": len(reference_events),
            "correlation_tags": correlation_tags
        }
    }

    # Calculate lag statistics if reference events exist
    if reference_events and local_publications:
        lags = [
            e["days_since_event"]
            for e in local_publications
            if "days_since_event" in e
        ]
        if lags:
            timeline_data["statistics"]["avg_lag_days"] = round(sum(lags) / len(lags), 1)
            timeline_data["statistics"]["min_lag_days"] = min(lags)
            timeline_data["statistics"]["max_lag_days"] = max(lags)

    # Save
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(timeline_data, f, indent=2, ensure_ascii=False)

    # Analysis Summary
    print("-" * 50)
    print(f"Timeline Generated: {output_file}")
    print(f"Total {local_label.lower()}: {len(local_publications)}")

    if reference_events and local_publications:
        lags = [
            e["days_since_event"]
            for e in local_publications
            if "days_since_event" in e
        ]
        if lags:
            avg_lag = sum(lags) / len(lags)
            print(f"Average information lag: {avg_lag:.1f} days")
            print(f"Range: {min(lags)} to {max(lags)} days")
    elif not reference_events:
        print("No reference events defined - generated simple publication timeline")


if __name__ == "__main__":
    main()
