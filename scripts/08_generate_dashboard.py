#!/usr/bin/env python3
"""
Dashboard Generator - Complete Implementation
Creates a single-page HTML dashboard with interactive visualizations.
Dark steampunk aesthetic with full functionality.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from config_loader import load_config


def load_data_files(data_dir: Path) -> dict:
    """Load all data files for dashboard."""
    data = {}

    # Required files
    required_files = {
        'tagged_articles': 'tagged_articles.json',
        'timeline': 'timeline.json',
        'text_analysis': 'text_analysis.json',
        'topic_model': 'topic_model.json',
    }

    # Optional files
    optional_files = {
        'entities': 'entities.json',
        'entity_network': 'entity_network.json',
        'entity_cooccurrence_d3': 'entity-cooccurrence_d3.json',
        'temporal_timeline': 'temporal-network_timeline.json'
    }

    # Load required files
    for key, filename in required_files.items():
        file_path = data_dir / filename
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data[key] = json.load(f)
        else:
            print(f"Warning: Required file not found: {filename}")
            data[key] = {}

    # Load optional files
    for key, filename in optional_files.items():
        file_path = data_dir / filename
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data[key] = json.load(f)

    return data


def load_metadata(project_root: Path) -> dict:
    """Load preprocessing metadata to get image paths."""
    metadata_file = project_root / "data" / "preprocessed" / "all_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def generate_dashboard_html(data: dict, config_loader, output_path: Path, project_root: Path):
    """Generate comprehensive HTML dashboard with steampunk theme."""

    project_metadata = config_loader.get_project_metadata()

    # Get project info
    project_name = project_metadata.get('name', 'Newspaper Analysis')
    project_description = project_metadata.get('description', '')

    # Load preprocessing metadata for image paths
    preprocessing_metadata = load_metadata(project_root)
    print(f"Loaded preprocessing metadata: {len(preprocessing_metadata)} PDF(s)")

    # Build article_id to image path mapping (by page and column)
    snippet_map = {}
    for pdf_meta in preprocessing_metadata:
        pdf_stem = pdf_meta.get('source_pdf', '')
        for page in pdf_meta.get('pages', []):
            page_num = page.get('page_num', 0)
            for snip in page.get('snippets', []):
                # Extract snippet path
                snip_path = snip.get('path', '')
                if snip_path:
                    # Convert absolute path to relative from dashboard/index.html
                    from pathlib import Path as P
                    snip_path_obj = P(snip_path)
                    filename = snip_path_obj.name
                    column = snip.get('column', 0)

                    snippet_data = {
                        'filename': filename,
                        'pdf_stem': pdf_stem,
                        'column': column,
                        'relative_path': f"../data/preprocessed/{pdf_stem}/{filename}"
                    }

                    # Store with page+column key for precise matching
                    col_key = f"{pdf_stem}_p{page_num}_c{column}"
                    if col_key not in snippet_map:
                        snippet_map[col_key] = []
                    snippet_map[col_key].append(snippet_data)

                    # Also store with page-only key for fallback
                    page_key = f"{pdf_stem}_p{page_num}"
                    if page_key not in snippet_map:
                        snippet_map[page_key] = []
                    snippet_map[page_key].append(snippet_data)

    print(f"Built snippet map with {len(snippet_map)} keys")
    if len(snippet_map) > 0:
        print(f"Sample keys: {list(snippet_map.keys())[:5]}")

    # Prepare data for embedding - Ensure we have default structures to prevent JS crashes
    timeline_data = json.dumps(data.get('timeline', {}))
    text_analysis_data = json.dumps(data.get('text_analysis', {}))
    tagged_articles_data = json.dumps(data.get('tagged_articles', {"articles": []}))
    entity_network_data = json.dumps(data.get('entity_cooccurrence_d3', {"nodes": [], "links": []}))
    snippet_map_json = json.dumps(snippet_map)
    topic_model_data = json.dumps(data.get('topic_model', {"models": []}))

    # Generate HTML
    # Note: Double curly braces {{ }} are used to escape CSS/JS braces from Python's f-string
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{project_name} - Research Dashboard</title>

    <!-- CSS Libraries -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Crimson+Text:ital,wght@0,400;0,600;0,700;1,400&family=Special+Elite&display=swap" rel="stylesheet">

    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        :root {{
            --bg-primary: #1a1410;
            --bg-secondary: #2d2419;
            --bg-tertiary: #3d3229;
            --accent-brass: #b8860b;
            --accent-copper: #cd7f32;
            --accent-bronze: #8b6f47;
            --text-primary: #e8dcc4;
            --text-secondary: #c9b896;
            --text-muted: #8b7e66;
            --border-color: #5a4a3a;
            --shadow: rgba(0, 0, 0, 0.6);
        }}

        body {{
            font-family: 'Crimson Text', serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            background-image:
                radial-gradient(circle at 20% 50%, rgba(184, 134, 11, 0.03) 0%, transparent 50%),
                radial-gradient(circle at 80% 80%, rgba(205, 127, 50, 0.03) 0%, transparent 50%);
        }}

        .header {{
            background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
            border-bottom: 3px solid var(--accent-brass);
            padding: 2.5rem;
            text-align: center;
            box-shadow: 0 4px 20px var(--shadow);
        }}

        .header h1 {{
            font-family: 'Special Elite', cursive;
            font-size: 3rem;
            color: var(--accent-brass);
            text-shadow: 2px 2px 4px var(--shadow);
            margin-bottom: 0.5rem;
        }}

        .stats-bar {{
            background: var(--bg-secondary);
            border-bottom: 2px solid var(--border-color);
            padding: 1.5rem;
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 1.5rem;
        }}

        .stat {{
            text-align: center;
            padding: 1.2rem;
            background: var(--bg-tertiary);
            border: 2px solid var(--accent-bronze);
            border-radius: 10px;
            min-width: 180px;
            box-shadow: 0 4px 10px var(--shadow);
        }}

        .stat-value {{
            font-family: 'Special Elite', cursive;
            font-size: 2.5rem;
            color: var(--accent-brass);
            display: block;
        }}

        .stat-label {{
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            color: var(--text-muted);
        }}

        .container {{
            max-width: 1600px;
            margin: 0 auto;
            padding: 2rem;
        }}

        .section {{
            background: var(--bg-secondary);
            margin-bottom: 2.5rem;
            border-radius: 12px;
            padding: 2rem;
            border: 2px solid var(--border-color);
            box-shadow: 0 8px 30px var(--shadow);
        }}

        .section-title {{
            font-family: 'Special Elite', cursive;
            font-size: 2rem;
            color: var(--accent-brass);
            border-bottom: 2px solid var(--accent-copper);
            padding-bottom: 0.5rem;
            margin-bottom: 1.5rem;
        }}

        #timelineChart {{
            width: 100%;
            height: 400px;
            background: var(--bg-tertiary);
            border-radius: 8px;
            padding: 1rem;
        }}

        #networkGraph {{
            width: 100%;
            height: 700px;
            background: var(--bg-tertiary);
            border-radius: 8px;
            position: relative;
            overflow: hidden;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-tertiary);
            border-radius: 8px;
            overflow: hidden;
        }}

        th {{
            background: var(--bg-primary);
            color: var(--accent-brass);
            text-transform: uppercase;
            padding: 1rem;
            text-align: left;
            border-bottom: 2px solid var(--accent-bronze);
        }}

        td {{
            padding: 1rem;
            border-bottom: 1px solid var(--border-color);
        }}

        tr:hover {{
            background: rgba(184, 134, 11, 0.1);
            cursor: pointer;
        }}

        .tag-badge {{
            background: var(--bg-primary);
            color: var(--text-secondary);
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.8rem;
            border: 1px solid var(--accent-bronze);
            margin-right: 4px;
        }}

        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
            overflow-y: auto;
        }}

        .modal-content {{
            background: var(--bg-secondary);
            margin: 2% auto;
            padding: 2.5rem;
            width: 85%;
            max-width: 1100px;
            border: 3px solid var(--accent-brass);
            border-radius: 10px;
            position: relative;
        }}

        .close {{
            position: absolute;
            right: 20px;
            top: 10px;
            font-size: 2.5rem;
            color: var(--accent-brass);
            cursor: pointer;
        }}

        .article-image {{
            max-width: 100%;
            border: 2px solid var(--border-color);
            margin: 1.5rem 0;
            display: block;
        }}

        .tooltip {{
            position: absolute;
            background: var(--bg-primary);
            border: 1px solid var(--accent-brass);
            padding: 10px;
            pointer-events: none;
            opacity: 0;
            z-index: 100;
            box-shadow: 0 4px 10px black;
        }}

        /* Floating Notepad Styles */
        #floatingNotepad {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 350px;
            background: var(--bg-secondary);
            border: 2px solid var(--accent-brass);
            border-radius: 8px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.5);
            z-index: 1000;
            display: none;
        }}

        #floatingNotepad.minimized {{
            height: 50px;
            overflow: hidden;
        }}

        .notepad-header {{
            background: linear-gradient(135deg, var(--accent-brass), var(--accent-bronze));
            padding: 12px 15px;
            cursor: move;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-radius: 6px 6px 0 0;
        }}

        .notepad-title {{
            font-family: 'Special Elite', monospace;
            font-weight: bold;
            color: var(--bg-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .notepad-controls {{
            display: flex;
            gap: 8px;
        }}

        .notepad-btn {{
            background: transparent;
            border: none;
            color: var(--bg-primary);
            cursor: pointer;
            font-size: 16px;
            padding: 4px 8px;
            border-radius: 3px;
            transition: background 0.2s;
        }}

        .notepad-btn:hover {{
            background: rgba(0,0,0,0.2);
        }}

        .notepad-content {{
            padding: 15px;
        }}

        #researchNotes {{
            width: 100%;
            min-height: 200px;
            background: var(--bg-primary);
            color: var(--text-primary);
            border: 1px solid var(--accent-bronze);
            border-radius: 5px;
            padding: 10px;
            font-family: 'Crimson Text', serif;
            font-size: 14px;
            line-height: 1.6;
            resize: vertical;
        }}

        #researchNotes:focus {{
            outline: none;
            border-color: var(--accent-brass);
        }}

        .notepad-footer {{
            padding: 10px 15px;
            background: var(--bg-tertiary);
            border-radius: 0 0 6px 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 12px;
            color: var(--text-muted);
        }}

        #notepadToggle {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--accent-brass), var(--accent-copper));
            border: 3px solid var(--accent-bronze);
            color: var(--bg-primary);
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 999;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        #notepadToggle:hover {{
            transform: scale(1.1);
            box-shadow: 0 6px 20px rgba(184, 134, 11, 0.5);
        }}

        #notepadToggle.hidden {{
            display: none;
        }}
    
    .topic-grid {{
    display: grid; 
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); 
    gap: 1rem; 
    margin-top: 1rem; 
}}
.topic-card {{ 
    background: var(--bg-tertiary); 
    padding: 1rem; 
    border: 1px solid var(--accent-brass); 
    border-radius: 4px;
    box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
}}
.topic-card h4 {{ 
    margin: 0 0 0.5rem 0; 
    color: var(--accent-copper); 
    font-family: 'Special Elite';
    font-size: 0.9rem;
}}
.word-pill {{ 
    display: inline-block; 
    background: rgba(184, 134, 11, 0.1); 
    padding: 2px 8px; 
    border-radius: 10px; 
    margin: 2px; 
    font-size: 0.8rem; 
    border: 1px solid var(--accent-bronze);
    color: var(--text-secondary);
}}
.topic-controls {{ margin-bottom: 1.5rem; }}
select#groupSelector {{
    background: var(--bg-tertiary);
    color: var(--accent-brass);
    border: 1px solid var(--accent-brass);
    padding: 8px;
    font-family: 'Special Elite';
}}    
    </style>
</head>
<body>
    <div class="header">
        <h1><i class="fas fa-cog fa-spin" style="font-size: 0.8em; vertical-align: middle;"></i> {project_name}</h1>
        <p>{project_description}</p>
    </div>

    <div class="stats-bar">
        <div class="stat">
            <span class="stat-value" id="totalArticles">0</span>
            <span class="stat-label">Articles</span>
        </div>
        <div class="stat">
            <span class="stat-value" id="totalEntities">0</span>
            <span class="stat-label">Entities</span>
        </div>
        <div class="stat">
            <span class="stat-value" id="timeSpan">-</span>
            <span class="stat-label">Time Span</span>
        </div>
    </div>

    <div class="container">
        <!-- Timeline Section -->
        <div class="section">
            <h2 class="section-title"><i class="fas fa-history"></i> Temporal Distribution</h2>
            <div id="timelineChart"></div>
        </div>

        <!-- Entity Network Section -->
        <div class="section">
            <h2 class="section-title"><i class="fas fa-project-diagram"></i> Entity Connections</h2>
            <div id="networkGraph"></div>
            <div id="networkLegend" style="margin-top:10px; display:flex; gap:15px; flex-wrap:wrap;"></div>
        </div>
    <div class="section">
    <h2 class="section-title"><i class="fas fa-gears"></i> Thematic Landscapes</h2>
        <div class="topic-controls">
        <label style="font-family:'Special Elite'; color:var(--text-muted);">Switch Analysis Group: </label>
        <select id="groupSelector" onchange="updateTopicDisplay()"></select>
        </div>
    
        <div style="display: grid; grid-template-columns: 1fr 2fr; gap: 20px;">
        <div id="topicList" class="topic-grid"></div>
        <div style="background: var(--bg-tertiary); padding: 1.5rem; border-radius: 8px; border: 1px solid var(--border-color);">
            <h4 style="font-family:'Special Elite'; color:var(--accent-brass); margin-top:0;">Thematic Evolution Over Time</h4>
            <div style="height: 350px;"><canvas id="topicTimeline"></canvas></div>
        </div>
    </div>
</div>
        <!-- Article Browser Section -->
        <div class="section">
            <h2 class="section-title"><i class="fas fa-search"></i> Archive Explorer</h2>
            <input type="text" id="articleSearch" placeholder="Filter by keyword or headline..." 
                   style="width: 100%; padding: 12px; margin-bottom: 20px; background: var(--bg-tertiary); color: white; border: 2px solid var(--border-color); border-radius: 5px;">
            <div id="articleTable"></div>
        </div>
    </div>
    <!-- Article Detail Modal -->
    <div id="articleModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal()">&times;</span>
            <div id="modalBody"></div>
        </div>
    </div>

    <!-- Floating Notepad -->
    <button id="notepadToggle" onclick="toggleNotepad()" title="Research Notes">
        <i class="fas fa-sticky-note"></i>
    </button>

    <div id="floatingNotepad">
        <div class="notepad-header" id="notepadHeader">
            <div class="notepad-title">
                <i class="fas fa-feather-alt"></i>
                Research Notes
            </div>
            <div class="notepad-controls">
                <button class="notepad-btn" onclick="minimizeNotepad()" title="Minimize">
                    <i class="fas fa-minus"></i>
                </button>
                <button class="notepad-btn" onclick="toggleNotepad()" title="Close">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        </div>
        <div class="notepad-content">
            <textarea id="researchNotes" placeholder="Document your observations and insights here..."></textarea>
        </div>
        <div class="notepad-footer">
            <span id="noteCount">0 characters</span>
            <button class="notepad-btn" onclick="saveNotes()" title="Save to localStorage" style="color: var(--accent-brass);">
                <i class="fas fa-save"></i> Save
            </button>
        </div>
    </div>

    <!-- JavaScript Libraries -->
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>

    <script>
        // Data Injection from Python
        const timelineData = {timeline_data};
        const taggedArticlesData = {tagged_articles_data};
        const entityNetworkData = {entity_network_data};
        const snippetMap = {snippet_map_json};
        const topicModelData = {topic_model_data}; // New Topic Model Data

        let topicChart = null; // Global reference for the topic chart

        document.addEventListener('DOMContentLoaded', () => {{
            console.log("Dashboard Loading...");
            
            // Initialization with error boundaries
            try {{ initializeStats(); }} catch(e) {{ console.error("Stats fail:", e); }}
            try {{ initializeTimeline(); }} catch(e) {{ console.error("Timeline fail:", e); }}
            try {{ initializeNetwork(); }} catch(e) {{ console.error("Network fail:", e); }}
            try {{ initializeTopicControls(); }} catch(e) {{ console.error("Topic modeling fail:", e); }}
            try {{ initializeArticleBrowser(); }} catch(e) {{ console.error("Browser fail:", e); }}
        }});

        function initializeStats() {{
            const articles = taggedArticlesData.articles || [];
            document.getElementById('totalArticles').textContent = articles.length;
            document.getElementById('totalEntities').textContent = (entityNetworkData.nodes || []).length;
            
            if (articles.length > 0) {{
                const dates = articles.map(a => a.date).filter(d => d).sort();
                if (dates.length > 0) {{
                    const startYear = new Date(dates[0]).getFullYear();
                    const endYear = new Date(dates[dates.length-1]).getFullYear();
                    document.getElementById('timeSpan').textContent = 
                        startYear === endYear ? startYear : `${{startYear}}-${{endYear}}`;
                }}
            }}
        }}

        // ================ TOPIC MODELING FUNCTIONS ================

        function initializeTopicControls() {{
            const selector = document.getElementById('groupSelector');
            if (!topicModelData.models || topicModelData.models.length === 0) {{
                console.log("No topic models found in data.");
                return;
            }}

            // Populate the dropdown with comparison groups
            topicModelData.models.forEach(m => {{
                const opt = document.createElement('option');
                opt.value = m.group_id;
                opt.textContent = m.group_label;
                selector.appendChild(opt);
            }});

            // Initial render
            updateTopicDisplay();
        }}

        function updateTopicDisplay() {{
            const selector = document.getElementById('groupSelector');
            if (!selector) return;
            
            const groupId = selector.value;
            const model = topicModelData.models.find(m => m.group_id === groupId);
            if (!model) return;

            // 1. Render Topic Cards (Word Pills)
            const listContainer = document.getElementById('topicList');
            listContainer.innerHTML = '';
            
            model.topics.forEach(t => {{
                const card = document.createElement('div');
                card.className = 'topic-card';
                
                // Take top 8 words for the pill display
                const words = t.top_words.slice(0, 8).map(w => 
                    `<span class="word-pill" title="Weight: ${{w.weight.toFixed(4)}}">${{w.word}}</span>`
                ).join('');
                
                card.innerHTML = `<h4>THEME ${{t.topic_id + 1}}</h4>${{words}}`;
                listContainer.appendChild(card);
            }});

            // 2. Update the Temporal Topic Chart
            renderTopicTimeline(model);
        }}

        function renderTopicTimeline(model) {{
            const ctx = document.getElementById('topicTimeline').getContext('2d');
            if (topicChart) topicChart.destroy();

            // Sort months/dates for the X axis
            const labels = Object.keys(model.temporal_distribution).sort();
            
            // Steampunk color palette for topics
            const colors = [
                '#b8860b', '#cd7f32', '#8b6f47', '#5a4a3a', '#e8dcc4', 
                '#4682B4', '#8B0000', '#228B22', '#DAA520', '#A9A9A9'
            ];

            const datasets = model.topics.map((t, idx) => {{
                return {{
                    label: `Theme ${{t.topic_id + 1}}`,
                    data: labels.map(l => model.temporal_distribution[l][t.topic_id] || 0),
                    borderColor: colors[idx % colors.length],
                    backgroundColor: colors[idx % colors.length] + '33', // 20% opacity
                    fill: true,
                    tension: 0.4
                }};
            }});

            topicChart = new Chart(ctx, {{
                type: 'line',
                data: {{ labels: labels, datasets: datasets }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        y: {{ 
                            stacked: true, 
                            beginAtZero: true, 
                            grid: {{ color: '#2d2419' }},
                            ticks: {{ color: '#8b7e66' }}
                        }},
                        x: {{ 
                            grid: {{ color: '#2d2419' }},
                            ticks: {{ color: '#8b7e66' }}
                        }}
                    }},
                    plugins: {{ 
                        legend: {{ 
                            position: 'bottom',
                            labels: {{ color: '#e8dcc4', font: {{ family: 'Crimson Text', size: 12 }} }} 
                        }} 
                    }}
                }}
            }});
        }}

        // ================ EXISTING CORE FUNCTIONS ================

        function initializeTimeline() {{
            const container = document.getElementById('timelineChart');
            const rawArticles = taggedArticlesData.articles || [];
            
            if (rawArticles.length === 0) {{
                container.innerHTML = "<p style='text-align:center; padding-top:50px;'>No date data available.</p>";
                return;
            }}

            const dayCounts = {{}};
            rawArticles.forEach(a => {{
                if(a.date) {{
                    const d = a.date.split('T')[0];
                    dayCounts[d] = (dayCounts[d] || 0) + 1;
                }}
            }});

            const chartData = Object.keys(dayCounts).sort().map(d => ({{ x: d, y: dayCounts[d] }}));
            const canvas = document.createElement('canvas');
            container.appendChild(canvas);

            new Chart(canvas, {{
                type: 'line',
                data: {{
                    datasets: [{{
                        label: 'Articles per Day',
                        data: chartData,
                        borderColor: '#cd7f32',
                        backgroundColor: 'rgba(205, 127, 50, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {{
                        x: {{ type: 'time', time: {{ unit: 'month' }}, ticks: {{ color: '#c9b896' }}, grid: {{ color: '#3d3229' }} }},
                        y: {{ beginAtZero: true, ticks: {{ color: '#c9b896', stepSize: 1 }}, grid: {{ color: '#3d3229' }} }}
                    }},
                    plugins: {{ legend: {{ labels: {{ color: '#e8dcc4', font: {{ family: 'Crimson Text' }} }} }} }}
                }}
            }});
        }}

        function initializeNetwork() {{
            const nodes = entityNetworkData.nodes || [];
            const links = entityNetworkData.links || [];
            const container = document.getElementById('networkGraph');
            
            if (nodes.length === 0) {{
                container.innerHTML = "<p style='text-align:center; padding-top:100px;'>Run entity extraction to populate network.</p>";
                return;
            }}

            const width = container.offsetWidth;
            const height = container.offsetHeight;
            const svg = d3.select("#networkGraph").append("svg").attr("width", "100%").attr("height", "100%").attr("viewBox", `0 0 ${{width}} ${{height}}`);
            const g = svg.append("g");
            svg.call(d3.zoom().scaleExtent([0.1, 8]).on("zoom", (e) => g.attr("transform", e.transform)));

            const simulation = d3.forceSimulation(nodes)
                .force("link", d3.forceLink(links).id(d => d.id).distance(100))
                .force("charge", d3.forceManyBody().strength(-400))
                .force("center", d3.forceCenter(width / 2, height / 2))
                .force("collision", d3.forceCollide().radius(25));

            const link = g.append("g").selectAll("line").data(links).enter().append("line")
                .attr("stroke", "#8b6f47").attr("stroke-opacity", 0.3).attr("stroke-width", d => Math.sqrt(d.weight || 1));

            const node = g.append("g").selectAll("g").data(nodes).enter().append("g")
                .call(d3.drag().on("start", dragStarted).on("drag", dragged).on("end", dragEnded));

            node.append("circle").attr("r", d => d.size || 10).attr("fill", d => d.color || "#b8860b").attr("stroke", "#1a1410").attr("stroke-width", 2);
            node.append("text").text(d => d.name).attr("fill", "#e8dcc4").attr("font-size", "10px").attr("dx", 12).attr("dy", 4).style("pointer-events", "none").style("text-shadow", "1px 1px 2px black");

            simulation.on("tick", () => {{
                link.attr("x1", d => d.source.x).attr("y1", d => d.source.y).attr("x2", d => d.target.x).attr("y2", d => d.target.y);
                node.attr("transform", d => `translate(${{d.x}}, ${{d.y}})`);
            }});

            function dragStarted(e, d) {{ if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }}
            function dragged(e, d) {{ d.fx = e.x; d.fy = e.y; }}
            function dragEnded(e, d) {{ if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; }}
        }}

        function initializeArticleBrowser() {{
            const articles = taggedArticlesData.articles || [];
            renderTable(articles);

            document.getElementById('articleSearch').addEventListener('input', (e) => {{
                const term = e.target.value.toLowerCase();
                const filtered = articles.filter(a => 
                    (a.headline || "").toLowerCase().includes(term) || 
                    (a.full_text || "").toLowerCase().includes(term)
                );
                renderTable(filtered);
            }});
        }}

        function renderTable(list) {{
            const container = document.getElementById('articleTable');
            if(list.length === 0) {{ container.innerHTML = "No matching articles found."; return; }}

            let html = `<table><thead><tr><th>Date</th><th>Headline</th><th>Categories</th></tr></thead><tbody>`;
            list.slice(0, 100).forEach(a => {{
                const tags = (a.tags || []).map(t => `<span class="tag-badge">${{t.tag}}</span>`).join('');
                html += `<tr onclick='showArticleDetail(${{JSON.stringify(a).replace(/'/g, "&apos;")}})'>
                    <td>${{a.date || "-"}}</td>
                    <td><strong>${{a.headline || "Untitled"}}</strong></td>
                    <td>${{tags}}</td>
                </tr>`;
            }});
            html += `</tbody></table>`;
            container.innerHTML = html;
        }}

        function showArticleDetail(article) {{
            const modal = document.getElementById('articleModal');
            const body = document.getElementById('modalBody');
            
            let imgHtml = "";
            const column = article.column !== undefined ? article.column : 0;
            const snippetKey = `${{article.source_pdf}}_p${{article.page_number}}_c${{column}}`;
            const pageKey = `${{article.source_pdf}}_p${{article.page_number}}`;

            const snippets = snippetMap[snippetKey] || snippetMap[pageKey];
            if (snippets && snippets.length > 0) {{
                imgHtml = '<div style="margin:15px 0;">';
                snippets.forEach(snip => {{
                    imgHtml += `<img src="${{snip.relative_path}}" class="article-image" style="max-width:100%; height:auto; margin-bottom:10px; border:2px solid var(--accent-bronze); border-radius:5px; display:block;">`;
                }});
                imgHtml += '</div>';
            }} else {{
                imgHtml = `<div style="padding:15px; background:var(--bg-tertiary); color:var(--text-muted);"><i class="fas fa-image"></i> Snippet image not available.</div>`;
            }}

            body.innerHTML = `
                <h2 style="color:var(--accent-brass); font-family:'Special Elite';">${{article.headline || "Untitled"}}</h2>
                <div style="display:grid; grid-template-columns: 400px 1fr; gap:20px;">
                    <div>${{imgHtml}}</div>
                    <div style="background:var(--bg-primary); padding:20px; white-space:pre-wrap; min-height:400px;">${{article.full_text || "Text content missing."}}</div>
                </div>
            `;
            modal.style.display = "block";
        }}

        function closeModal() {{ document.getElementById('articleModal').style.display = "none"; }}

        // ================ NOTEPAD FUNCTIONS ================
        function toggleNotepad() {{
            const notepad = document.getElementById('floatingNotepad');
            const toggle = document.getElementById('notepadToggle');
            if (notepad.style.display === 'none' || notepad.style.display === '') {{
                notepad.style.display = 'block';
                toggle.classList.add('hidden');
                loadNotes();
            }} else {{
                notepad.style.display = 'none';
                toggle.classList.remove('hidden');
            }}
        }}

        function minimizeNotepad() {{ document.getElementById('floatingNotepad').classList.toggle('minimized'); }}
        function saveNotes() {{ localStorage.setItem('researchNotes', document.getElementById('researchNotes').value); alert('Notes saved!'); }}
        function loadNotes() {{ document.getElementById('researchNotes').value = localStorage.getItem('researchNotes') || ""; updateNoteCount(); }}
        function updateNoteCount() {{ document.getElementById('noteCount').textContent = `${{document.getElementById('researchNotes').value.length}} characters`; }}

        window.onclick = function(event) {{ if (event.target == document.getElementById('articleModal')) closeModal(); }}
    </script>

</body>
</html>
"""

    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Dashboard successfully generated at: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate interactive HTML dashboard")
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--output', type=str, default=None, help='Output HTML path')
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    data_dir = project_root / "data" / "processed"
    
    # Load project config
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        return

    output_path = Path(args.output) if args.output else project_root / "dashboard" / "index.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    data = load_data_files(data_dir)

    print("Building steampunk dashboard...")
    generate_dashboard_html(data, config, output_path, project_root)

    print(f"\nDashboard ready! Open in browser via local server or:")
    print(f"file://{output_path.absolute()}")


if __name__ == "__main__":
    main()