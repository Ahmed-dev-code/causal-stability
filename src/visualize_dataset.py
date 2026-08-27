import json
import html
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/mavenere_subset_300.json")
OUTPUT_FILE = Path("data/mavenere_causal_explorer.html")


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both:
    # 1. A JSON list
    # 2. {"documents": [...]}
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if "documents" in data:
            return data["documents"]

        # In case the file contains one document
        if "story_id" in data:
            return [data]

    raise ValueError("Unsupported JSON format.")


# ============================================================
# PREPARE STORIES
# ============================================================

def prepare_stories(documents):
    stories = []

    for doc in documents:
        story_id = doc.get("story_id", doc.get("id", "unknown"))
        title = doc.get("title", story_id)
        text = doc.get("text", "")

        gold_graph = doc.get("gold_graph", {})
        gold_relations = gold_graph.get("relations", [])

        maven_graph = doc.get("maven_graph", {})
        maven_relations = maven_graph.get("relations", [])

        metadata = doc.get("metadata", {})

        # ----------------------------------------------------
        # Extract events from MAVEN graph
        # ----------------------------------------------------

        events = {}

        for rel in maven_relations:
            source_id = rel.get("source_event_id")
            target_id = rel.get("target_event_id")

            if source_id:
                events[source_id] = {
                    "id": source_id,
                    "trigger": rel.get("source_trigger", source_id),
                    "type": rel.get("source_event_type", ""),
                    "sent_id": rel.get("source_sent_id", -1),
                }

            if target_id:
                events[target_id] = {
                    "id": target_id,
                    "trigger": rel.get("target_trigger", target_id),
                    "type": rel.get("target_event_type", ""),
                    "sent_id": rel.get("target_sent_id", -1),
                }

        # ----------------------------------------------------
        # Build graph relations
        # ----------------------------------------------------

        relations = []

        seen = set()

        for rel in maven_relations:
            source = rel.get("source_event_id")
            target = rel.get("target_event_id")
            relation_type = rel.get("relation_type", "UNKNOWN")

            if not source or not target:
                continue

            # Remove accidental duplicates
            key = (source, target, relation_type)

            if key in seen:
                continue

            seen.add(key)

            relations.append({
                "source": source,
                "target": target,
                "type": relation_type,
                "distance": rel.get("sentence_distance", 0),
            })

        # ----------------------------------------------------
        # Get sentences
        # ----------------------------------------------------

        sentences = doc.get("sentences")

        if not sentences:
            sentences = split_sentences(text)

        stories.append({
            "id": story_id,
            "title": title,
            "text": text,
            "sentences": sentences,
            "events": list(events.values()),
            "relations": relations,
            "gold_relations": gold_relations,
            "metadata": metadata,
        })

    return stories


def split_sentences(text):
    """
    Simple fallback sentence splitter.
    The original MAVEN data normally already has sentence information.
    """
    import re

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    return [s for s in sentences if s]


# ============================================================
# HTML / JAVASCRIPT
# ============================================================

def generate_html(stories):

    # JSON safely embedded into JavaScript
    stories_json = json.dumps(
        stories,
        ensure_ascii=False
    ).replace("</", "<\\/")

    return f"""<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>MAVEN-ERE Causal Graph Explorer</title>

<style>

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Arial,
        sans-serif;

    background: #f5f7fa;
    color: #1f2937;
}}

header {{
    background: #111827;
    color: white;
    padding: 18px 24px;
}}

header h1 {{
    margin: 0;
    font-size: 22px;
}}

header p {{
    margin: 5px 0 0;
    color: #cbd5e1;
    font-size: 13px;
}}

.container {{
    display: grid;
    grid-template-columns: 280px 1fr;
    height: calc(100vh - 76px);
}}

.sidebar {{
    background: white;
    border-right: 1px solid #e5e7eb;
    overflow-y: auto;
    padding: 15px;
}}

.search {{
    width: 100%;
    padding: 9px;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    margin-bottom: 12px;
}}

.story-list {{
    display: flex;
    flex-direction: column;
    gap: 4px;
}}

.story-item {{
    padding: 9px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 13px;
}}

.story-item:hover {{
    background: #f3f4f6;
}}

.story-item.active {{
    background: #e5e7eb;
    font-weight: 600;
}}

.main {{
    overflow-y: auto;
    padding: 22px;
}}

.story-header {{
    margin-bottom: 15px;
}}

.story-title {{
    font-size: 25px;
    font-weight: 700;
    margin-bottom: 5px;
}}

.story-id {{
    font-size: 12px;
    color: #6b7280;
}}

.stats {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 15px 0;
}}

.stat {{
    background: white;
    border: 1px solid #e5e7eb;
    padding: 9px 13px;
    border-radius: 7px;
    font-size: 13px;
}}

.stat strong {{
    display: block;
    font-size: 17px;
}}

.card {{
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 9px;
    padding: 18px;
    margin-bottom: 18px;
}}

.card h2 {{
    font-size: 17px;
    margin-top: 0;
}}

.story-text {{
    line-height: 1.8;
    font-size: 15px;
}}

.sentence {{
    padding: 2px 3px;
    border-radius: 3px;
    transition: background 0.2s;
}}

.sentence.highlight {{
    background: #fef08a;
}}

.graph-container {{
    width: 100%;
    overflow: auto;
    background: #fafafa;
    border: 1px solid #e5e7eb;
    border-radius: 7px;
    min-height: 450px;
}}

svg {{
    display: block;
    min-width: 900px;
    min-height: 430px;
}}

.node {{
    cursor: pointer;
}}

.node-box {{
    fill: white;
    stroke: #374151;
    stroke-width: 2;
}}

.node.selected .node-box {{
    stroke: #2563eb;
    stroke-width: 4;
}}

.node-label {{
    font-size: 13px;
    font-weight: 600;
    pointer-events: none;
}}

.node-type {{
    font-size: 10px;
    fill: #6b7280;
    pointer-events: none;
}}

.edge {{
    stroke: #6b7280;
    stroke-width: 2;
    fill: none;
}}

.edge.cause {{
    stroke-width: 3;
}}

.edge-label {{
    font-size: 11px;
    fill: #374151;
}}

.legend {{
    display: flex;
    gap: 20px;
    font-size: 12px;
    margin-top: 10px;
}}

.legend-line {{
    display: inline-block;
    width: 30px;
    height: 3px;
    vertical-align: middle;
    margin-right: 5px;
    background: #6b7280;
}}

.legend-line.cause {{
    height: 4px;
}}

.relation {{
    padding: 8px 10px;
    border-bottom: 1px solid #eee;
    font-size: 13px;
}}

.badge {{
    display: inline-block;
    padding: 2px 7px;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 600;
    margin: 0 4px;
}}

.badge.cause {{
    background: #fee2e2;
}}

.badge.precondition {{
    background: #dbeafe;
}}

button {{
    border: 1px solid #d1d5db;
    background: white;
    border-radius: 5px;
    padding: 7px 12px;
    cursor: pointer;
}}

button:hover {{
    background: #f3f4f6;
}}

.navigation {{
    display: flex;
    gap: 8px;
    margin-bottom: 15px;
}}

.empty {{
    color: #6b7280;
    padding: 20px;
    text-align: center;
}}

@media (max-width: 800px) {{

    .container {{
        grid-template-columns: 1fr;
        height: auto;
    }}

    .sidebar {{
        max-height: 300px;
        border-right: none;
        border-bottom: 1px solid #ddd;
    }}

}}

</style>

</head>

<body>

<header>

<h1>MAVEN-ERE Causal Graph Explorer</h1>

<p>
Interactive visualization of the 300-story causal-stability dataset
</p>

</header>


<div class="container">


<!-- ======================================================
     SIDEBAR
======================================================= -->

<aside class="sidebar">

<input
    id="search"
    class="search"
    placeholder="Search stories..."
    oninput="filterStories()"
>

<div id="storyList" class="story-list"></div>

</aside>


<!-- ======================================================
     MAIN
======================================================= -->

<main class="main">

<div class="navigation">

<button onclick="previousStory()">← Previous</button>

<button onclick="nextStory()">Next →</button>

</div>


<div id="content"></div>

</main>

</div>


<script>

const stories = {stories_json};

let currentIndex = 0;


// ========================================================
// INITIALIZE
// ========================================================

function init() {{

    renderStoryList();

    if (stories.length > 0) {{
        renderStory(0);
    }}

}}


// ========================================================
// STORY LIST
// ========================================================

function renderStoryList() {{

    const list = document.getElementById("storyList");

    const search =
        document
            .getElementById("search")
            .value
            .toLowerCase();

    list.innerHTML = "";

    stories.forEach((story, index) => {{

        const text =
            (story.title + " " + story.id)
            .toLowerCase();

        if (!text.includes(search)) {{
            return;
        }}

        const item =
            document.createElement("div");

        item.className =
            "story-item" +
            (index === currentIndex ? " active" : "");

        item.textContent =
            story.title || story.id;

        item.onclick = () => renderStory(index);

        list.appendChild(item);

    }});

}}


function filterStories() {{
    renderStoryList();
}}


// ========================================================
// STORY RENDERING
// ========================================================

function renderStory(index) {{

    if (index < 0 || index >= stories.length) {{
        return;
    }}

    currentIndex = index;

    const story = stories[index];

    renderStoryList();

    const content =
        document.getElementById("content");

    const tokens =
        story.text.trim().split(/\\s+/).length;

    const relations =
        story.relations.length;

    const causeCount =
        story.relations.filter(
            r => r.type === "CAUSE"
        ).length;

    const preconditionCount =
        story.relations.filter(
            r => r.type === "PRECONDITION"
        ).length;

    const maxDistance =
        story.relations.length
        ? Math.max(
            ...story.relations.map(
                r => Number(r.distance || 0)
            )
        )
        : 0;


    // ----------------------------------------------------
    // Story text
    // ----------------------------------------------------

    let textHTML = "";

    story.sentences.forEach((sentence, i) => {{

        textHTML +=
            `<span
                id="sentence-${{i}}"
                class="sentence"
            >
                [${{i}}] ${{escapeHTML(sentence)}}
            </span> `;

    }});


    // ----------------------------------------------------
    // Relation list
    // ----------------------------------------------------

    let relationHTML = "";

    story.relations.forEach((rel, i) => {{

        const source =
            story.events.find(
                e => e.id === rel.source
            );

        const target =
            story.events.find(
                e => e.id === rel.target
            );

        const sourceName =
            source ? source.trigger : rel.source;

        const targetName =
            target ? target.trigger : rel.target;

        const badgeClass =
            rel.type.toLowerCase();

        relationHTML += `
            <div class="relation">

                <strong>${{escapeHTML(sourceName)}}</strong>

                <span class="badge ${{badgeClass}}">
                    ${{escapeHTML(rel.type)}}
                </span>

                <strong>${{escapeHTML(targetName)}}</strong>

                <span style="color:#777">
                    (distance: ${{rel.distance}})
                </span>

            </div>
        `;

    }});


    content.innerHTML = `

        <section class="story-header">

            <div class="story-title">
                ${{escapeHTML(story.title)}}
            </div>

            <div class="story-id">
                ${{escapeHTML(story.id)}}
            </div>

        </section>


        <section class="stats">

            <div class="stat">
                <strong>${{tokens}}</strong>
                Tokens
            </div>

            <div class="stat">
                <strong>${{story.sentences.length}}</strong>
                Sentences
            </div>

            <div class="stat">
                <strong>${{relations}}</strong>
                Relations
            </div>

            <div class="stat">
                <strong>${{causeCount}}</strong>
                CAUSE
            </div>

            <div class="stat">
                <strong>${{preconditionCount}}</strong>
                PRECONDITION
            </div>

            <div class="stat">
                <strong>${{maxDistance}}</strong>
                Max distance
            </div>

            <div class="stat">
                <strong>${{story.events.length}}</strong>
                Graph nodes
            </div>

        </section>


        <section class="card">

            <h2>Story</h2>

            <div class="story-text">
                ${{textHTML}}
            </div>

        </section>


        <section class="card">

            <h2>Causal graph</h2>

            <div class="graph-container">

                <div id="graph"></div>

            </div>

            <div class="legend">

                <div>
                    <span class="legend-line cause"></span>
                    CAUSE
                </div>

                <div>
                    <span class="legend-line"></span>
                    PRECONDITION
                </div>

            </div>

        </section>


        <section class="card">

            <h2>Causal relations</h2>

            ${{relationHTML || '<div class="empty">No relations</div>'}}

        </section>

    `;

    drawGraph(story);

}}


// ========================================================
// GRAPH DRAWING
// ========================================================

function drawGraph(story) {{

    const container =
        document.getElementById("graph");

    if (!story.events.length) {{

        container.innerHTML =
            '<div class="empty">No graph events</div>';

        return;

    }}


    const events = story.events;

    const relations = story.relations;


    // ----------------------------------------------------
    // Build graph structure
    // ----------------------------------------------------

    const incoming = {{}};
    const outgoing = {{}};

    events.forEach(e => {{

        incoming[e.id] = [];
        outgoing[e.id] = [];

    }});

    relations.forEach(r => {{

        if (!outgoing[r.source]) {{
            outgoing[r.source] = [];
        }}

        if (!incoming[r.target]) {{
            incoming[r.target] = [];
        }}

        outgoing[r.source].push(r.target);
        incoming[r.target].push(r.source);

    }});


    // ----------------------------------------------------
    // Assign levels
    // ----------------------------------------------------

    const levels = {{}};

    events.forEach(e => {{
        levels[e.id] = 0;
    }});


    // Repeated relaxation
    for (let i = 0; i < events.length; i++) {{

        relations.forEach(r => {{

            levels[r.target] =
                Math.max(
                    levels[r.target] || 0,
                    (levels[r.source] || 0) + 1
                );

        }});

    }}


    // ----------------------------------------------------
    // Group nodes by level
    // ----------------------------------------------------

    const groups = {{}};

    events.forEach(e => {{

        const level =
            levels[e.id] || 0;

        if (!groups[level]) {{
            groups[level] = [];
        }}

        groups[level].push(e);

    }});


    const levelKeys =
        Object.keys(groups)
            .map(Number)
            .sort((a,b) => a-b);


    const nodePositions = {{}};

    const columnWidth = 220;
    const rowHeight = 100;

    const width =
        Math.max(
            900,
            levelKeys.length * columnWidth + 100
        );

    const maxRows =
        Math.max(
            ...levelKeys.map(
                k => groups[k].length
            )
        );

    const height =
        Math.max(
            430,
            maxRows * rowHeight + 100
        );


    // ----------------------------------------------------
    // Position nodes
    // ----------------------------------------------------

    levelKeys.forEach(level => {{

        const nodes = groups[level];

        nodes.forEach((event, i) => {{

            const x =
                80 +
                level * columnWidth;

            const totalHeight =
                nodes.length * rowHeight;

            const startY =
                (height - totalHeight) / 2;

            const y =
                startY +
                i * rowHeight +
                rowHeight / 2;

            nodePositions[event.id] = {{
                x,
                y
            }};

        }});

    }});


    // ----------------------------------------------------
    // SVG
    // ----------------------------------------------------

    let svg = `

        <svg
            width="${{width}}"
            height="${{height}}"
            viewBox="0 0 ${{width}} ${{height}}"
        >

        <defs>

            <marker
                id="arrow"
                markerWidth="10"
                markerHeight="10"
                refX="8"
                refY="3"
                orient="auto"
                markerUnits="strokeWidth"
            >
                <path
                    d="M0,0 L0,6 L9,3 z"
                    fill="#6b7280"
                />
            </marker>

        </defs>

    `;


    // ----------------------------------------------------
    // Edges
    // ----------------------------------------------------

    relations.forEach((rel, index) => {{

        const source =
            nodePositions[rel.source];

        const target =
            nodePositions[rel.target];

        if (!source || !target) {{
            return;
        }}


        const x1 =
            source.x + 75;

        const y1 =
            source.y;

        const x2 =
            target.x - 75;

        const y2 =
            target.y;


        const middleX =
            (x1 + x2) / 2;


        const edgeClass =
            rel.type === "CAUSE"
                ? "edge cause"
                : "edge";


        svg += `

            <path
                class="${{edgeClass}}"
                d="M ${{x1}} ${{y1}}
                   C ${{middleX}} ${{y1}},
                     ${{middleX}} ${{y2}},
                     ${{x2}} ${{y2}}"
                marker-end="url(#arrow)"
            />

            <text
                x="${{middleX}}"
                y="${{(y1 + y2) / 2 - 5}}"
                text-anchor="middle"
                class="edge-label"
            >
                ${{escapeHTML(rel.type)}}
            </text>

        `;

    }});


    // ----------------------------------------------------
    // Nodes
    // ----------------------------------------------------

    events.forEach(event => {{

        const pos =
            nodePositions[event.id];

        const safeTrigger =
            escapeHTML(event.trigger);

        const safeType =
            escapeHTML(event.type);


        svg += `

            <g
                class="node"
                data-event-id="${{event.id}}"
                onclick="selectEvent('${{event.id}}')"
            >

                <rect
                    class="node-box"
                    x="${{pos.x - 75}}"
                    y="${{pos.y - 30}}"
                    width="150"
                    height="60"
                    rx="8"
                />

                <text
                    x="${{pos.x}}"
                    y="${{pos.y - 3}}"
                    text-anchor="middle"
                    class="node-label"
                >
                    ${{safeTrigger}}
                </text>

                <text
                    x="${{pos.x}}"
                    y="${{pos.y + 14}}"
                    text-anchor="middle"
                    class="node-type"
                >
                    ${{safeType}}
                </text>

            </g>

        `;

    }});


    svg += "</svg>";

    container.innerHTML = svg;

}}


// ========================================================
// EVENT SELECTION
// ========================================================

function selectEvent(eventId) {{

    const story =
        stories[currentIndex];

    const event =
        story.events.find(
            e => e.id === eventId
        );

    if (!event) {{
        return;
    }}


    // Remove previous selection

    document
        .querySelectorAll(".node")
        .forEach(node =>
            node.classList.remove("selected")
        );


    // Select current node

    const node =
        document.querySelector(
            `.node[data-event-id="${{eventId}}"]`
        );

    if (node) {{
        node.classList.add("selected");
    }}


    // Highlight sentence

    document
        .querySelectorAll(".sentence")
        .forEach(s =>
            s.classList.remove("highlight")
        );


    const sentence =
        document.getElementById(
            `sentence-${{event.sent_id}}`
        );

    if (sentence) {{

        sentence.classList.add("highlight");

        sentence.scrollIntoView({{
            behavior: "smooth",
            block: "center"
        }});

    }}

}}


// ========================================================
// NAVIGATION
// ========================================================

function previousStory() {{

    if (currentIndex > 0) {{
        renderStory(currentIndex - 1);
    }}

}}


function nextStory() {{

    if (currentIndex < stories.length - 1) {{
        renderStory(currentIndex + 1);
    }}

}}


// ========================================================
// HTML ESCAPING
// ========================================================

function escapeHTML(value) {{

    if (value === null || value === undefined) {{
        return "";
    }}

    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");

}}


// ========================================================
// START
// ========================================================

init();

</script>

</body>

</html>
"""


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MAVEN-ERE CAUSAL GRAPH EXPLORER")
    print("=" * 70)

    print()
    print("Input :", INPUT_FILE)
    print("Output:", OUTPUT_FILE)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print()
    print("Loading dataset...")

    documents = load_dataset(INPUT_FILE)

    print(f"Loaded documents: {len(documents)}")

    print()
    print("Preparing graphs...")

    stories = prepare_stories(documents)

    print(f"Prepared stories: {len(stories)}")

    print()
    print("Generating HTML...")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    html_content = generate_html(stories)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        f.write(html_content)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print()
    print(f"HTML created:")
    print(f"  {OUTPUT_FILE}")

    print()
    print("Open it directly in your browser.")


if __name__ == "__main__":
    main()