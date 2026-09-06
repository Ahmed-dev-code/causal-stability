#!/usr/bin/env python3

"""
Web-based human annotation tool for semantic-threshold calibration.

This keeps the same core logic as create_human_match_dataset.py:

    - Loads data/threshold_pool.json
    - Uses the "candidates" array
    - Samples 200 Qwen + 200 Gemma + 200 Llama
    - Uses random seed 42, so the sample is reproducible
    - Resumes from data/human_match_dataset.json
    - Saves after every annotation
    - human_match = 1 for a correct semantic match
    - human_match = 0 for an incorrect semantic match

Run from the project root:

    python src/web_annotation_tool.py

Then open:

    http://127.0.0.1:5000

Requirements:

    pip install flask
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template_string, request


# ============================================================
# Configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "threshold_pool.json"
OUTPUT_PATH = ROOT / "data" / "human_match_dataset.json"

TARGET_PER_MODEL = 200
RANDOM_SEED = 42

MODELS = ["qwen", "gemma", "llama"]

app = Flask(__name__)


# ============================================================
# Data loading
# ============================================================

def load_threshold_pool() -> Dict[str, Any]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing threshold pool:\n{DATA_PATH}"
        )

    with DATA_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(
            "threshold_pool.json must contain a JSON object."
        )

    candidates = data.get("candidates")

    if not isinstance(candidates, list) or not candidates:
        raise ValueError(
            "threshold_pool.json does not contain a valid "
            "'candidates' array."
        )

    return data


def group_by_model(
    candidates: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:

    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        model = str(
            candidate.get("model", "unknown")
        ).lower()

        groups[model].append(candidate)

    return dict(groups)


# ============================================================
# Sampling
# ============================================================

def build_selected_sample(
    candidates: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Same sampling logic as the CLI script:
    200 random candidates per model using seed 42.
    """

    rng = random.Random(RANDOM_SEED)
    groups = group_by_model(candidates)

    for model in MODELS:
        if model not in groups:
            raise ValueError(
                f"Model '{model}' was not found in threshold_pool.json."
            )

        if len(groups[model]) < TARGET_PER_MODEL:
            raise ValueError(
                f"Model '{model}' has only "
                f"{len(groups[model])} candidates; "
                f"{TARGET_PER_MODEL} are required."
            )

    selected: List[Dict[str, Any]] = []

    for model in MODELS:
        selected.extend(
            rng.sample(
                groups[model],
                TARGET_PER_MODEL,
            )
        )

    rng.shuffle(selected)

    return selected


# ============================================================
# Candidate identity
# ============================================================

def candidate_key(
    candidate: Dict[str, Any]
) -> str:

    candidate_id = candidate.get("id")

    if candidate_id is not None:
        return str(candidate_id)

    return "|".join(
        [
            str(candidate.get("model", "")),
            str(candidate.get("story_id", "")),
            str(candidate.get("prediction_index", "")),
            str(candidate.get("gold_index", "")),
        ]
    )


# ============================================================
# Annotation persistence
# ============================================================

def load_existing_annotations() -> List[Dict[str, Any]]:
    if not OUTPUT_PATH.exists():
        return []

    try:
        with OUTPUT_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data

    except json.JSONDecodeError:
        pass

    return []


def save_annotations(
    annotations: List[Dict[str, Any]]
) -> None:

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            annotations,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# Application state
# ============================================================

pool_data = load_threshold_pool()
all_candidates = pool_data["candidates"]

selected_candidates = build_selected_sample(
    all_candidates
)

annotations = load_existing_annotations()

annotated_by_id = {}

for record in annotations:
    if not isinstance(record, dict):
        continue

    candidate = record.get("candidate")

    if isinstance(candidate, dict):
        annotated_by_id[
            candidate_key(candidate)
        ] = record


# ============================================================
# Helper functions
# ============================================================

def annotation_counts() -> Dict[str, int]:

    counts = {
        model: 0
        for model in MODELS
    }

    for record in annotations:
        if not isinstance(record, dict):
            continue

        model = str(
            record.get("model", "")
        ).lower()

        if model in counts:
            counts[model] += 1

    return counts


def make_record(
    candidate: Dict[str, Any],
    human_match: int,
) -> Dict[str, Any]:

    return {
        "candidate_id": candidate.get("id"),
        "model": candidate.get("model"),
        "story_id": candidate.get("story_id"),
        "prediction_index": candidate.get(
            "prediction_index"
        ),
        "gold_index": candidate.get(
            "gold_index"
        ),
        "similarity": None,
        "human_match": human_match,
        "candidate": candidate,
    }


def rebuild_annotation_list(
    candidate: Dict[str, Any],
    human_match: int,
) -> None:
    """
    Update an existing annotation if present; otherwise append it.

    This prevents duplicate records when the user changes an answer.
    """

    key = candidate_key(candidate)

    global annotations, annotated_by_id

    new_record = make_record(
        candidate,
        human_match,
    )

    found = False

    for index, record in enumerate(annotations):
        if not isinstance(record, dict):
            continue

        existing_candidate = record.get("candidate")

        if (
            isinstance(existing_candidate, dict)
            and candidate_key(existing_candidate) == key
        ):
            annotations[index] = new_record
            found = True
            break

    if not found:
        annotations.append(new_record)

    annotated_by_id[key] = new_record

    save_annotations(annotations)


# ============================================================
# API
# ============================================================

@app.get("/api/status")
def api_status():

    counts = annotation_counts()

    annotated_keys = set(
        annotated_by_id.keys()
    )

    remaining = [
        candidate
        for candidate in selected_candidates
        if candidate_key(candidate)
        not in annotated_keys
    ]

    matches = sum(
        record.get("human_match") == 1
        for record in annotations
        if isinstance(record, dict)
    )

    non_matches = sum(
        record.get("human_match") == 0
        for record in annotations
        if isinstance(record, dict)
    )

    return jsonify(
        {
            "total": len(selected_candidates),
            "annotated": len(selected_candidates) - len(remaining),
            "remaining": len(remaining),
            "matches": matches,
            "non_matches": non_matches,
            "by_model": counts,
            "output_path": str(OUTPUT_PATH),
            "complete": len(remaining) == 0,
        }
    )


@app.get("/api/candidate/<int:index>")
def api_candidate(index: int):

    if index < 0 or index >= len(selected_candidates):
        return jsonify(
            {"error": "Invalid candidate index."}
        ), 404

    candidate = selected_candidates[index]
    key = candidate_key(candidate)

    existing = annotated_by_id.get(key)

    return jsonify(
        {
            "index": index,
            "total": len(selected_candidates),
            "candidate": candidate,
            "annotation": (
                existing.get("human_match")
                if existing
                else None
            ),
        }
    )


@app.post("/api/annotate")
def api_annotate():

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return jsonify(
            {"error": "Invalid request."}
        ), 400

    candidate_id = payload.get("candidate_id")
    human_match = payload.get("human_match")

    if human_match not in (0, 1):
        return jsonify(
            {
                "error":
                "human_match must be 0 or 1."
            }
        ), 400

    candidate = None

    for item in selected_candidates:
        if str(item.get("id")) == str(candidate_id):
            candidate = item
            break

    if candidate is None:
        return jsonify(
            {
                "error":
                "Candidate not found in selected sample."
            }
        ), 404

    rebuild_annotation_list(
        candidate,
        int(human_match),
    )

    return jsonify(
        {
            "success": True,
            "candidate_id": candidate_id,
            "human_match": human_match,
            "counts": annotation_counts(),
        }
    )


# ============================================================
# Web interface
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>Human Match Annotation</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: #f4f6f8;
    color: #1f2937;
    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        Roboto,
        Arial,
        sans-serif;
}

header {
    background: #111827;
    color: white;
    padding: 18px 28px;
    position: sticky;
    top: 0;
    z-index: 20;
}

.header-inner {
    max-width: 1400px;
    margin: auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 20px;
}

.title {
    font-size: 21px;
    font-weight: 700;
}

.subtitle {
    color: #cbd5e1;
    font-size: 13px;
    margin-top: 3px;
}

.stats {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

.stat {
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 7px 11px;
    font-size: 13px;
}

main {
    max-width: 1400px;
    margin: 22px auto;
    padding: 0 22px 80px;
}

.progress-container {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 15px 18px;
    margin-bottom: 18px;
}

.progress-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 9px;
    font-size: 14px;
}

.progress-bar {
    height: 9px;
    background: #e5e7eb;
    border-radius: 999px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    width: 0%;
    background: #2563eb;
    transition: width .2s ease;
}

.model-progress {
    display: flex;
    gap: 8px;
    margin-top: 10px;
    flex-wrap: wrap;
}

.model-pill {
    padding: 5px 10px;
    border-radius: 999px;
    background: #f3f4f6;
    font-size: 12px;
}

.layout {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 18px;
}

.card {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 1px 2px rgba(0,0,0,.04);
}

.card-header {
    padding: 13px 17px;
    background: #f8fafc;
    border-bottom: 1px solid #e5e7eb;
    font-weight: 700;
}

.card-body {
    padding: 18px;
}

.meta {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 9px;
    margin-bottom: 16px;
}

.meta-item {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 9px 11px;
}

.meta-label {
    color: #64748b;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .04em;
}

.meta-value {
    margin-top: 3px;
    font-size: 13px;
    font-weight: 600;
    overflow-wrap: anywhere;
}

.section {
    margin-top: 17px;
}

.section-title {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: .05em;
    color: #64748b;
    font-weight: 700;
    margin-bottom: 6px;
}

.text-box {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    background: #fff;
    padding: 12px;
    line-height: 1.55;
    font-size: 14px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}

.trigger {
    background: #f8fafc;
    font-weight: 600;
}

.relation-type {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 6px;
    background: #eef2ff;
    color: #3730a3;
    font-size: 12px;
    font-weight: 700;
}

.annotation-panel {
    margin-top: 18px;
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
}

.question {
    font-size: 17px;
    font-weight: 700;
    margin-bottom: 14px;
}

.buttons {
    display: flex;
    justify-content: center;
    gap: 14px;
}

.answer {
    border: 0;
    border-radius: 10px;
    padding: 14px 45px;
    font-size: 16px;
    font-weight: 700;
    cursor: pointer;
    transition: transform .08s ease, opacity .15s ease;
}

.answer:hover {
    transform: translateY(-1px);
}

.answer:disabled {
    opacity: .5;
    cursor: wait;
}

.yes {
    background: #16a34a;
    color: white;
}

.no {
    background: #dc2626;
    color: white;
}

.keyboard {
    margin-top: 10px;
    color: #64748b;
    font-size: 12px;
}

.navigation {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    margin-top: 15px;
}

.nav-button {
    border: 1px solid #d1d5db;
    background: white;
    border-radius: 8px;
    padding: 9px 15px;
    cursor: pointer;
    font-weight: 600;
}

.nav-button:hover {
    background: #f8fafc;
}

.nav-button:disabled {
    opacity: .4;
    cursor: not-allowed;
}

.status {
    margin-top: 10px;
    min-height: 20px;
    text-align: center;
    font-size: 13px;
    color: #64748b;
}

.completed {
    display: none;
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #065f46;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 18px;
}

@media (max-width: 950px) {

    .layout {
        grid-template-columns: 1fr;
    }

    .header-inner {
        align-items: flex-start;
        flex-direction: column;
    }
}

</style>

</head>

<body>

<header>

<div class="header-inner">

<div>
    <div class="title">Human Match Annotation</div>
    <div class="subtitle">
        Semantic threshold calibration
    </div>
</div>

<div class="stats">
    <div class="stat">
        Qwen: <strong id="qwenCount">0/200</strong>
    </div>

    <div class="stat">
        Gemma: <strong id="gemmaCount">0/200</strong>
    </div>

    <div class="stat">
        Llama: <strong id="llamaCount">0/200</strong>
    </div>
</div>

</div>

</header>


<main>

<div id="completed" class="completed">
    <strong>Annotation complete.</strong>
    All 600 selected candidates have been annotated.
    Your results are saved to
    <code>data/human_match_dataset.json</code>.
</div>


<div class="progress-container">

    <div class="progress-top">

        <strong id="progressText">
            Loading...
        </strong>

        <span id="matchStats">
            Matches: 0 · Non-matches: 0
        </span>

    </div>

    <div class="progress-bar">
        <div id="progressFill" class="progress-fill"></div>
    </div>

    <div class="model-progress">
        <span class="model-pill">
            Qwen: <strong id="qwenProgress">0/200</strong>
        </span>

        <span class="model-pill">
            Gemma: <strong id="gemmaProgress">0/200</strong>
        </span>

        <span class="model-pill">
            Llama: <strong id="llamaProgress">0/200</strong>
        </span>
    </div>

</div>


<div id="annotationArea">

<div class="layout">

    <!-- ================================================= -->
    <!-- Prediction -->
    <!-- ================================================= -->

    <div class="card">

        <div class="card-header">
            Predicted Relation
        </div>

        <div class="card-body">

            <div class="meta">

                <div class="meta-item">
                    <div class="meta-label">Model</div>
                    <div
                        class="meta-value"
                        id="model"
                    ></div>
                </div>

                <div class="meta-item">
                    <div class="meta-label">Story ID</div>
                    <div
                        class="meta-value"
                        id="storyId"
                    ></div>
                </div>

                <div class="meta-item">
                    <div class="meta-label">Prediction Index</div>
                    <div
                        class="meta-value"
                        id="predictionIndex"
                    ></div>
                </div>

                <div class="meta-item">
                    <div class="meta-label">Candidate ID</div>
                    <div
                        class="meta-value"
                        id="candidateId"
                    ></div>
                </div>

            </div>


            <div class="section">

                <div class="section-title">
                    Cause
                </div>

                <div
                    class="text-box"
                    id="predictedCause"
                ></div>

            </div>


            <div class="section">

                <div class="section-title">
                    Effect
                </div>

                <div
                    class="text-box"
                    id="predictedEffect"
                ></div>

            </div>


            <div class="section">

                <div class="section-title">
                    Relation Type
                </div>

                <span
                    class="relation-type"
                    id="predictedType"
                ></span>

            </div>

        </div>

    </div>


    <!-- ================================================= -->
    <!-- Gold -->
    <!-- ================================================= -->

    <div class="card">

        <div class="card-header">
            Gold Relation
        </div>

        <div class="card-body">


            <div class="section">

                <div class="section-title">
                    Cause Trigger
                </div>

                <div
                    class="text-box trigger"
                    id="goldCause"
                ></div>

            </div>


            <div class="section">

                <div class="section-title">
                    Cause Context
                </div>

                <div
                    class="text-box"
                    id="goldCauseContext"
                ></div>

            </div>


            <div class="section">

                <div class="section-title">
                    Effect Trigger
                </div>

                <div
                    class="text-box trigger"
                    id="goldEffect"
                ></div>

            </div>


            <div class="section">

                <div class="section-title">
                    Effect Context
                </div>

                <div
                    class="text-box"
                    id="goldEffectContext"
                ></div>

            </div>


            <div class="section">

                <div class="section-title">
                    Relation Type
                </div>

                <span
                    class="relation-type"
                    id="goldType"
                ></span>

            </div>

        </div>

    </div>

</div>


<div class="annotation-panel">

    <div class="question">
        Does the predicted relation correctly correspond
        to the gold causal relation?
    </div>

    <div class="buttons">

        <button
            class="answer yes"
            id="yesButton"
            onclick="annotate(1)"
        >
            ✓ YES — Match
        </button>

        <button
            class="answer no"
            id="noButton"
            onclick="annotate(0)"
        >
            ✗ NO — Not a Match
        </button>

    </div>

    <div class="keyboard">
        Keyboard: <strong>Y</strong> = Match &nbsp;·&nbsp;
        <strong>N</strong> = Not a match &nbsp;·&nbsp;
        <strong>←</strong>/<strong>→</strong> = Navigate
    </div>

    <div
        class="status"
        id="status"
    ></div>

</div>


<div class="navigation">

    <button
        class="nav-button"
        id="previousButton"
        onclick="previousCandidate()"
    >
        ← Previous
    </button>

    <button
        class="nav-button"
        id="nextButton"
        onclick="nextCandidate()"
    >
        Next →
    </button>

</div>

</div>

</main>


<script>

let currentIndex = 0;
let currentCandidate = null;
let total = 0;
let busy = false;


// ============================================================
// Utility
// ============================================================

function valueOrFallback(value) {

    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {
        return "[not available]";
    }

    return String(value);
}


// ============================================================
// Status
// ============================================================

async function refreshStatus() {

    const response =
        await fetch("/api/status");

    const data =
        await response.json();

    document.getElementById(
        "progressText"
    ).textContent =
        `${data.annotated} / ${data.total} annotated`;

    document.getElementById(
        "matchStats"
    ).textContent =
        `Matches: ${data.matches} · Non-matches: ${data.non_matches}`;

    const percentage =
        data.total
        ? (data.annotated / data.total) * 100
        : 0;

    document.getElementById(
        "progressFill"
    ).style.width =
        `${percentage}%`;

    document.getElementById(
        "qwenCount"
    ).textContent =
        `${data.by_model.qwen}/200`;

    document.getElementById(
        "gemmaCount"
    ).textContent =
        `${data.by_model.gemma}/200`;

    document.getElementById(
        "llamaCount"
    ).textContent =
        `${data.by_model.llama}/200`;

    document.getElementById(
        "qwenProgress"
    ).textContent =
        `${data.by_model.qwen}/200`;

    document.getElementById(
        "gemmaProgress"
    ).textContent =
        `${data.by_model.gemma}/200`;

    document.getElementById(
        "llamaProgress"
    ).textContent =
        `${data.by_model.llama}/200`;

    if (data.complete) {

        document.getElementById(
            "completed"
        ).style.display = "block";

    }

    return data;
}


// ============================================================
// Candidate loading
// ============================================================

async function loadCandidate(index) {

    if (index < 0 || index >= total) {
        return;
    }

    const response =
        await fetch(
            `/api/candidate/${index}`
        );

    const data =
        await response.json();

    currentIndex = data.index;
    currentCandidate = data.candidate;

    renderCandidate(
        data.candidate,
        data.annotation
    );

    updateNavigation();

    updateStatusText(
        data.annotation
    );
}


function renderCandidate(
    candidate,
    annotation
) {

    document.getElementById(
        "model"
    ).textContent =
        valueOrFallback(candidate.model);

    document.getElementById(
        "storyId"
    ).textContent =
        valueOrFallback(candidate.story_id);

    document.getElementById(
        "predictionIndex"
    ).textContent =
        valueOrFallback(candidate.prediction_index);

    document.getElementById(
        "candidateId"
    ).textContent =
        valueOrFallback(candidate.id);

    document.getElementById(
        "predictedCause"
    ).textContent =
        valueOrFallback(
            candidate.predicted_cause
        );

    document.getElementById(
        "predictedEffect"
    ).textContent =
        valueOrFallback(
            candidate.predicted_effect
        );

    document.getElementById(
        "predictedType"
    ).textContent =
        valueOrFallback(
            candidate.predicted_relation_type
        );

    document.getElementById(
        "goldCause"
    ).textContent =
        valueOrFallback(
            candidate.gold_cause
        );

    document.getElementById(
        "goldCauseContext"
    ).textContent =
        valueOrFallback(
            candidate.gold_cause_context
        );

    document.getElementById(
        "goldEffect"
    ).textContent =
        valueOrFallback(
            candidate.gold_effect
        );

    document.getElementById(
        "goldEffectContext"
    ).textContent =
        valueOrFallback(
            candidate.gold_effect_context
        );

    document.getElementById(
        "goldType"
    ).textContent =
        valueOrFallback(
            candidate.gold_relation_type
        );

    const yesButton =
        document.getElementById(
            "yesButton"
        );

    const noButton =
        document.getElementById(
            "noButton"
        );

    yesButton.style.outline =
        annotation === 1
        ? "3px solid #86efac"
        : "none";

    noButton.style.outline =
        annotation === 0
        ? "3px solid #fca5a5"
        : "none";
}


// ============================================================
// Annotation
// ============================================================

async function annotate(
    humanMatch
) {

    if (busy || !currentCandidate) {
        return;
    }

    busy = true;

    document.getElementById(
        "yesButton"
    ).disabled = true;

    document.getElementById(
        "noButton"
    ).disabled = true;

    document.getElementById(
        "status"
    ).textContent =
        "Saving...";

    try {

        const response =
            await fetch(
                "/api/annotate",
                {
                    method: "POST",
                    headers: {
                        "Content-Type":
                            "application/json"
                    },
                    body: JSON.stringify(
                        {
                            candidate_id:
                                currentCandidate.id,
                            human_match:
                                humanMatch
                        }
                    )
                }
            );

        const data =
            await response.json();

        if (!response.ok) {
            throw new Error(
                data.error || "Save failed."
            );
        }

        updateStatusText(
            humanMatch
        );

        await refreshStatus();

        // Automatically move to the next candidate.
        if (currentIndex < total - 1) {

            await loadCandidate(
                currentIndex + 1
            );

        } else {

            await loadCandidate(
                currentIndex
            );
        }

    } catch (error) {

        document.getElementById(
            "status"
        ).textContent =
            `Error: ${error.message}`;

    } finally {

        busy = false;

        document.getElementById(
            "yesButton"
        ).disabled = false;

        document.getElementById(
            "noButton"
        ).disabled = false;
    }
}


// ============================================================
// Navigation
// ============================================================

function updateNavigation() {

    document.getElementById(
        "previousButton"
    ).disabled =
        currentIndex <= 0;

    document.getElementById(
        "nextButton"
    ).disabled =
        currentIndex >= total - 1;
}


async function previousCandidate() {

    if (currentIndex > 0) {
        await loadCandidate(
            currentIndex - 1
        );
    }
}


async function nextCandidate() {

    if (currentIndex < total - 1) {
        await loadCandidate(
            currentIndex + 1
        );
    }
}


// ============================================================
// Status text
// ============================================================

function updateStatusText(
    annotation
) {

    if (annotation === 1) {

        document.getElementById(
            "status"
        ).textContent =
            "Previously labeled: MATCH";

    } else if (annotation === 0) {

        document.getElementById(
            "status"
        ).textContent =
            "Previously labeled: NOT A MATCH";

    } else {

        document.getElementById(
            "status"
        ).textContent =
            "Not yet annotated";

    }
}


// ============================================================
// Keyboard shortcuts
// ============================================================

document.addEventListener(
    "keydown",
    async function(event) {

        // Don't trigger shortcuts while typing.
        if (
            event.target.tagName === "INPUT" ||
            event.target.tagName === "TEXTAREA"
        ) {
            return;
        }

        const key =
            event.key.toLowerCase();

        if (key === "y") {

            await annotate(1);

        } else if (key === "n") {

            await annotate(0);

        } else if (event.key === "ArrowLeft") {

            await previousCandidate();

        } else if (event.key === "ArrowRight") {

            await nextCandidate();

        }
    }
);


// ============================================================
// Initialization
// ============================================================

async function initialize() {

    try {

        const status =
            await refreshStatus();

        total = status.total;

        /*
         * Start at the first unannotated candidate.
         * This makes resuming convenient.
         */

        let firstUnannotated = null;

        for (
            let i = 0;
            i < total;
            i++
        ) {

            const response =
                await fetch(
                    `/api/candidate/${i}`
                );

            const data =
                await response.json();

            if (data.annotation === null) {

                firstUnannotated = i;
                break;

            }
        }

        if (firstUnannotated !== null) {

            await loadCandidate(
                firstUnannotated
            );

        } else {

            await loadCandidate(
                total - 1
            );

        }

    } catch (error) {

        document.getElementById(
            "status"
        ).textContent =
            `Error: ${error.message}`;

    }
}


initialize();

</script>

</body>

</html>
"""


# ============================================================
# Routes
# ============================================================

@app.get("/")
def index():
    return render_template_string(HTML)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("WEB HUMAN MATCH ANNOTATION TOOL")
    print("=" * 70)

    print(
        f"Threshold pool: {DATA_PATH}"
    )

    print(
        f"Output:          {OUTPUT_PATH}"
    )

    print(
        f"Sample:          {TARGET_PER_MODEL} per model "
        f"({TARGET_PER_MODEL * len(MODELS)} total)"
    )

    print(
        "Models:          Qwen, Gemma, Llama"
    )

    print(
        "\nOpen in your browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "\nPress CTRL+C to stop the server."
    )

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
