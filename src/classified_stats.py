#!/usr/bin/env python3

"""
Calculate evaluation statistics by story difficulty.

Inputs:
    data/mavenere_subset_300_classified.json
    data/matches.json

Output:
    data/classified_stats.csv

The script:
    1. Loads the classified MAVEN-ERE dataset.
    2. Gets Easy / Medium / Hard for each story.
    3. Loads matches.json.
    4. Joins the two files using story_id.
    5. Aggregates TP, FP, FN and match statistics.
    6. Calculates Precision, Recall and F1 per category.
    7. Saves everything to CSV.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

CLASSIFIED_PATH = (
    ROOT / "data" / "mavenere_subset_300_classified.json"
)

MATCHES_PATH = (
    ROOT / "data" / "reorder" / "qwen_matches.json"
)

OUTPUT_PATH = (
    ROOT / "data" / "reorder" / "qwen_classified_stats.csv"
)


# ============================================================
# LOAD JSON
# ============================================================

def load_json(path: Path) -> Any:

    if not path.exists():
        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ============================================================
# LOAD CLASSIFICATIONS
# ============================================================

def load_classifications(
    path: Path,
) -> Dict[str, str]:

    data = load_json(path)

    # --------------------------------------------------------
    # Your classification script is expected to produce:
    #
    # {
    #     ...
    #     "stories": [
    #         {
    #             "story_id": "...",
    #             ...
    #             "difficulty": {
    #                 ...
    #                 "category": "Easy"
    #             }
    #         }
    #     ]
    # }
    # --------------------------------------------------------

    if isinstance(data, dict):

        stories = data.get(
            "stories",
            []
        )

    elif isinstance(data, list):

        stories = data

    else:

        raise ValueError(
            "Unexpected format in classified dataset."
        )

    classifications = {}

    for story in stories:

        if not isinstance(story, dict):
            continue

        story_id = story.get(
            "story_id"
        )

        if story_id is None:
            continue

        # ----------------------------------------------------
        # Expected location:
        #
        # story["difficulty"]["category"]
        # ----------------------------------------------------

        difficulty = story.get(
            "difficulty",
            {}
        )

        category = None

        if isinstance(
            difficulty,
            dict,
        ):

            category = difficulty.get(
                "category"
            )

        # ----------------------------------------------------
        # Fallback in case the previous script stored
        # the category directly.
        # ----------------------------------------------------

        if category is None:

            category = story.get(
                "category"
            )

        if category not in {
            "Easy",
            "Medium",
            "Hard",
        }:

            print(
                f"WARNING: No valid category for "
                f"story {story_id}: {category}"
            )

            continue

        classifications[
            str(story_id)
        ] = category

    return classifications


# ============================================================
# LOAD MATCHES
# ============================================================

def load_match_records(
    path: Path,
) -> List[Dict[str, Any]]:

    data = load_json(path)

    # --------------------------------------------------------
    # Your matches.json structure is:
    #
    # {
    #     "stories": [
    #         {...},
    #         {...}
    #     ]
    # }
    # --------------------------------------------------------

    if not isinstance(data, dict):

        raise ValueError(
            "matches.json should contain a JSON object."
        )

    stories = data.get(
        "stories"
    )

    if not isinstance(
        stories,
        list,
    ):

        raise ValueError(
            "Could not find 'stories' array "
            "in matches.json."
        )

    records = []

    for story in stories:

        if not isinstance(
            story,
            dict,
        ):
            continue

        if story.get(
            "story_id"
        ) is None:

            continue

        records.append(
            story
        )

    if not records:

        raise ValueError(
            "No story-level match records "
            "were found in matches.json."
        )

    return records


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    tp: int,
    fp: int,
    fn: int,
) -> tuple[float, float, float]:

    if tp + fp > 0:

        precision = (
            tp / (tp + fp)
        )

    else:

        precision = 0.0

    if tp + fn > 0:

        recall = (
            tp / (tp + fn)
        )

    else:

        recall = 0.0

    if precision + recall > 0:

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    return (
        precision,
        recall,
        f1,
    )


# ============================================================
# AGGREGATE
# ============================================================

def aggregate_results(
    records: List[Dict[str, Any]],
    classifications: Dict[str, str],
) -> Dict[str, Dict[str, int]]:

    categories = {

        "Easy": defaultdict(int),

        "Medium": defaultdict(int),

        "Hard": defaultdict(int),

        "Unknown": defaultdict(int),
    }

    for record in records:

        story_id = str(
            record["story_id"]
        )

        category = classifications.get(
            story_id,
            "Unknown",
        )

        stats = categories[
            category
        ]

        # ----------------------------------------------------
        # Number of stories
        # ----------------------------------------------------

        stats[
            "stories"
        ] += 1

        # ----------------------------------------------------
        # Core evaluation metrics
        # ----------------------------------------------------

        stats[
            "tp"
        ] += int(
            record.get(
                "tp",
                0,
            ) or 0
        )

        stats[
            "fp"
        ] += int(
            record.get(
                "fp",
                0,
            ) or 0
        )

        stats[
            "fn"
        ] += int(
            record.get(
                "fn",
                0,
            ) or 0
        )

        # ----------------------------------------------------
        # Matching information
        # ----------------------------------------------------

        stats[
            "lexical_matches"
        ] += int(
            record.get(
                "lexical_matches",
                0,
            ) or 0
        )

        stats[
            "semantic_fallback_matches"
        ] += int(
            record.get(
                "semantic_fallback_matches",
                0,
            ) or 0
        )

        stats[
            "no_matches"
        ] += int(
            record.get(
                "no_matches",
                0,
            ) or 0
        )

        # ----------------------------------------------------
        # Gold / prediction relation counts
        # ----------------------------------------------------

        stats[
            "gold_relations"
        ] += int(
            record.get(
                "gold_relations",
                0,
            ) or 0
        )

        stats[
            "predicted_relations"
        ] += int(
            record.get(
                "predicted_relations",
                0,
            ) or 0
        )

    return categories


# ============================================================
# CREATE CSV ROWS
# ============================================================

def create_rows(
    totals: Dict[str, Dict[str, int]],
) -> List[Dict[str, Any]]:

    rows = []

    # --------------------------------------------------------
    # Easy / Medium / Hard
    # --------------------------------------------------------

    for category in (
        "Easy",
        "Medium",
        "Hard",
    ):

        stats = totals[
            category
        ]

        tp = stats[
            "tp"
        ]

        fp = stats[
            "fp"
        ]

        fn = stats[
            "fn"
        ]

        precision, recall, f1 = (
            calculate_metrics(
                tp,
                fp,
                fn,
            )
        )

        rows.append(
            {
                "category": category,

                "stories": stats[
                    "stories"
                ],

                "gold_relations": stats[
                    "gold_relations"
                ],

                "predicted_relations": stats[
                    "predicted_relations"
                ],

                "tp": tp,

                "fp": fp,

                "fn": fn,

                "precision": precision,

                "recall": recall,

                "f1": f1,

                "lexical_matches": stats[
                    "lexical_matches"
                ],

                "semantic_fallback_matches": stats[
                    "semantic_fallback_matches"
                ],

                "no_matches": stats[
                    "no_matches"
                ],
            }
        )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    overall = defaultdict(int)

    for category in (
        "Easy",
        "Medium",
        "Hard",
    ):

        for key, value in totals[
            category
        ].items():

            overall[
                key
            ] += value

    tp = overall[
        "tp"
    ]

    fp = overall[
        "fp"
    ]

    fn = overall[
        "fn"
    ]

    precision, recall, f1 = (
        calculate_metrics(
            tp,
            fp,
            fn,
        )
    )

    rows.append(
        {
            "category": "Overall",

            "stories": overall[
                "stories"
            ],

            "gold_relations": overall[
                "gold_relations"
            ],

            "predicted_relations": overall[
                "predicted_relations"
            ],

            "tp": tp,

            "fp": fp,

            "fn": fn,

            "precision": precision,

            "recall": recall,

            "f1": f1,

            "lexical_matches": overall[
                "lexical_matches"
            ],

            "semantic_fallback_matches": overall[
                "semantic_fallback_matches"
            ],

            "no_matches": overall[
                "no_matches"
            ],
        }
    )

    return rows


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    path: Path,
    rows: List[Dict[str, Any]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "category",
        "stories",
        "gold_relations",
        "predicted_relations",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "lexical_matches",
        "semantic_fallback_matches",
        "no_matches",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                row
            )


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    rows: List[Dict[str, Any]],
) -> None:

    print()
    print("=" * 110)
    print("EVALUATION BY STORY DIFFICULTY")
    print("=" * 110)

    print(
        f"{'Category':<10}"
        f"{'Stories':>8}"
        f"{'Gold':>8}"
        f"{'Pred':>8}"
        f"{'TP':>8}"
        f"{'FP':>8}"
        f"{'FN':>8}"
        f"{'Prec.':>10}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
    )

    print("-" * 110)

    for row in rows:

        print(
            f"{row['category']:<10}"
            f"{row['stories']:>8}"
            f"{row['gold_relations']:>8}"
            f"{row['predicted_relations']:>8}"
            f"{row['tp']:>8}"
            f"{row['fp']:>8}"
            f"{row['fn']:>8}"
            f"{row['precision']:>10.4f}"
            f"{row['recall']:>10.4f}"
            f"{row['f1']:>10.4f}"
        )

    print("=" * 110)

    print(
        f"\nDetailed CSV:"
        f"\n{OUTPUT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 80)
    print("MAVEN-ERE CLASSIFIED EVALUATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Load classification
    # --------------------------------------------------------

    print(
        "\nLoading classified dataset..."
    )

    classifications = (
        load_classifications(
            CLASSIFIED_PATH
        )
    )

    print(
        f"Classified stories: "
        f"{len(classifications)}"
    )

    # --------------------------------------------------------
    # Show category distribution
    # --------------------------------------------------------

    category_counts = defaultdict(int)

    for category in classifications.values():

        category_counts[
            category
        ] += 1

    print(
        "\nClassification distribution:"
    )

    for category in (
        "Easy",
        "Medium",
        "Hard",
    ):

        print(
            f"  {category:<8}: "
            f"{category_counts[category]}"
        )

    # --------------------------------------------------------
    # Load matches
    # --------------------------------------------------------

    print(
        "\nLoading matches.json..."
    )

    records = load_match_records(
        MATCHES_PATH
    )

    print(
        f"Match records: "
        f"{len(records)}"
    )

    # --------------------------------------------------------
    # Check coverage
    # --------------------------------------------------------

    match_story_ids = {
        str(record["story_id"])
        for record in records
    }

    classified_story_ids = set(
        classifications.keys()
    )

    missing_classification = (
        match_story_ids
        - classified_story_ids
    )

    missing_matches = (
        classified_story_ids
        - match_story_ids
    )

    if missing_classification:

        print(
            f"\nWARNING: "
            f"{len(missing_classification)} "
            "stories have no classification."
        )

    if missing_matches:

        print(
            f"\nWARNING: "
            f"{len(missing_matches)} "
            "classified stories have no match result."
        )

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    totals = aggregate_results(
        records,
        classifications,
    )

    # --------------------------------------------------------
    # Create CSV rows
    # --------------------------------------------------------

    rows = create_rows(
        totals
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_csv(
        OUTPUT_PATH,
        rows,
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_results(
        rows
    )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()
