#!/usr/bin/env python3

"""
Classify MAVEN-ERE stories into Easy / Medium / Hard.

Input:
    data/mavenere_subset_300_context.json

Output:
    data/mavenere_subset_300_classified.json

Difficulty is based on three dimensions:

1. Story length
       -> number of sentences

2. Causal complexity
       -> number of gold/MAVEN causal relations

3. Causal distance
       -> maximum sentence distance between cause and effect

Each dimension receives:

    0 = low difficulty
    1 = medium difficulty
    2 = high difficulty

Total score:

    0-2 = Easy
    3-4 = Medium
    5-6 = Hard

The thresholds for each dimension are calculated from the
actual 300-story dataset using the 33rd and 66th percentiles.

This avoids choosing arbitrary thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import quantiles
from typing import Any, Dict, List


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "mavenere_subset_300_context.json"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "mavenere_subset_300_classified.json"
)


# ============================================================
# Helpers
# ============================================================

def load_json(path: Path) -> Any:

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def save_json(
    path: Path,
    data: Any,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def get_stories(
    data: Any,
) -> List[Dict[str, Any]]:

    if isinstance(data, list):

        return [
            item
            for item in data
            if isinstance(item, dict)
        ]

    if isinstance(data, dict):

        if isinstance(
            data.get("stories"),
            list,
        ):

            return [
                item
                for item in data["stories"]
                if isinstance(item, dict)
            ]

        if "story_id" in data:
            return [data]

    raise ValueError(
        "Could not find stories in the input JSON."
    )


# ============================================================
# Percentile thresholds
# ============================================================

def calculate_thresholds(
    values: List[float],
) -> tuple[float, float]:

    if len(values) < 3:
        raise ValueError(
            "Need at least 3 stories "
            "to calculate percentile thresholds."
        )

    q1, q2 = quantiles(
        values,
        n=3,
        method="inclusive",
    )

    return q1, q2


# ============================================================
# Dimension scoring
# ============================================================

def score_dimension(
    value: float,
    low_threshold: float,
    high_threshold: float,
) -> int:
    """
    Convert a numerical value into:

        0 = low
        1 = medium
        2 = high
    """

    if value <= low_threshold:
        return 0

    if value <= high_threshold:
        return 1

    return 2


# ============================================================
# Difficulty label
# ============================================================

def difficulty_from_score(
    score: int,
) -> str:

    if score <= 2:
        return "Easy"

    if score <= 4:
        return "Medium"

    return "Hard"


# ============================================================
# Main classification
# ============================================================

def classify_dataset(
    stories: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:

    # --------------------------------------------------------
    # Extract measurements
    # --------------------------------------------------------

    measurements = []

    for story in stories:

        metadata = story.get(
            "metadata",
            {},
        )

        num_sentences = metadata.get(
            "num_sentences"
        )

        num_relations = metadata.get(
            "num_maven_relations"
        )

        max_distance = metadata.get(
            "max_sentence_distance"
        )

        # ----------------------------------------------------
        # Fallbacks
        # ----------------------------------------------------

        if num_sentences is None:

            text = story.get(
                "text",
                "",
            )

            num_sentences = max(
                1,
                len(
                    [
                        x
                        for x in text.split(".")
                        if x.strip()
                    ]
                ),
            )

        if num_relations is None:

            maven_graph = story.get(
                "maven_graph",
                {},
            )

            relations = maven_graph.get(
                "relations",
                [],
            )

            num_relations = len(
                relations
            )

        if max_distance is None:

            maven_graph = story.get(
                "maven_graph",
                {},
            )

            relations = maven_graph.get(
                "relations",
                [],
            )

            distances = [
                r.get(
                    "sentence_distance",
                    0,
                )
                for r in relations
                if isinstance(r, dict)
            ]

            max_distance = (
                max(distances)
                if distances
                else 0
            )

        measurements.append(
            {
                "num_sentences": float(
                    num_sentences
                ),
                "num_relations": float(
                    num_relations
                ),
                "max_distance": float(
                    max_distance
                ),
            }
        )

    # --------------------------------------------------------
    # Calculate dataset-specific thresholds
    # --------------------------------------------------------

    sentence_values = [
        x["num_sentences"]
        for x in measurements
    ]

    relation_values = [
        x["num_relations"]
        for x in measurements
    ]

    distance_values = [
        x["max_distance"]
        for x in measurements
    ]

    sentence_low, sentence_high = (
        calculate_thresholds(
            sentence_values
        )
    )

    relation_low, relation_high = (
        calculate_thresholds(
            relation_values
        )
    )

    distance_low, distance_high = (
        calculate_thresholds(
            distance_values
        )
    )

    # --------------------------------------------------------
    # Classify each story
    # --------------------------------------------------------

    classified = []

    for story, values in zip(
        stories,
        measurements,
    ):

        # -----------------------------------------------
        # Dimension scores
        # -----------------------------------------------

        length_score = score_dimension(
            values["num_sentences"],
            sentence_low,
            sentence_high,
        )

        complexity_score = score_dimension(
            values["num_relations"],
            relation_low,
            relation_high,
        )

        distance_score = score_dimension(
            values["max_distance"],
            distance_low,
            distance_high,
        )

        # -----------------------------------------------
        # Total difficulty
        # -----------------------------------------------

        total_score = (
            length_score
            + complexity_score
            + distance_score
        )

        difficulty = difficulty_from_score(
            total_score
        )

        # -----------------------------------------------
        # Add classification
        # -----------------------------------------------

        story_copy = dict(
            story
        )

        story_copy["difficulty"] = {
            "category": difficulty,

            "score": total_score,

            "max_score": 6,

            "dimensions": {
                "story_length": {
                    "value": int(
                        values["num_sentences"]
                    ),
                    "score": length_score,
                },

                "causal_complexity": {
                    "value": int(
                        values["num_relations"]
                    ),
                    "score": complexity_score,
                },

                "causal_distance": {
                    "value": int(
                        values["max_distance"]
                    ),
                    "score": distance_score,
                },
            },
        }

        classified.append(
            story_copy
        )

    # --------------------------------------------------------
    # Metadata describing the classification
    # --------------------------------------------------------

    classification_metadata = {

        "classification_method": (
            "Dataset-relative difficulty "
            "based on story length, causal "
            "complexity, and maximum causal distance."
        ),

        "dimensions": {
            "story_length": {
                "source": "metadata.num_sentences",
                "low_threshold": sentence_low,
                "high_threshold": sentence_high,
            },

            "causal_complexity": {
                "source": "metadata.num_maven_relations",
                "low_threshold": relation_low,
                "high_threshold": relation_high,
            },

            "causal_distance": {
                "source": "metadata.max_sentence_distance",
                "low_threshold": distance_low,
                "high_threshold": distance_high,
            },
        },

        "dimension_scoring": {
            "0": "Low",
            "1": "Medium",
            "2": "High",
        },

        "difficulty_scoring": {
            "0-2": "Easy",
            "3-4": "Medium",
            "5-6": "Hard",
        },

        "total_stories": len(
            classified
        ),
    }

    return (
        classified,
        classification_metadata,
    )


# ============================================================
# Statistics
# ============================================================

def print_statistics(
    stories: List[Dict[str, Any]],
) -> None:

    counts = {
        "Easy": 0,
        "Medium": 0,
        "Hard": 0,
    }

    print("\n")
    print("=" * 80)
    print("DIFFICULTY DISTRIBUTION")
    print("=" * 80)

    for story in stories:

        difficulty = (
            story
            .get("difficulty", {})
            .get("category")
        )

        if difficulty in counts:
            counts[difficulty] += 1

    total = len(stories)

    for category in (
        "Easy",
        "Medium",
        "Hard",
    ):

        count = counts[category]

        percentage = (
            count / total * 100
            if total
            else 0
        )

        print(
            f"{category:<10} "
            f"{count:>4} stories "
            f"({percentage:>5.1f}%)"
        )

    print("=" * 80)

    # --------------------------------------------------------
    # Examples
    # --------------------------------------------------------

    print("\nExamples:")

    for category in (
        "Easy",
        "Medium",
        "Hard",
    ):

        examples = [
            story
            for story in stories
            if story.get(
                "difficulty",
                {},
            ).get("category")
            == category
        ]

        print(
            f"\n{category}:"
        )

        for story in examples[:3]:

            difficulty = story[
                "difficulty"
            ]

            print(
                f"  {story.get('title', 'Untitled')}"
            )

            print(
                f"    score="
                f"{difficulty['score']}/6, "
                f"sentences="
                f"{difficulty['dimensions']['story_length']['value']}, "
                f"relations="
                f"{difficulty['dimensions']['causal_complexity']['value']}, "
                f"max_distance="
                f"{difficulty['dimensions']['causal_distance']['value']}"
            )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("=" * 80)
    print("MAVEN-ERE STORY DIFFICULTY CLASSIFICATION")
    print("=" * 80)

    print(
        f"\nLoading:\n{INPUT_PATH}"
    )

    data = load_json(
        INPUT_PATH
    )

    stories = get_stories(
        data
    )

    print(
        f"Loaded {len(stories)} stories."
    )

    classified, metadata = (
        classify_dataset(
            stories
        )
    )

    # --------------------------------------------------------
    # Put metadata at the top of output
    # --------------------------------------------------------

    output = {
        "classification_metadata": metadata,
        "stories": classified,
    }

    save_json(
        OUTPUT_PATH,
        output,
    )

    print_statistics(
        classified
    )

    print(
        f"\nSaved classified dataset to:"
        f"\n{OUTPUT_PATH}"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
