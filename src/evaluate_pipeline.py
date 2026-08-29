#!/usr/bin/env python3

"""
Evaluate LLM causal extraction against MAVEN-ERE gold annotations.

Expected gold input format:

[
    {
        "story_id": "...",
        "title": "...",
        "original": {
            "text": "..."
        },
        "gold_relations": [
            {
                "cause": "sold",
                "effect": "lawsuits",
                "relation_type": "PRECONDITION",

                "type": {
                    "source_event_type": "Commerce_sell",
                    "target_event_type": "Legality"
                },

                "context": {
                    "cause": "...",
                    "effect": "..."
                }
            }
        ],
        "revisions": []
    }
]

Expected prediction structure:

        {
            "story_id": "...",
            "models": {
                "qwen3:8b": {
                    "original": {
                        "relations": [...]
                    },
                    "paraphrase": {
                        "relations": [...]
                    }
                }
            }
        }

The script is deliberately flexible and can also work with a simpler
extractions.json format.

Semantic matching
-----------------

A predicted relation matches a gold relation when:

    semantic_similarity(predicted_cause, gold_cause) >= threshold
AND
    semantic_similarity(predicted_effect, gold_effect) >= threshold

Relation type is evaluated separately.

The semantic similarity model defaults to:

    sentence-transformers/all-MiniLM-L6-v2

Install:

    pip install sentence-transformers torch

Usage:

    python evaluate.py \
        data/mavenere_subset_300_enriched.json \
        data/extractions.json

Optional threshold:

    python evaluate.py \
        data/mavenere_subset_300_enriched.json \
        data/extractions.json \
        --threshold 0.70

Output:

    data/evaluation_results.json
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Optional sentence-transformers import
# ---------------------------------------------------------------------------

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_THRESHOLD = 0.70

DEFAULT_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

DEFAULT_OUTPUT = (
    "data/evaluation_results.json"
)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def load_json(path):
    """Load JSON."""

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def save_json(data, path):
    """Save JSON."""

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2
        )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_text(text):
    """
    Normalize text for exact matching.

    Semantic matching is used separately.
    """

    if text is None:
        return ""

    text = str(text).lower().strip()

    # Collapse whitespace.
    text = " ".join(text.split())

    return text


def normalize_relation_type(value):
    """Normalize relation type."""

    if value is None:
        return None

    value = str(value).upper().strip()

    return value


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def precision_recall_f1(tp, fp, fn):
    """Calculate precision, recall and F1."""

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    if precision + recall > 0:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )
    else:
        f1 = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ---------------------------------------------------------------------------
# Semantic model
# ---------------------------------------------------------------------------

class SemanticMatcher:
    """
    Sentence-transformer based semantic similarity.

    Embeddings are cached so that identical event descriptions
    do not need to be encoded repeatedly.
    """

    def __init__(
        self,
        model_name=DEFAULT_MODEL
    ):

        if SentenceTransformer is None:

            raise ImportError(
                "sentence-transformers is not installed.\n"
                "Install it with:\n"
                "pip install sentence-transformers"
            )

        print(
            f"Loading semantic model: {model_name}"
        )

        self.model = SentenceTransformer(
            model_name
        )

        self.cache = {}

    def embedding(self, text):

        text = normalize_text(text)

        if text == "":
            return None

        if text not in self.cache:

            vector = self.model.encode(
                text,
                normalize_embeddings=True
            )

            self.cache[text] = vector

        return self.cache[text]

    def similarity(
        self,
        text_a,
        text_b
    ):

        a = self.embedding(text_a)
        b = self.embedding(text_b)

        if a is None or b is None:
            return 0.0

        return float(
            np.dot(a, b)
        )


# ---------------------------------------------------------------------------
# Gold relation helpers
# ---------------------------------------------------------------------------

def get_gold_relations(story):
    """
    Get gold relations.

    Supports both:

        gold_relations

    and:

        gold_graph.relations
    """

    if "gold_relations" in story:

        return story.get(
            "gold_relations",
            []
        )

    return (
        story
        .get("gold_graph", {})
        .get("relations", [])
    )


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def get_prediction_relations(item, model=None, revision="original"):
    """
    Extract predicted causal relations from a model output item.

    Expected prediction structure:

        {
            "story_id": "...",
            "models": {
                "qwen3:8b": {
                    "original": {
                        "relations": [...]
                    },
                    "paraphrase": {
                        "relations": [...]
                    }
                }
            }
        }

    Args:
        item:
            One story/prediction dictionary.

        model:
            Model name to evaluate, e.g. "qwen3:8b".
            If None, the first available model is used.

        revision:
            Revision to evaluate, e.g. "original", "lexical",
            "syntactic", "restructuring", or "substantial".

    Returns:
        List of predicted relations.
    """

    if not isinstance(item, dict):
        return []


    # ===============================================================
    #  models.<model>.<revision>.relations
    #
    # This is the structure used by your extractions.json
    # ===============================================================

    models = item.get("models")

    if not isinstance(models, dict):
        return []

    # ---------------------------------------------------------------
    # If a specific model was requested
    # ---------------------------------------------------------------

    if model is not None:

        model_output = models.get(model)

        if not isinstance(model_output, dict):
            return []

        revision_output = model_output.get(revision)

        if not isinstance(revision_output, dict):
            return []

        relations = revision_output.get("relations")

        if isinstance(relations, list):
            return relations

        return []

    # ---------------------------------------------------------------
    # No model specified:
    # use the first available model
    # ---------------------------------------------------------------

    for model_name, model_output in models.items():

        if not isinstance(model_output, dict):
            continue

        revision_output = model_output.get(revision)

        if not isinstance(revision_output, dict):
            continue

        relations = revision_output.get("relations")

        if isinstance(relations, list):
            return relations

    return []



def normalize_prediction_relation(relation):
    """
    Convert prediction into:

        {
            "cause": ...,
            "effect": ...,
            "relation_type": ...
        }
    """

    if not isinstance(
        relation,
        dict
    ):
        return None

    cause = (
        relation.get("cause")
        or relation.get("source")
        or relation.get("source_trigger")
    )

    effect = (
        relation.get("effect")
        or relation.get("target")
        or relation.get("target_trigger")
    )

    relation_type = (
        relation.get("relation_type")
        or relation.get("type")
    )

    if isinstance(
        relation_type,
        dict
    ):
        relation_type = (
            relation_type.get("relation_type")
            or relation_type.get("type")
        )

    if not cause or not effect:

        return None

    return {
        "cause": str(cause).strip(),
        "effect": str(effect).strip(),
        "relation_type": normalize_relation_type(
            relation_type
        )
    }


# ---------------------------------------------------------------------------
# Relation matching
# ---------------------------------------------------------------------------

def exact_relation_match(
    predicted,
    gold
):
    """
    Exact normalized cause/effect matching.
    """

    return (
        normalize_text(
            predicted["cause"]
        )
        ==
        normalize_text(
            gold["cause"]
        )
        and

        normalize_text(
            predicted["effect"]
        )
        ==
        normalize_text(
            gold["effect"]
        )
    )

import re


def normalize_text(text):
    """
    Normalize text for lexical matching.
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove punctuation.
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Collapse whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def lexical_match(
    predicted_text,
    gold_context
):
    """
    Check whether the predicted event description
    occurs inside the gold event context.
    """

    predicted = normalize_text(
        predicted_text
    )

    context = normalize_text(
        gold_context
    )

    if not predicted or not context:
        return False

    return predicted in context


def naive_relation_score(
    predicted,
    gold
):
    """
    Match prediction against the cause/effect
    contexts of the gold relation.
    """

    predicted_cause = predicted.get(
        "cause",
        ""
    )

    predicted_effect = predicted.get(
        "effect",
        ""
    )

    gold_context = gold.get(
        "context",
        {}
    )

    if not isinstance(
        gold_context,
        dict
    ):
        return {
            "cause_match": False,
            "effect_match": False,
            "relation_match": False
        }

    gold_cause_context = gold_context.get(
        "cause",
        ""
    )

    gold_effect_context = gold_context.get(
        "effect",
        ""
    )

    cause_match = lexical_match(
        predicted_cause,
        gold_cause_context
    )

    effect_match = lexical_match(
        predicted_effect,
        gold_effect_context
    )

    relation_match = (
        cause_match
        and
        effect_match
    )

    return {
        "cause_match": cause_match,
        "effect_match": effect_match,
        "relation_match": relation_match
    }

def semantic_relation_score(
    predicted,
    gold,
    matcher
):
    """
    Calculate semantic similarity between a predicted relation
    and a gold relation.

    For both cause and effect, compare the predicted text against:

        1. Gold trigger
        2. Gold context

    The maximum similarity is used.

    This prevents the short gold trigger (e.g. "award") from
    unfairly penalizing a longer, semantically correct prediction
    (e.g. "tour won the 'Breakthrough Artist' award").
    """

    # ===============================================================
    # Prediction
    # ===============================================================

    predicted_cause = predicted.get(
        "cause",
        ""
    )

    predicted_effect = predicted.get(
        "effect",
        ""
    )

    # ===============================================================
    # Gold triggers
    # ===============================================================

    gold_cause = gold.get(
        "cause",
        ""
    )

    gold_effect = gold.get(
        "effect",
        ""
    )

    # ===============================================================
    # Gold contexts
    # ===============================================================

    context = gold.get(
        "context",
        {}
    )

    if not isinstance(
        context,
        dict
    ):
        context = {}

    gold_cause_context = context.get(
        "cause",
        ""
    )

    gold_effect_context = context.get(
        "effect",
        ""
    )

    # ===============================================================
    # Cause similarity
    # ===============================================================

    cause_trigger_similarity = matcher.similarity(
        predicted_cause,
        gold_cause
    )

    cause_context_similarity = 0.0

    if gold_cause_context:

        cause_context_similarity = matcher.similarity(
            predicted_cause,
            gold_cause_context
        )

    cause_similarity = max(
        cause_trigger_similarity,
        cause_context_similarity
    )

    # ===============================================================
    # Effect similarity
    # ===============================================================

    effect_trigger_similarity = matcher.similarity(
        predicted_effect,
        gold_effect
    )

    effect_context_similarity = 0.0

    if gold_effect_context:

        effect_context_similarity = matcher.similarity(
            predicted_effect,
            gold_effect_context
        )

    effect_similarity = max(
        effect_trigger_similarity,
        effect_context_similarity
    )

    # ===============================================================
    # Overall relation similarity
    # ===============================================================

    relation_similarity = (
        cause_similarity
        +
        effect_similarity
    ) / 2.0

    return {
        "cause_similarity":
            cause_similarity,

        "effect_similarity":
            effect_similarity,

        "relation_similarity":
            relation_similarity,

        # Keep these for debugging.
        "cause_trigger_similarity":
            cause_trigger_similarity,

        "cause_context_similarity":
            cause_context_similarity,

        "effect_trigger_similarity":
            effect_trigger_similarity,

        "effect_context_similarity":
            effect_context_similarity
    }


# ---------------------------------------------------------------------------
# Greedy semantic matching
# ---------------------------------------------------------------------------

def match_relations(
    predicted_relations,
    gold_relations,
    matcher=None,
    threshold=DEFAULT_THRESHOLD
):
    """
    Match a prediction to a gold relation using the semantic similarity
    of the predicted cause/effect against the gold cause/effect context,
    with a one-to-one greedy assignment.
    """

    candidates = []

    # ===============================================================
    # Find candidate matches
    # ===============================================================

    for pred_idx, predicted in enumerate(
        predicted_relations
    ):

        for gold_idx, gold in enumerate(
            gold_relations
        ):

            scores = semantic_relation_score(
                predicted,
                gold,
                matcher
            )

            if (
                scores["cause_similarity"] >= threshold
                and
                scores["effect_similarity"] >= threshold
            ):

                candidates.append(
                    (
                        pred_idx,
                        gold_idx,
                        scores
                    )
                )

    # ===============================================================
    # Greedy one-to-one matching
    # ===============================================================

    matched_predictions = set()
    matched_gold = set()

    matches = []

    for (
        pred_idx,
        gold_idx,
        scores
    ) in candidates:

        if pred_idx in matched_predictions:
            continue

        if gold_idx in matched_gold:
            continue

        matched_predictions.add(
            pred_idx
        )

        matched_gold.add(
            gold_idx
        )

        matches.append(
            {
                "prediction_index":
                    pred_idx,

                "gold_index":
                    gold_idx,

                "cause_similarity":
                    scores["cause_similarity"],

                "effect_similarity":
                    scores["effect_similarity"],

                "relation_similarity":
                    scores["relation_similarity"]
            }
        )

    # ===============================================================
    # Unmatched predictions
    # ===============================================================

    unmatched_predictions = [
        i
        for i in range(
            len(predicted_relations)
        )
        if i not in matched_predictions
    ]

    # ===============================================================
    # Unmatched gold
    # ===============================================================

    unmatched_gold = [
        i
        for i in range(
            len(gold_relations)
        )
        if i not in matched_gold
    ]

    return (
        matches,
        unmatched_predictions,
        unmatched_gold
    )

    # # ===============================================================
    # # Summary
    # # ===============================================================

    # print()
    # print("=" * 80)
    # print("NAIVE MATCHING SUMMARY")
    # print("=" * 80)

    # print(
    #     f"Candidate matches:      {len(candidates)}"
    # )

    # print(
    #     f"Final matches:          {len(matches)}"
    # )

    # print(
    #     f"Unmatched predictions:  "
    #     f"{len(unmatched_predictions)}"
    # )

    # print(
    #     f"Unmatched gold:         "
    #     f"{len(unmatched_gold)}"
    # )

    # print("=" * 80)

    # return (
    #     matches,
    #     unmatched_predictions,
    #     unmatched_gold
    # )

# ---------------------------------------------------------------------------
# Evaluate one story
# ---------------------------------------------------------------------------

def evaluate_story(
    story_id,
    gold_relations,
    predicted_relations,
    matcher,
    threshold
):
    """
    Evaluate one story.
    """

    # ---------------------------------------------------------------
    # Normalize predictions.
    # ---------------------------------------------------------------

    normalized_predictions = []

    for relation in predicted_relations:

        normalized = (
            normalize_prediction_relation(
                relation
            )
        )

        if normalized is not None:

            normalized_predictions.append(
                normalized
            )

    # ---------------------------------------------------------------
    # Normalize gold relations.
    # ---------------------------------------------------------------

    normalized_gold = []


    for relation in gold_relations:

        if not isinstance(relation, dict):
            continue

        cause = relation.get("cause")
        effect = relation.get("effect")

        if not cause or not effect:
            continue

        normalized_gold.append(relation)

    # ---------------------------------------------------------------
    # Semantic relation matching.
    # ---------------------------------------------------------------

    (
        matches,
        unmatched_predictions,
        unmatched_gold
    ) = match_relations(
        normalized_predictions,
        normalized_gold,
        matcher,
        threshold
    )

    tp = len(matches)

    fp = len(
        unmatched_predictions
    )

    fn = len(
        unmatched_gold
    )

    relation_metrics = precision_recall_f1(
        tp,
        fp,
        fn
    )

    # ---------------------------------------------------------------
    # Cause-level matching.
    #
    # A cause is considered correct when the predicted cause matches
    # a gold cause semantically, independently of the effect.
    # ---------------------------------------------------------------

    cause_candidates = []

    for p_idx, predicted in enumerate(
        normalized_predictions
    ):

        for g_idx, gold in enumerate(
            normalized_gold
        ):

            similarity = matcher.similarity(
                predicted["cause"],
                gold["cause"]
            )

            if similarity >= threshold:

                cause_candidates.append(
                    (
                        similarity,
                        p_idx,
                        g_idx
                    )
                )

    cause_candidates.sort(
        reverse=True
    )

    used_predictions = set()
    used_gold = set()

    cause_tp = 0

    for similarity, p_idx, g_idx in cause_candidates:

        if p_idx in used_predictions:
            continue

        if g_idx in used_gold:
            continue

        used_predictions.add(
            p_idx
        )

        used_gold.add(
            g_idx
        )

        cause_tp += 1

    cause_fp = (
        len(normalized_predictions)
        - cause_tp
    )

    cause_fn = (
        len(normalized_gold)
        - cause_tp
    )

    cause_metrics = precision_recall_f1(
        cause_tp,
        cause_fp,
        cause_fn
    )

    # ---------------------------------------------------------------
    # Effect-level matching.
    # ---------------------------------------------------------------

    effect_candidates = []

    for p_idx, predicted in enumerate(
        normalized_predictions
    ):

        for g_idx, gold in enumerate(
            normalized_gold
        ):

            similarity = matcher.similarity(
                predicted["effect"],
                gold["effect"]
            )

            if similarity >= threshold:

                effect_candidates.append(
                    (
                        similarity,
                        p_idx,
                        g_idx
                    )
                )

    effect_candidates.sort(
        reverse=True
    )

    used_predictions = set()
    used_gold = set()

    effect_tp = 0

    for similarity, p_idx, g_idx in effect_candidates:

        if p_idx in used_predictions:
            continue

        if g_idx in used_gold:
            continue

        used_predictions.add(
            p_idx
        )

        used_gold.add(
            g_idx
        )

        effect_tp += 1

    effect_fp = (
        len(normalized_predictions)
        - effect_tp
    )

    effect_fn = (
        len(normalized_gold)
        - effect_tp
    )

    effect_metrics = precision_recall_f1(
        effect_tp,
        effect_fp,
        effect_fn
    )

    # ---------------------------------------------------------------
    # Relation type accuracy.
    #
    # Only evaluate relation type for relations where cause/effect
    # successfully matched.
    # ---------------------------------------------------------------

    type_correct = 0
    type_total = len(matches)

    type_details = []

    for match in matches:

        predicted = normalized_predictions[
            match["prediction_index"]
        ]

        gold = normalized_gold[
            match["gold_index"]
        ]

        correct = (
            predicted["relation_type"]
            ==
            gold["relation_type"]
        )

        if correct:
            type_correct += 1

        type_details.append(
            {
                "prediction_index":
                    match["prediction_index"],

                "gold_index":
                    match["gold_index"],

                "predicted_type":
                    predicted["relation_type"],

                "gold_type":
                    gold["relation_type"],

                "correct":
                    correct
            }
        )

    type_accuracy = (
        type_correct / type_total
        if type_total > 0
        else 0.0
    )

    # ---------------------------------------------------------------
    # Return story results.
    # ---------------------------------------------------------------

    return {
        "story_id": story_id,

        "counts": {
            "gold_relations":
                len(normalized_gold),

            "predicted_relations":
                len(normalized_predictions),

            "true_positives":
                tp,

            "false_positives":
                fp,

            "false_negatives":
                fn
        },

        "relation": relation_metrics,

        "cause": cause_metrics,

        "effect": effect_metrics,

        "relation_type": {
            "correct":
                type_correct,

            "total":
                type_total,

            "accuracy":
                type_accuracy
        },

        "matches": matches,

        "type_details": type_details,

        "unmatched_predictions": [
            normalized_predictions[i]
            for i in unmatched_predictions
        ],

        "unmatched_gold": [
            normalized_gold[i]
            for i in unmatched_gold
        ]
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate_results(story_results):
    """
    Aggregate micro-level counts over all stories.
    """

    relation_tp = 0
    relation_fp = 0
    relation_fn = 0

    cause_tp = 0
    cause_fp = 0
    cause_fn = 0

    effect_tp = 0
    effect_fp = 0
    effect_fn = 0

    type_correct = 0
    type_total = 0

    total_gold = 0
    total_predictions = 0

    for result in story_results:

        counts = result["counts"]

        relation_tp += counts[
            "true_positives"
        ]

        relation_fp += counts[
            "false_positives"
        ]

        relation_fn += counts[
            "false_negatives"
        ]

        total_gold += counts[
            "gold_relations"
        ]

        total_predictions += counts[
            "predicted_relations"
        ]

        # Cause
        cause = result["cause"]

        # Reconstruct TP/FP/FN from P/R is unsafe,
        # so use the relation counts stored below if available.
        #
        # We calculate them directly from the metrics and counts
        # using the number of predictions/gold.

        cause_f1 = cause["f1"]

        # These values are recalculated below from all story
        # matches using a simpler aggregate method.
        #
        # For robust micro metrics, use relation-level counts.

        effect = result["effect"]

        # Type
        type_correct += result[
            "relation_type"
        ]["correct"]

        type_total += result[
            "relation_type"
        ]["total"]

    relation_metrics = precision_recall_f1(
        relation_tp,
        relation_fp,
        relation_fn
    )

    # -----------------------------------------------------------------------
    # Macro metrics
    # -----------------------------------------------------------------------

    if story_results:

        macro_relation_precision = np.mean(
            [
                r["relation"]["precision"]
                for r in story_results
            ]
        )

        macro_relation_recall = np.mean(
            [
                r["relation"]["recall"]
                for r in story_results
            ]
        )

        macro_relation_f1 = np.mean(
            [
                r["relation"]["f1"]
                for r in story_results
            ]
        )

        macro_cause_precision = np.mean(
            [
                r["cause"]["precision"]
                for r in story_results
            ]
        )

        macro_cause_recall = np.mean(
            [
                r["cause"]["recall"]
                for r in story_results
            ]
        )

        macro_cause_f1 = np.mean(
            [
                r["cause"]["f1"]
                for r in story_results
            ]
        )

        macro_effect_precision = np.mean(
            [
                r["effect"]["precision"]
                for r in story_results
            ]
        )

        macro_effect_recall = np.mean(
            [
                r["effect"]["recall"]
                for r in story_results
            ]
        )

        macro_effect_f1 = np.mean(
            [
                r["effect"]["f1"]
                for r in story_results
            ]
        )

    else:

        macro_relation_precision = 0.0
        macro_relation_recall = 0.0
        macro_relation_f1 = 0.0

        macro_cause_precision = 0.0
        macro_cause_recall = 0.0
        macro_cause_f1 = 0.0

        macro_effect_precision = 0.0
        macro_effect_recall = 0.0
        macro_effect_f1 = 0.0

    # -----------------------------------------------------------------------
    # Return
    # -----------------------------------------------------------------------

    return {

        "micro": {

            "relation": relation_metrics,

            "counts": {
                "gold_relations":
                    total_gold,

                "predicted_relations":
                    total_predictions,

                "true_positives":
                    relation_tp,

                "false_positives":
                    relation_fp,

                "false_negatives":
                    relation_fn
            },

            "relation_type": {
                "correct":
                    type_correct,

                "total":
                    type_total,

                "accuracy": (
                    type_correct / type_total
                    if type_total > 0
                    else 0.0
                )
            }
        },

        "macro": {

            "relation": {
                "precision":
                    float(macro_relation_precision),

                "recall":
                    float(macro_relation_recall),

                "f1":
                    float(macro_relation_f1)
            },

            "cause": {
                "precision":
                    float(macro_cause_precision),

                "recall":
                    float(macro_cause_recall),

                "f1":
                    float(macro_cause_f1)
            },

            "effect": {
                "precision":
                    float(macro_effect_precision),

                "recall":
                    float(macro_effect_recall),

                "f1":
                    float(macro_effect_f1)
            }
        }
    }


# ---------------------------------------------------------------------------
# Find prediction item
# ---------------------------------------------------------------------------

def build_prediction_index(predictions):
    """
    Index predictions by story_id.
    """

    index = {}

    if isinstance(
        predictions,
        list
    ):

        for item in predictions:

            if not isinstance(
                item,
                dict
            ):
                continue

            story_id = item.get(
                "story_id"
            )

            if story_id:
                index[story_id] = item

    elif isinstance(
        predictions,
        dict
    ):

        # ---------------------------------------------------------------
        # Direct story_id -> item mapping
        # ---------------------------------------------------------------

        for key, value in predictions.items():

            if isinstance(
                value,
                dict
            ):

                if value.get(
                    "story_id"
                ):

                    index[
                        value["story_id"]
                    ] = value

                else:

                    index[key] = value

    return index


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    gold_data,
    prediction_data,
    threshold,
    model_name
):
    """
    Evaluate complete dataset.
    """

    matcher = SemanticMatcher(
        model_name
    )

    prediction_index = (
        build_prediction_index(
            prediction_data
        )
    )

    story_results = []

    missing_predictions = []

    # ---------------------------------------------------------------
    # Evaluate each gold story.
    # ---------------------------------------------------------------

    for story in gold_data:

        story_id = story.get(
            "story_id"
        )

        gold_relations = (
            get_gold_relations(
                story
            )
        )

        prediction_item = (
            prediction_index.get(
                story_id
            )
        )

        if prediction_item is None:

            missing_predictions.append(
                story_id
            )

            predicted_relations = []

        else:

            predicted_relations = (
                get_prediction_relations(
                    prediction_item
                )
            )

        result = evaluate_story(
            story_id,
            gold_relations,
            predicted_relations,
            matcher,
            threshold
        )

        story_results.append(
            result
        )

    # ---------------------------------------------------------------
    # Aggregate.
    # ---------------------------------------------------------------

    aggregate = aggregate_results(
        story_results
    )

    return {
        "configuration": {
            "semantic_model":
                model_name,

            "threshold":
                threshold,

            "matching": (
                "cause_similarity >= threshold "
                "AND effect_similarity >= threshold"
            )
        },

        "dataset": {
            "stories":
                len(gold_data),

            "stories_with_predictions":
                len(gold_data)
                - len(missing_predictions),

            "stories_missing_predictions":
                len(missing_predictions),

            "missing_story_ids":
                missing_predictions
        },

        "aggregate": aggregate,

        "stories": story_results
    }


# ---------------------------------------------------------------------------
# Pretty console report
# ---------------------------------------------------------------------------

def print_report(results):

    aggregate = results[
        "aggregate"
    ]

    micro = aggregate[
        "micro"
    ]

    macro = aggregate[
        "macro"
    ]

    relation = micro[
        "relation"
    ]

    cause = macro[
        "cause"
    ]

    effect = macro[
        "effect"
    ]

    print()
    print("=" * 80)
    print("CAUSAL EXTRACTION EVALUATION")
    print("=" * 80)

    print()
    print("DATASET")
    print("-" * 80)

    dataset = results[
        "dataset"
    ]

    print(
        f"Stories:                 "
        f"{dataset['stories']}"
    )

    print(
        f"With predictions:        "
        f"{dataset['stories_with_predictions']}"
    )

    print(
        f"Missing predictions:     "
        f"{dataset['stories_missing_predictions']}"
    )

    print()
    print("MICRO RELATION METRICS")
    print("-" * 80)

    print(
        f"Gold relations:          "
        f"{micro['counts']['gold_relations']}"
    )

    print(
        f"Predicted relations:     "
        f"{micro['counts']['predicted_relations']}"
    )

    print(
        f"True positives:          "
        f"{micro['counts']['true_positives']}"
    )

    print(
        f"False positives:         "
        f"{micro['counts']['false_positives']}"
    )

    print(
        f"False negatives:         "
        f"{micro['counts']['false_negatives']}"
    )

    print(
        f"Precision:               "
        f"{relation['precision']:.4f}"
    )

    print(
        f"Recall:                  "
        f"{relation['recall']:.4f}"
    )

    print(
        f"F1:                      "
        f"{relation['f1']:.4f}"
    )

    print()
    print("MACRO METRICS")
    print("-" * 80)

    print(
        f"Relation Precision:      "
        f"{macro['relation']['precision']:.4f}"
    )

    print(
        f"Relation Recall:         "
        f"{macro['relation']['recall']:.4f}"
    )

    print(
        f"Relation F1:             "
        f"{macro['relation']['f1']:.4f}"
    )

    print(
        f"Cause F1:                "
        f"{cause['f1']:.4f}"
    )

    print(
        f"Effect F1:               "
        f"{effect['f1']:.4f}"
    )

    print()
    print("RELATION TYPE")
    print("-" * 80)

    type_result = micro[
        "relation_type"
    ]

    print(
        f"Correct:                 "
        f"{type_result['correct']}"
    )

    print(
        f"Evaluated:               "
        f"{type_result['total']}"
    )

    print(
        f"Accuracy:                "
        f"{type_result['accuracy']:.4f}"
    )

    print()
    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate LLM causal extraction "
            "against MAVEN-ERE gold relations."
        )
    )

    parser.add_argument(
        "gold",
        help=(
            "Gold evaluation JSON file."
        )
    )

    parser.add_argument(
        "predictions",
        help=(
            "LLM extraction JSON file."
        )
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=(
            "Semantic similarity threshold "
            f"(default: {DEFAULT_THRESHOLD})"
        )
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Sentence-transformers model."
        )
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=(
            "Output evaluation JSON."
        )
    )

    return parser.parse_args()


def main():

    args = parse_args()

    if not 0.0 <= args.threshold <= 1.0:

        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    print(
        f"Gold:        {args.gold}"
    )

    print(
        f"Predictions: {args.predictions}"
    )

    print(
        f"Threshold:   {args.threshold}"
    )

    gold_data = load_json(
        args.gold
    )
    gold_data = gold_data[:10]  # Limit to first 100 stories for testing

    prediction_data = load_json(
        args.predictions
    )

    prediction_data = prediction_data[:10]

    # Make sure gold is a list.
    if isinstance(
        gold_data,
        dict
    ):

        gold_data = [gold_data]



    results = evaluate_dataset(
        gold_data,
        prediction_data,
        args.threshold,
        args.model
    )

    save_json(
        results,
        args.output
    )

    print_report(
        results
    )

    print()
    print(
        f"Detailed results written to: "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
