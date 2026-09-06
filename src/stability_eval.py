#!/usr/bin/env python3
"""
Stability evaluation for causal-relation extraction under text revisions.

Reuses the trigger-based matching logic from `causal_relation_matcher.py`
unchanged, and runs it four times per story/model — once against the
original text's predictions and once against each revision's predictions
(paraphrase / reorder / context_distance) — all scored against the SAME
gold relation list. It then compares the four outcomes to quantify how
*stable* each model's extractions are under perturbation.

Input layout (as clarified):
    - 3 "original extraction" files, one per model. Each story looks like:
        {
          "story_id": "...",
          "original": {"text": "...", "gold_graph": {"relations": [...]}},
          "revisions": [],
          "models": {"<model_name>": {"original": {"relations": [...]}}}
        }
    - 3 "revisions" files, one per model. Each story looks like:
        {
          "story_id": "...",
          "original": {"text": "...", "gold_graph": {"relations": [...]}},  # gold here has "context"
          "revisions": [{"revision_id": "..._paraphrase", "type": "paraphrase", "text": "..."}, ...],
          "models": {
            "<model_name>": {
              "<story_id>_paraphrase": {"relations": [...]},
              "<story_id>_reorder": {"relations": [...]},
              "<story_id>_context_distance": {"relations": [...]}
            }
          }
        }

The model name for each file is inferred from the "models" dict itself
(the first non-empty key found), NOT from the filename, so files can be
passed in any order.

Usage: //powershell
python src/stability_eval.py `
    --originals data/gemma_extractions.json data/llama_extractions.json data/qwen_extractions.json `
    --revisions data/gemma_extractions_revisions.json data/llama_extractions_revisions.json data/qwen_extractions_revisions.json `
    --threshold 0.5 `
    --out stability_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Dict, List, Optional

from causal_relation_matcher_rev import (
    DefaultSemanticMatcher,
    EmbeddingSemanticMatcher,
    SemanticMatcher,
    get_gold_relations,
    get_prediction_relations,
    match_relations,
)

REVISION_TYPES = ["paraphrase", "reorder", "context_distance"]
CONDITIONS = ["original"] + REVISION_TYPES


def get_gold_relations_from_story(story: Optional[dict]) -> List[dict]:
    """
    Gold relations live under story["original"]["gold_graph"]["relations"]
    in these files. Falls back to causal_relation_matcher's own
    get_gold_relations() (which checks top-level "gold_graph" /
    "gold_relations") for compatibility with other layouts.
    """
    if not isinstance(story, dict):
        return []
    original = story.get("original")
    if isinstance(original, dict):
        gold_graph = original.get("gold_graph")
        if isinstance(gold_graph, dict):
            relations = gold_graph.get("relations")
            if isinstance(relations, list):
                return relations
    return get_gold_relations(story)


# --------------------------------------------------------------------------
# Loading / indexing
# --------------------------------------------------------------------------

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def infer_model_name(stories: List[dict], source: str) -> str:
    """Find the (single) model key used inside a file's "models" dicts."""
    names = set()
    for story in stories:
        if not isinstance(story, dict):
            continue
        models = story.get("models")
        if isinstance(models, dict):
            names.update(models.keys())
    if not names:
        raise ValueError(f"No 'models' key found in any story in {source}")
    if len(names) > 1:
        print(
            f"[warning] {source} contains multiple model keys {sorted(names)}; "
            f"expected exactly one per file. Using all of them separately.",
            file=sys.stderr,
        )
    return names


def index_by_story_id(stories: List[dict]) -> Dict[str, dict]:
    return {s["story_id"]: s for s in stories if isinstance(s, dict) and s.get("story_id")}


def load_model_files(paths: List[str], label: str) -> Dict[str, Dict[str, dict]]:
    """
    Returns {model_name: {story_id: story_dict}} across all given files,
    inferring the model name from each file's own "models" dict.
    """
    by_model: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for path in paths:
        stories = load_json(path)
        model_names = infer_model_name(stories, path)
        story_index = index_by_story_id(stories)
        for model_name in model_names:
            for story_id, story in story_index.items():
                by_model[model_name][story_id] = story
        print(f"[info] {label} file {path}: model(s)={sorted(model_names)}, stories={len(story_index)}", file=sys.stderr)
    return by_model


def collect_all_texts(originals_by_model, revisions_by_model) -> List[str]:
    """Gather every string that might get embedded, for one batched warm_cache()."""
    texts = []

    def add_relations(relations):
        for r in relations or []:
            texts.append(r.get("cause", ""))
            texts.append(r.get("effect", ""))
            ctx = r.get("context", {}) or {}
            texts.append(ctx.get("cause", ""))
            texts.append(ctx.get("effect", ""))

    for model, stories in {**originals_by_model, **revisions_by_model}.items():
        for story in stories.values():
            add_relations(get_gold_relations_from_story(story))
            models_dict = story.get("models", {}) or {}
            model_output = models_dict.get(model, {}) or {}
            for revision_output in model_output.values():
                if isinstance(revision_output, dict):
                    add_relations(revision_output.get("relations"))

    return [t for t in texts if t]


# --------------------------------------------------------------------------
# Per-condition evaluation
# --------------------------------------------------------------------------

def evaluate_condition(predictions: List[dict], gold_relations: List[dict],
                        semantic_matcher: Optional[SemanticMatcher], threshold: float) -> dict:
    matches, unmatched_pred, unmatched_gold = match_relations(
        predictions, gold_relations, semantic_matcher, threshold, verbose=False,
    )
    tp, fp, fn = len(matches), len(unmatched_pred), len(unmatched_gold)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    matched_gold_methods = {m["gold_index"]: m["match_method"] for m in matches}

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "n_predictions": len(predictions),
        "n_gold": len(gold_relations),
        "matched_gold_indices": set(matched_gold_methods.keys()),
        "matched_gold_methods": matched_gold_methods,  # gold_index -> "lexical"/"semantic"
    }


def evaluate_story_model(story_id: str, gold_relations: List[dict],
                          orig_story: Optional[dict], rev_story: Optional[dict],
                          model: str, semantic_matcher: Optional[SemanticMatcher],
                          threshold: float) -> dict:
    per_condition = {}

    if orig_story is not None:
        preds = get_prediction_relations(orig_story, model=model, revision="original")
        per_condition["original"] = evaluate_condition(preds, gold_relations, semantic_matcher, threshold)

    if rev_story is not None:
        for rev_type in REVISION_TYPES:
            preds = get_prediction_relations(rev_story, model=model, revision=rev_type)
            per_condition[rev_type] = evaluate_condition(preds, gold_relations, semantic_matcher, threshold)

    return {
        "story_id": story_id,
        "n_gold": len(gold_relations),
        "conditions": per_condition,
    }


# --------------------------------------------------------------------------
# Stability metrics (built from the per-condition evaluations above)
# --------------------------------------------------------------------------

def compute_stability_for_story_model(story_result: dict) -> dict:
    conditions = story_result["conditions"]
    present_conditions = [c for c in CONDITIONS if c in conditions]
    n_gold = story_result["n_gold"]

    # --- 1. Metric drift: revision - original, per revision type ---
    drift = {}
    if "original" in conditions:
        orig = conditions["original"]
        for rev_type in REVISION_TYPES:
            if rev_type in conditions:
                rev = conditions[rev_type]
                drift[rev_type] = {
                    "delta_precision": rev["precision"] - orig["precision"],
                    "delta_recall": rev["recall"] - orig["recall"],
                    "delta_f1": rev["f1"] - orig["f1"],
                }

    # --- 2. Per-gold-relation recovery vector across present conditions ---
    recovery = {}          # gold_index -> {condition: bool}
    method_by_cond = {}    # gold_index -> {condition: "lexical"/"semantic"/None}
    for gi in range(n_gold):
        recovery[gi] = {}
        method_by_cond[gi] = {}
        for cond in present_conditions:
            matched = gi in conditions[cond]["matched_gold_indices"]
            recovery[gi][cond] = matched
            method_by_cond[gi][cond] = conditions[cond]["matched_gold_methods"].get(gi) if matched else None

    n_cond = len(present_conditions)
    per_relation_stability = {}
    for gi, cond_map in recovery.items():
        recovered_count = sum(1 for v in cond_map.values() if v)
        score = recovered_count / n_cond if n_cond else 0.0
        if n_cond and recovered_count == n_cond:
            bucket = "fully_stable"
        elif recovered_count == 0:
            bucket = "fully_unstable"
        else:
            bucket = "partially_stable"
        per_relation_stability[gi] = {"score": score, "bucket": bucket, "recovery": cond_map}

    # --- 3. Original-vs-revision agreement (Jaccard + flip rate) ---
    agreement = {}
    if "original" in conditions:
        orig_matched = conditions["original"]["matched_gold_indices"]
        for rev_type in REVISION_TYPES:
            if rev_type not in conditions:
                continue
            rev_matched = conditions[rev_type]["matched_gold_indices"]
            union = orig_matched | rev_matched
            jaccard = len(orig_matched & rev_matched) / len(union) if union else 1.0
            flips = sum(
                1 for gi in range(n_gold)
                if (gi in orig_matched) != (gi in rev_matched)
            )
            flip_rate = flips / n_gold if n_gold else 0.0
            agreement[rev_type] = {"jaccard": jaccard, "flip_rate": flip_rate}

    # --- 4. Lexical-match degradation under perturbation ---
    lexical_degradation = {}
    if "original" in conditions:
        lexical_in_orig = [
            gi for gi, m in conditions["original"]["matched_gold_methods"].items()
            if m == "lexical"
        ]
        for rev_type in REVISION_TYPES:
            if rev_type not in conditions or not lexical_in_orig:
                continue
            degraded_to_semantic = 0
            lost_entirely = 0
            still_lexical = 0
            for gi in lexical_in_orig:
                method = conditions[rev_type]["matched_gold_methods"].get(gi)
                if method == "lexical":
                    still_lexical += 1
                elif method == "semantic":
                    degraded_to_semantic += 1
                else:
                    lost_entirely += 1
            n = len(lexical_in_orig)
            lexical_degradation[rev_type] = {
                "n_lexical_in_original": n,
                "still_lexical": still_lexical,
                "degraded_to_semantic": degraded_to_semantic,
                "lost_entirely": lost_entirely,
                "degraded_to_semantic_rate": degraded_to_semantic / n,
                "lost_entirely_rate": lost_entirely / n,
            }

    overall_relation_stability = (
        sum(v["score"] for v in per_relation_stability.values()) / len(per_relation_stability)
        if per_relation_stability else None
    )

    return {
        "metric_drift": drift,
        "per_relation_stability": per_relation_stability,
        "overall_relation_stability": overall_relation_stability,
        "agreement_vs_original": agreement,
        "lexical_degradation_vs_original": lexical_degradation,
    }


# --------------------------------------------------------------------------
# Aggregation across stories, per model and overall
# --------------------------------------------------------------------------

def _avg(values: List[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def aggregate_model_results(story_results: List[dict], stability_results: List[dict]) -> dict:
    # Aggregate raw P/R/F1 per condition (micro over TP/FP/FN, like the original script)
    agg_conditions = {}
    for cond in CONDITIONS:
        tp = sum(sr["conditions"][cond]["tp"] for sr in story_results if cond in sr["conditions"])
        fp = sum(sr["conditions"][cond]["fp"] for sr in story_results if cond in sr["conditions"])
        fn = sum(sr["conditions"][cond]["fn"] for sr in story_results if cond in sr["conditions"])
        if tp or fp or fn:
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            agg_conditions[cond] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    agg_drift = {}
    for rev_type in REVISION_TYPES:
        deltas = [sr["metric_drift"][rev_type] for sr in stability_results if rev_type in sr["metric_drift"]]
        if deltas:
            agg_drift[rev_type] = {
                "avg_delta_precision": _avg([d["delta_precision"] for d in deltas]),
                "avg_delta_recall": _avg([d["delta_recall"] for d in deltas]),
                "avg_delta_f1": _avg([d["delta_f1"] for d in deltas]),
            }

    agg_agreement = {}
    for rev_type in REVISION_TYPES:
        pairs = [sr["agreement_vs_original"][rev_type] for sr in stability_results if rev_type in sr["agreement_vs_original"]]
        if pairs:
            agg_agreement[rev_type] = {
                "avg_jaccard": _avg([p["jaccard"] for p in pairs]),
                "avg_flip_rate": _avg([p["flip_rate"] for p in pairs]),
            }

    agg_lexical_degradation = {}
    for rev_type in REVISION_TYPES:
        entries = [
            sr["lexical_degradation_vs_original"][rev_type]
            for sr in stability_results if rev_type in sr["lexical_degradation_vs_original"]
        ]
        if entries:
            total_lexical = sum(e["n_lexical_in_original"] for e in entries)
            total_degraded = sum(e["degraded_to_semantic"] for e in entries)
            total_lost = sum(e["lost_entirely"] for e in entries)
            agg_lexical_degradation[rev_type] = {
                "n_lexical_in_original": total_lexical,
                "degraded_to_semantic_rate": total_degraded / total_lexical if total_lexical else None,
                "lost_entirely_rate": total_lost / total_lexical if total_lexical else None,
            }

    bucket_counts = defaultdict(int)
    for sr in stability_results:
        for rel in sr["per_relation_stability"].values():
            bucket_counts[rel["bucket"]] += 1

    return {
        "n_stories": len(story_results),
        "conditions": agg_conditions,
        "metric_drift": agg_drift,
        "agreement_vs_original": agg_agreement,
        "lexical_degradation_vs_original": agg_lexical_degradation,
        "relation_stability_buckets": dict(bucket_counts),
        "overall_relation_stability": _avg([sr["overall_relation_stability"] for sr in stability_results]),
    }


# --------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------

def print_model_summary(model: str, agg: dict) -> None:
    print()
    print("=" * 80)
    print(f"MODEL: {model}   ({agg['n_stories']} stories)")
    print("=" * 80)
    print(f"{'Condition':<18}{'TP':>6}{'FP':>6}{'FN':>6}{'Precision':>12}{'Recall':>10}{'F1':>10}")
    for cond in CONDITIONS:
        c = agg["conditions"].get(cond)
        if c:
            print(f"{cond:<18}{c['tp']:>6}{c['fp']:>6}{c['fn']:>6}{c['precision']:>12.4f}{c['recall']:>10.4f}{c['f1']:>10.4f}")

    print()
    print(f"{'Revision':<18}{'ΔPrecision':>12}{'ΔRecall':>10}{'ΔF1':>10}{'Jaccard':>10}{'FlipRate':>10}{'Sem.degrade%':>14}{'Lost%':>8}")
    for rev_type in REVISION_TYPES:
        d = agg["metric_drift"].get(rev_type, {})
        a = agg["agreement_vs_original"].get(rev_type, {})
        l = agg["lexical_degradation_vs_original"].get(rev_type, {})
        print(
            f"{rev_type:<18}"
            f"{d.get('avg_delta_precision', float('nan')):>12.4f}"
            f"{d.get('avg_delta_recall', float('nan')):>10.4f}"
            f"{d.get('avg_delta_f1', float('nan')):>10.4f}"
            f"{a.get('avg_jaccard', float('nan')):>10.4f}"
            f"{a.get('avg_flip_rate', float('nan')):>10.4f}"
            f"{(l.get('degraded_to_semantic_rate') or float('nan')) * 100:>13.1f}%"
            f"{(l.get('lost_entirely_rate') or float('nan')) * 100:>7.1f}%"
        )

    print()
    buckets = agg["relation_stability_buckets"]
    total = sum(buckets.values()) or 1
    print("Gold-relation recovery stability (across original + all revisions):")
    for bucket in ["fully_stable", "partially_stable", "fully_unstable"]:
        n = buckets.get(bucket, 0)
        print(f"    {bucket:<18}{n:>6}   ({100 * n / total:.1f}%)")
    print(f"    overall_relation_stability = {agg['overall_relation_stability']:.4f}"
          if agg['overall_relation_stability'] is not None else "    overall_relation_stability = n/a")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stability evaluation across original + revision extractions.")
    parser.add_argument("--originals", nargs="+", required=True, help="Original-extraction JSON files (one per model).")
    parser.add_argument("--revisions", nargs="+", required=True, help="Revisions JSON files (one per model).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Semantic similarity threshold.")
    parser.add_argument("--out", type=str, default=None, help="Path to write the full JSON report.")
    parser.add_argument("--no-embeddings", action="store_true", help="Force the lightweight difflib matcher instead of sentence-transformers.")
    args = parser.parse_args()

    originals_by_model = load_model_files(args.originals, "original")
    revisions_by_model = load_model_files(args.revisions, "revision")

    all_models = sorted(set(originals_by_model) | set(revisions_by_model))
    for model in all_models:
        if model not in originals_by_model:
            print(f"[warning] model '{model}' has revision data but no original-extraction file.", file=sys.stderr)
        if model not in revisions_by_model:
            print(f"[warning] model '{model}' has original-extraction data but no revisions file.", file=sys.stderr)

    if args.no_embeddings:
        semantic_matcher = DefaultSemanticMatcher()
    else:
        try:
            semantic_matcher = EmbeddingSemanticMatcher()
            semantic_matcher.warm_cache(collect_all_texts(originals_by_model, revisions_by_model))
            print(f"[info] Using EmbeddingSemanticMatcher on device={semantic_matcher.device}", file=sys.stderr)
        except ImportError as e:
            print(f"[warning] {e}\nFalling back to DefaultSemanticMatcher.", file=sys.stderr)
            semantic_matcher = DefaultSemanticMatcher()

    report = {"models": {}}

    for model in all_models:
        orig_stories = originals_by_model.get(model, {})
        rev_stories = revisions_by_model.get(model, {})
        all_story_ids = sorted(set(orig_stories) | set(rev_stories))

        story_results, stability_results = [], []
        for story_id in all_story_ids:
            orig_story = orig_stories.get(story_id)
            rev_story = rev_stories.get(story_id)
            # Prefer the context-rich gold from the revisions file; fall back to the original file.
            gold_relations = get_gold_relations_from_story(rev_story) if rev_story else []
            if not gold_relations and orig_story:
                gold_relations = get_gold_relations_from_story(orig_story)

            sr = evaluate_story_model(story_id, gold_relations, orig_story, rev_story, model, semantic_matcher, args.threshold)
            story_results.append(sr)
            stability_results.append(compute_stability_for_story_model(sr))

        agg = aggregate_model_results(story_results, stability_results)
        print_model_summary(model, agg)

        # Make story_results JSON-serializable (sets -> sorted lists)
        serializable_stories = []
        for sr, stab in zip(story_results, stability_results):
            cond_copy = {}
            for cond, c in sr["conditions"].items():
                c2 = dict(c)
                c2["matched_gold_indices"] = sorted(c2["matched_gold_indices"])
                cond_copy[cond] = c2
            serializable_stories.append({
                "story_id": sr["story_id"],
                "n_gold": sr["n_gold"],
                "conditions": cond_copy,
                "stability": stab,
            })

        report["models"][model] = {"aggregate": agg, "stories": serializable_stories}

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[info] Wrote full report to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()