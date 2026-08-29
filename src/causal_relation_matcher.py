#!/usr/bin/env python3
"""
Trigger-based causal-relation evaluation pipeline.

Matches LLM-predicted causal relations (natural-language cause/effect
descriptions) against MAVEN-ERE gold relations (short event triggers +
disambiguating sentence context), following this priority:

    1. Exact trigger matching   (gold trigger found as a token in the
                                  prediction, and confirmed present in the
                                  gold context)
    2. Semantic fallback        (embedding/string similarity against the
                                  gold context, only when lexical matching
                                  fails)
    3. No match

Gold relation format (NOT modified by this script):

    {
        "cause": "sold",
        "effect": "lawsuits",
        "relation_type": "PRECONDITION",
        "context": {
            "cause": "Furthermore, tickets sold extremely quickly ...",
            "effect": "Ticket scalping became so extensive ... lawsuits ..."
        }
    }

Predicted relation format:

    {
        "cause": "tickets sold extremely quickly",
        "effect": "the tour won the Breakthrough Artist award",
        "relation_type": "PRECONDITION"   # optional
    }
"""
from __future__ import annotations

import json
import re
import string
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable, List, Optional, Tuple


try:
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - matching still works without scipy
    _HAS_SCIPY = False


# --------------------------------------------------------------------------
# Tokenization / normalization
# --------------------------------------------------------------------------

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    return " ".join(text.lower().translate(_PUNCT_TABLE).split())


def tokenize(text: str) -> List[str]:
    return normalize(text).split()


def trigger_in_tokens(trigger: str, tokens: List[str]) -> bool:
    """
    Exact-token check: does `trigger` (possibly multi-word, e.g. "took place")
    occur as a contiguous sequence of exact tokens inside `tokens`?
    """
    trig_tokens = tokenize(trigger)
    if not trig_tokens or not tokens:
        return False
    n = len(trig_tokens)
    for i in range(len(tokens) - n + 1):
        if tokens[i:i + n] == trig_tokens:
            return True
    return False


def trigger_in_text(trigger: str, text: Optional[str]) -> bool:
    if not text:
        return False
    return trigger_in_tokens(trigger, tokenize(text))


# --------------------------------------------------------------------------
# Semantic similarity — pluggable
# --------------------------------------------------------------------------

class SemanticMatcher:
    """
    Interface for the semantic fallback. Plug in your existing
    embedding-based matcher by subclassing this and implementing
    `similarity`, or by passing any object/function with a compatible
    `.similarity(a, b) -> float` method into `match_relations`.
    """

    def similarity(self, text_a: str, text_b: str) -> float:
        raise NotImplementedError


class DefaultSemanticMatcher(SemanticMatcher):
    """
    Lightweight placeholder (character/word sequence similarity via
    difflib) so this script runs standalone with no extra dependencies.

    Replace this with your real embedding-based matcher, e.g.:

        from sentence_transformers import SentenceTransformer
        import numpy as np

        class EmbeddingSemanticMatcher(SemanticMatcher):
            def __init__(self, model_name="all-MiniLM-L6-v2"):
                self.model = SentenceTransformer(model_name)

            def similarity(self, text_a, text_b):
                a, b = self.model.encode([text_a, text_b])
                return float(
                    np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                )

    and pass `semantic_matcher=EmbeddingSemanticMatcher()` to
    `match_relations` / `evaluate`.
    """

    def similarity(self, text_a: str, text_b: str) -> float:
        a, b = normalize(text_a), normalize(text_b)
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()


class EmbeddingSemanticMatcher(SemanticMatcher):
    """
    Local sentence-embedding matcher using `sentence-transformers`, sized to
    run comfortably on an 8GB-VRAM GPU (e.g. RTX 5060).

    Install:
        pip install sentence-transformers

    For CUDA support matching a very recent GPU (RTX 50-series / Blackwell),
    make sure torch is installed with a CUDA build that supports it — the
    default `pip install torch` may lag behind brand-new hardware, in which
    case grab a current build from https://pytorch.org/get-started/locally/.

    Model choice:
        - "sentence-transformers/all-MiniLM-L6-v2"   ~90MB,  fastest, lower quality
        - "BAAI/bge-base-en-v1.5"                     ~440MB, good default (used here)
        - "BAAI/bge-large-en-v1.5"                    ~1.3GB, best quality, still
          trivial for 8GB VRAM if you want the extra accuracy

    All of these fit in VRAM many times over on an 8GB card even at full
    precision; fp16 is used by default purely for speed, not because it's
    needed to fit.

    Performance note: `match_relations` calls `.similarity()` once per
    (prediction, gold) pair, i.e. O(n*m) calls. Encoding one string at a
    time on a GPU wastes most of its throughput. This class caches every
    embedding it computes, and exposes `warm_cache()` to batch-encode all
    texts up front — call it once before `match_relations`/`evaluate` and
    every subsequent `.similarity()` call becomes a cache hit plus a single
    dot product.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 64,
        use_fp16: bool = True,
    ):
        try:
            import torch
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "EmbeddingSemanticMatcher requires sentence-transformers and torch. "
                "Install with: pip install sentence-transformers"
            ) from e

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        if use_fp16 and self.device == "cuda":
            self.model.half()
        self.batch_size = batch_size
        self._cache: dict = {}

    def warm_cache(self, texts: List[str]) -> None:
        """Batch-encode every unique text once, up front, on the GPU."""
        unique_texts = sorted({t for t in texts if t})
        to_encode = [t for t in unique_texts if t not in self._cache]
        if not to_encode:
            return
        embeddings = self.model.encode(
            to_encode,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for text, emb in zip(to_encode, embeddings):
            self._cache[text] = emb

    def _encode(self, text: str):
        if not text:
            text = ""
        if text not in self._cache:
            emb = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self._cache[text] = emb
        return self._cache[text]

    def similarity(self, text_a: str, text_b: str) -> float:
        if not text_a or not text_b:
            return 0.0
        a = self._encode(text_a)
        b = self._encode(text_b)
        # Embeddings are already L2-normalized, so dot product == cosine similarity.
        return float(a @ b)


def collect_texts_for_warmup(predictions: List[dict], gold_relations: List[dict]) -> List[str]:
    """
    Gather every string EmbeddingSemanticMatcher will ever be asked to
    encode for this dataset, so `warm_cache()` can batch them all in one
    GPU pass instead of encoding one-by-one during matching.
    """
    texts = []
    for pred in predictions:
        texts.append(pred.get("cause", ""))
        texts.append(pred.get("effect", ""))
    for gold in gold_relations:
        texts.append(gold.get("cause", ""))
        texts.append(gold.get("effect", ""))
        context = gold.get("context", {}) or {}
        texts.append(context.get("cause", ""))
        texts.append(context.get("effect", ""))
    return [t for t in texts if t]


def get_gold_relations(story: dict) -> List[dict]:
    """Extract gold relations from a story object, supporting both formats."""
    if not isinstance(story, dict):
        return []

    if "gold_relations" in story:
        relations = story.get("gold_relations", [])
        return relations if isinstance(relations, list) else []

    gold_graph = story.get("gold_graph", {})
    if isinstance(gold_graph, dict):
        relations = gold_graph.get("relations", [])
        return relations if isinstance(relations, list) else []

    return []


def get_prediction_relations(item: dict, model: Optional[str] = None, revision: str = "original") -> List[dict]:
    """Extract predictions for a single story from the model output structure."""
    if not isinstance(item, dict):
        return []

    if "relations" in item and isinstance(item["relations"], list):
        return item["relations"]

    models = item.get("models")
    if not isinstance(models, dict):
        return []

    if model is not None:
        model_output = models.get(model)
        if not isinstance(model_output, dict):
            return []
        revision_output = model_output.get(revision)
        if isinstance(revision_output, dict):
            relations = revision_output.get("relations")
            if isinstance(relations, list):
                return relations
        return []

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


def build_prediction_index(prediction_data) -> dict:
    """Index output records by story_id so each gold story can be matched to its own model predictions."""
    index = {}

    if isinstance(prediction_data, list):
        for item in prediction_data:
            if not isinstance(item, dict):
                continue
            story_id = item.get("story_id")
            if story_id:
                index[story_id] = item
    elif isinstance(prediction_data, dict):
        for key, value in prediction_data.items():
            if isinstance(value, dict):
                story_id = value.get("story_id") or key
                index[story_id] = value
    return index


def evaluate_story_pair(
    story_id: str,
    gold_story: dict,
    prediction_item: Optional[dict],
    semantic_matcher: Optional[SemanticMatcher],
    semantic_threshold: float,
    verbose: bool = False,
) -> dict:
    """Evaluate one story against its same-ID prediction entry."""
    gold_relations = get_gold_relations(gold_story)
    prediction_relations = get_prediction_relations(prediction_item) if prediction_item else []

    stats = evaluate(
        prediction_relations,
        gold_relations,
        semantic_matcher=semantic_matcher,
        semantic_threshold=semantic_threshold,
        verbose=verbose,
    )

    stats["story_id"] = story_id
    stats["gold_relations"] = len(gold_relations)
    stats["predicted_relations"] = len(prediction_relations)
    return stats


# --------------------------------------------------------------------------
# Per-side (cause / effect) lexical check
# --------------------------------------------------------------------------

@dataclass
class SideCheck:
    trigger: str
    predicted_text: str
    gold_context: Optional[str]
    trigger_in_prediction: bool
    trigger_in_context: bool
    lexical_match: bool
    semantic_similarity: Optional[float] = None


def check_side(
    predicted_text: str,
    gold_trigger: str,
    gold_context: Optional[str],
    semantic_matcher: Optional[SemanticMatcher],
    always_compute_semantic: bool,
) -> SideCheck:
    pred_tokens = tokenize(predicted_text)
    trigger_in_prediction = trigger_in_tokens(gold_trigger, pred_tokens)
    trigger_in_context = trigger_in_text(gold_trigger, gold_context)
    lexical_match = trigger_in_prediction and trigger_in_context

    semantic_similarity = None
    if semantic_matcher is not None and (always_compute_semantic or not lexical_match):
        # Compare the prediction against the gold context (primary signal);
        # also check the raw gold trigger and keep the stronger of the two.
        sim_context = semantic_matcher.similarity(predicted_text, gold_context or "")
        sim_trigger = semantic_matcher.similarity(predicted_text, gold_trigger)
        semantic_similarity = max(sim_context, sim_trigger)

    return SideCheck(
        trigger=gold_trigger,
        predicted_text=predicted_text,
        gold_context=gold_context,
        trigger_in_prediction=trigger_in_prediction,
        trigger_in_context=trigger_in_context,
        lexical_match=lexical_match,
        semantic_similarity=semantic_similarity,
    )


# --------------------------------------------------------------------------
# Pairwise (prediction, gold) comparison
# --------------------------------------------------------------------------

@dataclass
class PairDiagnostic:
    prediction_index: int
    gold_index: int
    cause: SideCheck
    effect: SideCheck
    relation_lexical_match: bool
    relation_semantic_match: bool
    relation_similarity: Optional[float]
    match_method: str  # "lexical" / "semantic" / "none"

    def final_match(self) -> bool:
        return self.match_method in ("lexical", "semantic")


def compare_pair(
    pred_idx: int,
    gold_idx: int,
    prediction: dict,
    gold: dict,
    semantic_matcher: Optional[SemanticMatcher],
    semantic_threshold: float,
    always_compute_semantic: bool,
) -> PairDiagnostic:
    gold_context = gold.get("context", {}) or {}

    cause_check = check_side(
        prediction.get("cause", ""),
        gold.get("cause", ""),
        gold_context.get("cause"),
        semantic_matcher,
        always_compute_semantic,
    )
    effect_check = check_side(
        prediction.get("effect", ""),
        gold.get("effect", ""),
        gold_context.get("effect"),
        semantic_matcher,
        always_compute_semantic,
    )

    relation_lexical_match = cause_check.lexical_match and effect_check.lexical_match

    match_method = "none"
    relation_semantic_match = False
    relation_similarity = None

    if relation_lexical_match:
        match_method = "lexical"
    elif semantic_matcher is not None:
        # ensure similarities were computed (they are, since lexical failed
        # and always_compute_semantic defaults handle that, but recompute
        # defensively if a side matched lexically and thus skipped it)
        if cause_check.semantic_similarity is None:
            cause_check.semantic_similarity = max(
                semantic_matcher.similarity(cause_check.predicted_text, cause_check.gold_context or ""),
                semantic_matcher.similarity(cause_check.predicted_text, cause_check.trigger),
            )
        if effect_check.semantic_similarity is None:
            effect_check.semantic_similarity = max(
                semantic_matcher.similarity(effect_check.predicted_text, effect_check.gold_context or ""),
                semantic_matcher.similarity(effect_check.predicted_text, effect_check.trigger),
            )
        relation_semantic_match = (
            cause_check.semantic_similarity >= semantic_threshold
            and effect_check.semantic_similarity >= semantic_threshold
        )
        relation_similarity = (cause_check.semantic_similarity + effect_check.semantic_similarity) / 2
        if relation_semantic_match:
            match_method = "semantic"

    return PairDiagnostic(
        prediction_index=pred_idx,
        gold_index=gold_idx,
        cause=cause_check,
        effect=effect_check,
        relation_lexical_match=relation_lexical_match,
        relation_semantic_match=relation_semantic_match,
        relation_similarity=relation_similarity,
        match_method=match_method,
    )


# --------------------------------------------------------------------------
# One-to-one assignment
# --------------------------------------------------------------------------

def _max_cardinality_assignment(pairs: List[Tuple[int, int]], n_pred: int, n_gold: int) -> List[Tuple[int, int]]:
    """Maximize the number of assigned (pred, gold) pairs among candidates."""
    if not pairs:
        return []
    if _HAS_SCIPY:
        BIG = 1e6
        cost = np.full((n_pred, n_gold), BIG)
        for p, g in pairs:
            cost[p, g] = 0.0
        row_ind, col_ind = linear_sum_assignment(cost)
        return [(int(r), int(c)) for r, c in zip(row_ind, col_ind) if cost[r, c] < BIG]

    # Fallback: simple augmenting-path bipartite matching (Kuhn's algorithm)
    adj = {}
    for p, g in pairs:
        adj.setdefault(p, []).append(g)
    match_gold = {}

    def try_assign(p, visited):
        for g in adj.get(p, []):
            if g in visited:
                continue
            visited.add(g)
            if g not in match_gold or try_assign(match_gold[g], visited):
                match_gold[g] = p
                return True
        return False

    for p in adj:
        try_assign(p, set())
    return [(p, g) for g, p in match_gold.items()]


def _max_weight_assignment(
    scored_pairs: List[Tuple[int, int, float]], n_pred: int, n_gold: int
) -> List[Tuple[int, int]]:
    """Maximize total similarity among candidate (pred, gold, score) pairs."""
    if not scored_pairs:
        return []
    if _HAS_SCIPY:
        BIG = 1e6
        cost = np.full((n_pred, n_gold), BIG)
        for p, g, score in scored_pairs:
            cost[p, g] = -score
        row_ind, col_ind = linear_sum_assignment(cost)
        return [(int(r), int(c)) for r, c in zip(row_ind, col_ind) if cost[r, c] < BIG]

    # Fallback: greedy by descending score
    used_p, used_g = set(), set()
    result = []
    for p, g, score in sorted(scored_pairs, key=lambda x: -x[2]):
        if p in used_p or g in used_g:
            continue
        used_p.add(p)
        used_g.add(g)
        result.append((p, g))
    return result


# --------------------------------------------------------------------------
# Main matching entry point
# --------------------------------------------------------------------------

def match_relations(
    predictions: List[dict],
    gold_relations: List[dict],
    semantic_matcher: Optional[SemanticMatcher] = None,
    semantic_threshold: float = 0.5,
    verbose: bool = False,
    always_compute_semantic_for_diagnostics: bool = True,
) -> Tuple[List[dict], List[int], List[int]]:
    """
    Returns (matches, unmatched_prediction_indices, unmatched_gold_indices).

    Each match dict:
        {
            "prediction_index": ...,
            "gold_index": ...,
            "cause_trigger": ...,
            "effect_trigger": ...,
            "cause_lexical_match": bool,
            "effect_lexical_match": bool,
            "cause_semantic_similarity": float | None,
            "effect_semantic_similarity": float | None,
            "relation_similarity": float | None,
            "match_method": "lexical" | "semantic",
        }
    """
    n_pred, n_gold = len(predictions), len(gold_relations)

    all_diagnostics: List[PairDiagnostic] = []
    for pi, pred in enumerate(predictions):
        for gi, gold in enumerate(gold_relations):
            diag = compare_pair(
                pi, gi, pred, gold,
                semantic_matcher, semantic_threshold,
                always_compute_semantic_for_diagnostics,
            )
            all_diagnostics.append(diag)
            if verbose:
                print_diagnostic(diag)

    diag_lookup = {(d.prediction_index, d.gold_index): d for d in all_diagnostics}

    # Phase 1: exact-trigger (lexical) matches — maximize count.
    lexical_candidates = [(d.prediction_index, d.gold_index) for d in all_diagnostics if d.match_method == "lexical"]
    lexical_assignment = _max_cardinality_assignment(lexical_candidates, n_pred, n_gold)

    matched_pred = {p for p, _ in lexical_assignment}
    matched_gold = {g for _, g in lexical_assignment}

    # Phase 2: semantic fallback among remaining predictions/gold — maximize
    # total similarity.
    semantic_candidates = [
        (d.prediction_index, d.gold_index, d.relation_similarity)
        for d in all_diagnostics
        if d.match_method == "semantic"
        and d.prediction_index not in matched_pred
        and d.gold_index not in matched_gold
    ]
    semantic_assignment = _max_weight_assignment(semantic_candidates, n_pred, n_gold)

    matches = []
    for p, g in lexical_assignment + semantic_assignment:
        d = diag_lookup[(p, g)]
        matches.append({
            "prediction_index": p,
            "gold_index": g,
            "cause_trigger": d.cause.trigger,
            "effect_trigger": d.effect.trigger,
            "cause_lexical_match": d.cause.lexical_match,
            "effect_lexical_match": d.effect.lexical_match,
            "cause_semantic_similarity": d.cause.semantic_similarity,
            "effect_semantic_similarity": d.effect.semantic_similarity,
            "relation_similarity": d.relation_similarity,
            "match_method": d.match_method,
        })
        matched_pred.add(p)
        matched_gold.add(g)

    unmatched_predictions = [i for i in range(n_pred) if i not in matched_pred]
    unmatched_gold = [i for i in range(n_gold) if i not in matched_gold]

    return matches, unmatched_predictions, unmatched_gold


# --------------------------------------------------------------------------
# Diagnostics printing
# --------------------------------------------------------------------------

def _fmt_sim(sim: Optional[float]) -> str:
    return f"{sim:.2f}" if sim is not None else "n/a"


def print_diagnostic(d: PairDiagnostic) -> None:
    print("-" * 80)
    print(f"PRED {d.prediction_index} <-> GOLD {d.gold_index}")
    print()
    print("Predicted cause:")
    print(f"    {d.cause.predicted_text}")
    print()
    print("Gold cause trigger:")
    print(f"    {d.cause.trigger}")
    print()
    print("Gold cause context:")
    print(f"    {d.cause.gold_context}")
    print()
    print(f"Cause trigger in prediction:\n    {d.cause.trigger_in_prediction}")
    print()
    print(f"Cause trigger in gold context:\n    {d.cause.trigger_in_context}")
    print()
    print(f"Cause lexical match:\n    {d.cause.lexical_match}")
    print()
    print(f"Cause semantic similarity:\n    {_fmt_sim(d.cause.semantic_similarity)}")
    print()
    print("Predicted effect:")
    print(f"    {d.effect.predicted_text}")
    print()
    print("Gold effect trigger:")
    print(f"    {d.effect.trigger}")
    print()
    print("Gold effect context:")
    print(f"    {d.effect.gold_context}")
    print()
    print(f"Effect trigger in prediction:\n    {d.effect.trigger_in_prediction}")
    print()
    print(f"Effect trigger in gold context:\n    {d.effect.trigger_in_context}")
    print()
    print(f"Effect lexical match:\n    {d.effect.lexical_match}")
    print()
    print(f"Effect semantic similarity:\n    {_fmt_sim(d.effect.semantic_similarity)}")
    print()
    print(f"Final relation match:\n    {d.final_match()}")
    print()
    print(f"Match method:\n    {d.match_method}")


# --------------------------------------------------------------------------
# Evaluation statistics
# --------------------------------------------------------------------------

def evaluate(
    predictions: List[dict],
    gold_relations: List[dict],
    semantic_matcher: Optional[SemanticMatcher] = None,
    semantic_threshold: float = 0.5,
    verbose: bool = False,
) -> dict:
    matches, unmatched_predictions, unmatched_gold = match_relations(
        predictions, gold_relations, semantic_matcher, semantic_threshold, verbose
    )

    tp = len(matches)
    fp = len(unmatched_predictions)
    fn = len(unmatched_gold)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    lexical_matches = [m for m in matches if m["match_method"] == "lexical"]
    semantic_matches = [m for m in matches if m["match_method"] == "semantic"]

    stats = {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "lexical_matches": len(lexical_matches),
        "semantic_fallback_matches": len(semantic_matches),
        "no_matches": fp,  # unmatched predictions = predictions with no accepted match
        "matches": matches,
        "unmatched_predictions": unmatched_predictions,
        "unmatched_gold": unmatched_gold,
    }
    return stats


def print_stats(stats: dict) -> None:
    print()
    print("MATCHING")
    print("-" * 80)
    print(f"Lexical matches:          {stats['lexical_matches']}")
    print(f"Semantic fallback:        {stats['semantic_fallback_matches']}")
    print(f"No match:                 {stats['no_matches']}")
    print()
    print("EVALUATION")
    print("-" * 80)
    print(f"TP:                       {stats['tp']}")
    print(f"FP:                       {stats['fp']}")
    print(f"FN:                       {stats['fn']}")
    print(f"Precision:                {stats['precision']:.4f}")
    print(f"Recall:                   {stats['recall']:.4f}")
    print(f"F1:                       {stats['f1']:.4f}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python causal_relation_matcher.py gold.json predictions.json "
            "[--threshold 0.5] [--quiet] [--out matches.json]",
            file=sys.stderr,
        )
        sys.exit(1)

    gold_path, pred_path = sys.argv[1], sys.argv[2]
    threshold = 0.5
    verbose = True
    out_path = None

    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == "--threshold":
            threshold = float(args[i + 1])
            i += 2
        elif args[i] == "--quiet":
            verbose = False
            i += 1
        elif args[i] == "--out":
            out_path = args[i + 1]
            i += 2
        else:
            i += 1

    with open(gold_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
    gold_data = gold_data
    with open(pred_path, "r", encoding="utf-8") as f:
        prediction_data = json.load(f)
    prediction_data = prediction_data

    if not isinstance(gold_data, list):
        gold_data = [gold_data]

    prediction_index = build_prediction_index(prediction_data)
    all_story_stats = []
    missing_story_ids = []


    gold_texts = []
    for story in gold_data:
        if not isinstance(story, dict):
            continue
        for rel in get_gold_relations(story):
            gold_texts.extend([
                rel.get("cause", ""),
                rel.get("effect", ""),
                (rel.get("context", {}) or {}).get("cause", ""),
                (rel.get("context", {}) or {}).get("effect", ""),
            ])

    try:
        matcher = EmbeddingSemanticMatcher()
        matcher.warm_cache([t for t in gold_texts if t])
        print(f"Using EmbeddingSemanticMatcher on device={matcher.device}", file=sys.stderr)
    except ImportError as e:
        print(f"[warning] {e}\nFalling back to DefaultSemanticMatcher (no GPU embeddings).", file=sys.stderr)
        matcher = DefaultSemanticMatcher()

    # Evaluate each story by its own story_id.
    for story in gold_data:
        if not isinstance(story, dict):
            continue

        story_id = story.get("story_id")
        if not story_id:
            continue

        prediction_item = prediction_index.get(story_id)
        if prediction_item is None:
            missing_story_ids.append(story_id)
            story_stats = evaluate_story_pair(
                story_id,
                story,
                None,
                matcher,
                threshold,
                verbose=verbose,
            )
        else:
            story_stats = evaluate_story_pair(
                story_id,
                story,
                prediction_item,
                matcher,
                threshold,
                verbose=verbose,
            )

        all_story_stats.append(story_stats)
        print(f"\nStory {story_id}: TP={story_stats['tp']} FP={story_stats['fp']} FN={story_stats['fn']} F1={story_stats['f1']:.4f}")

    total_tp = sum(s['tp'] for s in all_story_stats)
    total_fp = sum(s['fp'] for s in all_story_stats)
    total_fn = sum(s['fn'] for s in all_story_stats)

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) else 0.0
    )

    aggregated = {
        "stories": all_story_stats,
        "missing_story_ids": missing_story_ids,
        "story_count": len(all_story_stats),
        "missing_count": len(missing_story_ids),
        "overall": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
            "lexical_matches": sum(s['lexical_matches'] for s in all_story_stats),
            "semantic_fallback_matches": sum(s['semantic_fallback_matches'] for s in all_story_stats),
            "no_matches": sum(s['no_matches'] for s in all_story_stats),
        },
        "average_f1": (
            sum(s['f1'] for s in all_story_stats) / len(all_story_stats)
            if all_story_stats else 0.0
        ),
    }

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(aggregated, f, ensure_ascii=False, indent=2)
        print(f"\nWrote per-story match results to {out_path}", file=sys.stderr)
    else:
        print_stats({
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "lexical_matches": sum(s['lexical_matches'] for s in all_story_stats),
            "semantic_fallback_matches": sum(s['semantic_fallback_matches'] for s in all_story_stats),
            "no_matches": sum(s['no_matches'] for s in all_story_stats),
            "precision": overall_precision,
            "recall": overall_recall,
            "f1": overall_f1,
            "matches": [],
            "unmatched_predictions": [],
            "unmatched_gold": [],
        })


if __name__ == "__main__":
    main()