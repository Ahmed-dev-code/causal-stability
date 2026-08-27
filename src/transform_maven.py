#!/usr/bin/env python3
"""
Transform MAVEN-ERE-style JSONL documents into the target causal-graph format:

story_id, title, text, gold_graph.relations, maven_graph.relations, metadata, revisions

Usage:
    python transform_maven.py input.jsonl output.jsonl
"""
import json
import sys


def rep_mention(events_by_id, event_id):
    """Pick a representative mention for an event (first mention listed)."""
    ev = events_by_id[event_id]
    m = ev["mention"][0]
    return ev, m


def build_relations(events_by_id, rel_type, pairs):
    rels = []
    for src_id, tgt_id in pairs:
        src_ev, src_m = rep_mention(events_by_id, src_id)
        tgt_ev, tgt_m = rep_mention(events_by_id, tgt_id)
        sent_dist = abs(src_m["sent_id"] - tgt_m["sent_id"])
        rels.append({
            "source_event_id": src_id,
            "target_event_id": tgt_id,
            "source_trigger": src_m["trigger_word"],
            "target_trigger": tgt_m["trigger_word"],
            "source_event_type": src_ev["type"],
            "target_event_type": tgt_ev["type"],
            "source_mention_id": src_m["id"],
            "target_mention_id": tgt_m["id"],
            "source_sent_id": src_m["sent_id"],
            "target_sent_id": tgt_m["sent_id"],
            "relation_type": rel_type,
            "sentence_distance": sent_dist,
        })
    return rels


def transform_doc(doc):
    events_by_id = {e["id"]: e for e in doc["events"]}
    causal = doc.get("causal_relations", {})

    maven_relations = (
        build_relations(events_by_id, "CAUSE", causal.get("CAUSE", []))
        + build_relations(events_by_id, "PRECONDITION", causal.get("PRECONDITION", []))
    )

    # gold_graph: normalized textual (trigger-word level) propositions,
    # one entry per maven_graph relation, in the same order.
    gold_relations = [
        {"cause": r["source_trigger"], "effect": r["target_trigger"], "relation_type": r["relation_type"]}
        for r in maven_relations
    ]

    seen = set()
    for r in gold_relations:
        seen.add((r["cause"], r["effect"], r["relation_type"]))

    num_maven = len(maven_relations)
    num_unique_textual = len(seen)
    num_duplicate = num_maven - num_unique_textual
    duplication_ratio = round(num_duplicate / num_maven, 4) if num_maven else 0.0

    num_cause = sum(1 for r in maven_relations if r["relation_type"] == "CAUSE")
    num_precondition = sum(1 for r in maven_relations if r["relation_type"] == "PRECONDITION")

    causal_nodes = set()
    out_deg, in_deg = {}, {}
    for r in maven_relations:
        s, t = r["source_event_id"], r["target_event_id"]
        causal_nodes.add(s)
        causal_nodes.add(t)
        out_deg[s] = out_deg.get(s, 0) + 1
        in_deg[t] = in_deg.get(t, 0) + 1

    branching_nodes = {n for n in causal_nodes if out_deg.get(n, 0) > 1}
    convergence_nodes = {n for n in causal_nodes if in_deg.get(n, 0) > 1}
    chain_nodes = {n for n in causal_nodes if in_deg.get(n, 0) >= 1 and out_deg.get(n, 0) >= 1}

    causal_sentences = sorted({r["source_sent_id"] for r in maven_relations} | {r["target_sent_id"] for r in maven_relations})

    distances = [r["sentence_distance"] for r in maven_relations]
    mean_sentence_distance = round(sum(distances) / len(distances), 4) if distances else 0.0
    max_sentence_distance = max(distances) if distances else 0
    same_sentence_relations = sum(1 for d in distances if d == 0)
    cross_sentence_relations = sum(1 for d in distances if d > 0)

    story_id = doc.get("id") or doc.get("story_id")
    text = " ".join(doc["sentences"])

    return {
        "story_id": story_id,
        "title": doc["title"],
        "text": text,
        "gold_graph": {
            "relations": gold_relations
        },
        "maven_graph": {
            "relations": maven_relations
        },
        "metadata": {
            "document_id": story_id,
            "title": doc["title"],
            "num_sentences": len(doc["sentences"]),
            "num_events": len(doc["events"]),
            "num_maven_relations": num_maven,
            "num_unique_textual_relations": num_unique_textual,
            "num_duplicate_textual_relations": num_duplicate,
            "duplication_ratio": duplication_ratio,
            "num_cause_relations": num_cause,
            "num_precondition_relations": num_precondition,
            "num_causal_nodes": len(causal_nodes),
            "num_branching_nodes": len(branching_nodes),
            "num_convergence_nodes": len(convergence_nodes),
            "num_chain_nodes": len(chain_nodes),
            "has_branching": len(branching_nodes) > 0,
            "has_chain": len(chain_nodes) > 0,
            "num_causal_sentences": len(causal_sentences),
            "causal_sentences": causal_sentences,
            "mean_sentence_distance": mean_sentence_distance,
            "max_sentence_distance": max_sentence_distance,
            "same_sentence_relations": same_sentence_relations,
            "cross_sentence_relations": cross_sentence_relations,
        },
        "revisions": [],
    }


def main():
    if len(sys.argv) != 3:
        print("Usage: python transform_maven.py input.jsonl output.json", file=sys.stderr)
        sys.exit(1)

    in_path, out_path = sys.argv[1], sys.argv[2]
    n_ok, n_err = 0, 0
    out_docs = []
    with open(in_path, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                out_docs.append(transform_doc(doc))
                n_ok += 1
            except Exception as e:
                print(f"[line {line_no}] skipped: {e}", file=sys.stderr)
                n_err += 1

    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(out_docs, fout, ensure_ascii=False, indent=2)

    print(f"Done. {n_ok} documents transformed, {n_err} skipped. Wrote JSON array to {out_path}.", file=sys.stderr)


if __name__ == "__main__":
    main()