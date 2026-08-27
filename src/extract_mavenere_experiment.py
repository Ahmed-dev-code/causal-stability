import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_jsonl(path):
    """Load MAVEN-ERE JSONL file."""
    documents = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARNING] Invalid JSON on line {line_num}: {e}")

    return documents


def build_event_index(doc):
    """
    Build:

        event_id -> event information

    Each event can have multiple mentions because MAVEN contains
    coreferential event mentions.
    """
    events = {}

    for event in doc.get("events", []):
        event_id = event["id"]

        mentions = []

        for mention in event.get("mention", []):
            mentions.append({
                "mention_id": mention["id"],
                "trigger": mention["trigger_word"],
                "sent_id": mention["sent_id"],
                "offset": mention["offset"],
            })

        events[event_id] = {
            "event_id": event_id,
            "type": event.get("type"),
            "type_id": event.get("type_id"),
            "mentions": mentions,
        }

    return events


def extract_causal_relations(doc, events):
    """
    Convert MAVEN-ERE causal relations from event IDs into
    readable causal relation objects.

    For each event pair we keep all combinations of event mentions.
    """

    relations = []

    causal_relations = doc.get("causal_relations", {})

    for relation_type in ["CAUSE", "PRECONDITION"]:

        for pair in causal_relations.get(relation_type, []):

            if len(pair) != 2:
                continue

            source_id, target_id = pair

            if source_id not in events or target_id not in events:
                continue

            source_event = events[source_id]
            target_event = events[target_id]

            source_mentions = source_event["mentions"]
            target_mentions = target_event["mentions"]

            # Normally there should be mentions, but keep the
            # relation even if annotation is incomplete.
            if not source_mentions or not target_mentions:
                relations.append({
                    "relation_type": relation_type,
                    "source_event_id": source_id,
                    "target_event_id": target_id,
                    "source_event_type": source_event["type"],
                    "target_event_type": target_event["type"],
                    "source_mention": None,
                    "target_mention": None,
                    "sentence_distance": None,
                })
                continue

            for source_mention in source_mentions:
                for target_mention in target_mentions:

                    distance = abs(
                        source_mention["sent_id"]
                        - target_mention["sent_id"]
                    )

                    relations.append({
                        "relation_type": relation_type,

                        "source_event_id": source_id,
                        "target_event_id": target_id,

                        "source_event_type": source_event["type"],
                        "target_event_type": target_event["type"],

                        "source_mention": source_mention,
                        "target_mention": target_mention,

                        "sentence_distance": distance,
                    })

    return relations


def calculate_graph_properties(relations):
    """
    Calculate simple causal graph statistics.

    We treat both CAUSE and PRECONDITION as causal edges here.
    """

    outgoing = defaultdict(set)
    incoming = defaultdict(set)

    for relation in relations:
        source = relation["source_event_id"]
        target = relation["target_event_id"]

        outgoing[source].add(target)
        incoming[target].add(source)

    # Nodes participating in causal graph
    nodes = set(outgoing.keys()) | set(incoming.keys())

    # Branching:
    # one event causes/preconditions multiple events
    branching_nodes = {
        node for node, targets in outgoing.items()
        if len(targets) > 1
    }

    # Convergence:
    # multiple events lead to the same event
    convergence_nodes = {
        node for node, sources in incoming.items()
        if len(sources) > 1
    }

    # A rough chain indicator:
    # at least one node has both incoming and outgoing edges.
    chain_nodes = {
        node for node in nodes
        if incoming.get(node) and outgoing.get(node)
    }

    return {
        "num_causal_nodes": len(nodes),
        "num_branching_nodes": len(branching_nodes),
        "num_convergence_nodes": len(convergence_nodes),
        "num_chain_nodes": len(chain_nodes),
        "has_branching": len(branching_nodes) > 0,
        "has_chain": len(chain_nodes) > 0,
    }


def extract_document(doc):
    """
    Extract one MAVEN-ERE document into the experimental format.
    """

    events = build_event_index(doc)

    relations = extract_causal_relations(doc, events)

    if not relations:
        return None

    graph_stats = calculate_graph_properties(relations)

    # Keep the original document text.
    sentences = doc.get("sentences", [])

    # Create a compact event representation.
    event_list = []

    for event in events.values():

        event_list.append({
            "event_id": event["event_id"],
            "type": event["type"],
            "type_id": event["type_id"],
            "mentions": event["mentions"],
        })

    # Sentence-level causal density
    causal_sentences = set()

    for relation in relations:

        source = relation["source_mention"]
        target = relation["target_mention"]

        if source:
            causal_sentences.add(source["sent_id"])

        if target:
            causal_sentences.add(target["sent_id"])

    return {
        "document_id": doc.get("id"),
        "title": doc.get("title"),

        "sentences": sentences,

        "num_sentences": len(sentences),

        "events": event_list,

        "causal_relations": relations,

        "num_causal_relations": len(relations),

        "graph": graph_stats,

        "causal_sentences": sorted(causal_sentences),

        "num_causal_sentences": len(causal_sentences),
    }


def main():

    parser = argparse.ArgumentParser(
        description="Extract causal experimental examples from MAVEN-ERE"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to MAVEN-ERE JSONL file"
    )

    parser.add_argument(
        "--output",
        default="data/mavenere_experiments.jsonl",
        help="Output JSONL file"
    )

    parser.add_argument(
        "--min-relations",
        type=int,
        default=1,
        help="Minimum number of causal relations per document"
    )

    parser.add_argument(
        "--max-relations",
        type=int,
        default=None,
        help="Optional maximum number of causal relations"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MAVEN-ERE EXPERIMENT EXTRACTION")
    print("=" * 80)

    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print()

    documents = load_jsonl(input_path)

    print(f"Loaded documents: {len(documents):,}")

    extracted = []

    for doc in documents:

        result = extract_document(doc)

        if result is None:
            continue

        num_relations = result["num_causal_relations"]

        if num_relations < args.min_relations:
            continue

        if (
            args.max_relations is not None
            and num_relations > args.max_relations
        ):
            continue

        extracted.append(result)

    with open(output_path, "w", encoding="utf-8") as f:

        for item in extracted:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False
                )
                + "\n"
            )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    total_relations = sum(
        x["num_causal_relations"]
        for x in extracted
    )

    documents_with_chains = sum(
        x["graph"]["has_chain"]
        for x in extracted
    )

    documents_with_branching = sum(
        x["graph"]["has_branching"]
        for x in extracted
    )

    print()
    print("=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)

    print(f"Documents extracted:       {len(extracted):,}")
    print(f"Causal relations:          {total_relations:,}")

    print(
        f"Documents with chains:     "
        f"{documents_with_chains:,}"
    )

    print(
        f"Documents with branching:  "
        f"{documents_with_branching:,}"
    )

    if extracted:

        avg_relations = total_relations / len(extracted)

        print(
            f"Average relations/doc:     "
            f"{avg_relations:.2f}"
        )

    print()
    print(f"Saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()