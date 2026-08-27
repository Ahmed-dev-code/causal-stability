import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import statistics


def load_jsonl(path):
    """Load a JSONL file."""
    documents = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: Could not parse line {line_num}: {e}")

    return documents


def build_event_index(doc):
    """
    Map event ID -> event information.

    Each event may contain multiple coreferential mentions.
    For our analysis, we keep all mentions.
    """
    event_index = {}

    for event in doc.get("events", []):
        event_id = event["id"]

        event_index[event_id] = {
            "id": event_id,
            "type": event.get("type"),
            "type_id": event.get("type_id"),
            "mentions": event.get("mention", []),
        }

    return event_index


def get_event_location(event):
    """
    Return sentence IDs and trigger words for an event.
    """
    sent_ids = []
    triggers = []

    for mention in event.get("mentions", []):
        if "sent_id" in mention:
            sent_ids.append(mention["sent_id"])

        if "trigger_word" in mention:
            triggers.append(mention["trigger_word"])

    return sent_ids, triggers


def get_causal_relations(doc):
    """
    Extract CAUSE and PRECONDITION relations.

    Returns a list of:
        {
            source_id,
            target_id,
            relation_type
        }
    """
    causal = doc.get("causal_relations", {})

    relations = []

    for relation_type in ["CAUSE", "PRECONDITION"]:
        for pair in causal.get(relation_type, []):
            if len(pair) != 2:
                continue

            relations.append({
                "source_id": pair[0],
                "target_id": pair[1],
                "relation_type": relation_type,
            })

    return relations


def relation_details(relation, event_index):
    """
    Add readable information to a causal relation.
    """

    source_id = relation["source_id"]
    target_id = relation["target_id"]

    source = event_index.get(source_id)
    target = event_index.get(target_id)

    if source is None or target is None:
        return {
            **relation,
            "source_trigger": "[UNKNOWN]",
            "target_trigger": "[UNKNOWN]",
            "source_type": "[UNKNOWN]",
            "target_type": "[UNKNOWN]",
            "source_sentences": [],
            "target_sentences": [],
        }

    source_sents, source_triggers = get_event_location(source)
    target_sents, target_triggers = get_event_location(target)

    return {
        **relation,
        "source_trigger": " / ".join(source_triggers),
        "target_trigger": " / ".join(target_triggers),
        "source_type": source["type"],
        "target_type": target["type"],
        "source_sentences": source_sents,
        "target_sentences": target_sents,
    }


def analyze(documents):
    stats = {
        "documents": len(documents),
        "documents_with_causal": 0,
        "documents_with_multiple_causal": 0,

        "sentences": 0,
        "events": 0,
        "event_mentions": 0,

        "causal_relations": 0,
        "cause_relations": 0,
        "precondition_relations": 0,

        "same_sentence_relations": 0,
        "cross_sentence_relations": 0,

        "relation_distance": Counter(),
        "relations_per_document": [],

        "causal_event_types": Counter(),
        "causal_pairs": Counter(),

        "documents_with_chains": 0,
        "documents_with_branching": 0,
    }

    detailed_relations = []

    for doc in documents:

        sentences = doc.get("sentences", [])
        events = doc.get("events", [])

        stats["sentences"] += len(sentences)
        stats["events"] += len(events)

        for event in events:
            stats["event_mentions"] += len(event.get("mention", []))

        event_index = build_event_index(doc)

        relations = get_causal_relations(doc)

        num_relations = len(relations)

        stats["causal_relations"] += num_relations
        stats["relations_per_document"].append(num_relations)

        if num_relations > 0:
            stats["documents_with_causal"] += 1

        if num_relations > 1:
            stats["documents_with_multiple_causal"] += 1

        # --------------------------------------------------
        # Analyze causal graph
        # --------------------------------------------------

        graph = defaultdict(set)
        indegree = Counter()
        outdegree = Counter()

        for relation in relations:

            source = relation["source_id"]
            target = relation["target_id"]

            graph[source].add(target)

            outdegree[source] += 1
            indegree[target] += 1

            if relation["relation_type"] == "CAUSE":
                stats["cause_relations"] += 1
            elif relation["relation_type"] == "PRECONDITION":
                stats["precondition_relations"] += 1

            source_event = event_index.get(source)
            target_event = event_index.get(target)

            if source_event and target_event:

                source_sents, source_triggers = get_event_location(source_event)
                target_sents, target_triggers = get_event_location(target_event)

                if source_sents and target_sents:

                    # Use the closest pair of mentions.
                    distances = [
                        abs(a - b)
                        for a in source_sents
                        for b in target_sents
                    ]

                    distance = min(distances)

                    stats["relation_distance"][distance] += 1

                    if distance == 0:
                        stats["same_sentence_relations"] += 1
                    else:
                        stats["cross_sentence_relations"] += 1

                stats["causal_event_types"][source_event["type"]] += 1
                stats["causal_event_types"][target_event["type"]] += 1

                stats["causal_pairs"][
                    (
                        source_event["type"],
                        target_event["type"]
                    )
                ] += 1

            detailed_relations.append(
                relation_details(relation, event_index)
                | {
                    "doc_id": doc.get("id"),
                    "title": doc.get("title"),
                    "sentences": sentences,
                }
            )

        # --------------------------------------------------
        # Detect branching
        # --------------------------------------------------

        if any(degree > 1 for degree in outdegree.values()):
            stats["documents_with_branching"] += 1

        # --------------------------------------------------
        # Detect causal chains
        #
        # A -> B -> C
        # --------------------------------------------------

        has_chain = False

        for middle in graph:

            predecessors = [
                source
                for source, targets in graph.items()
                if middle in targets
            ]

            successors = graph[middle]

            if predecessors and successors:
                has_chain = True
                break

        if has_chain:
            stats["documents_with_chains"] += 1

    return stats, detailed_relations


def print_stats(stats):

    print("=" * 90)
    print("MAVEN-ERE CAUSAL RELATION STATISTICS")
    print("=" * 90)

    print()

    print("DATASET")
    print("-" * 90)

    print(f"Documents:                         {stats['documents']:,}")
    print(f"Sentences:                         {stats['sentences']:,}")
    print(f"Events:                            {stats['events']:,}")
    print(f"Event mentions:                    {stats['event_mentions']:,}")

    print()

    print("CAUSAL RELATIONS")
    print("-" * 90)

    print(f"Total causal relations:            {stats['causal_relations']:,}")
    print(f"CAUSE:                             {stats['cause_relations']:,}")
    print(f"PRECONDITION:                      {stats['precondition_relations']:,}")

    if stats["documents"] > 0:
        print(
            f"Documents containing causal:      "
            f"{stats['documents_with_causal']:,} "
            f"({100 * stats['documents_with_causal'] / stats['documents']:.2f}%)"
        )

    print(
        f"Documents with >1 causal relation: "
        f"{stats['documents_with_multiple_causal']:,}"
    )

    print()

    print("SENTENCE DISTANCE")
    print("-" * 90)

    print(
        f"Same-sentence relations:          "
        f"{stats['same_sentence_relations']:,}"
    )

    print(
        f"Cross-sentence relations:         "
        f"{stats['cross_sentence_relations']:,}"
    )

    print()

    if stats["relation_distance"]:

        print("Sentence distance distribution")
        print("-" * 90)

        total = sum(stats["relation_distance"].values())

        for distance in sorted(stats["relation_distance"]):

            count = stats["relation_distance"][distance]
            percentage = 100 * count / total

            print(
                f"Distance {distance:>2}: "
                f"{count:>7,} "
                f"({percentage:6.2f}%)"
            )

    print()

    print("CAUSAL GRAPH STRUCTURE")
    print("-" * 90)

    print(
        f"Documents with causal chains:     "
        f"{stats['documents_with_chains']:,}"
    )

    print(
        f"Documents with branching:         "
        f"{stats['documents_with_branching']:,}"
    )

    print()

    if stats["relations_per_document"]:

        values = stats["relations_per_document"]

        print("RELATIONS PER DOCUMENT")
        print("-" * 90)

        print(f"Mean:                               {statistics.mean(values):.2f}")
        print(f"Median:                             {statistics.median(values):.2f}")
        print(f"Maximum:                            {max(values):,}")

        distribution = Counter(values)

        for n in sorted(distribution)[:20]:

            count = distribution[n]

            print(
                f"{n:>3} relations: "
                f"{count:>7,} documents"
            )

    print()

    print("MOST COMMON EVENT TYPES IN CAUSAL RELATIONS")
    print("-" * 90)

    for event_type, count in stats["causal_event_types"].most_common(20):

        print(
            f"{event_type:<35} {count:>8,}"
        )

    print()

    print("MOST COMMON CAUSAL EVENT-TYPE PAIRS")
    print("-" * 90)

    for (source, target), count in stats["causal_pairs"].most_common(20):

        print(
            f"{source:<30} -> {target:<30} {count:>7,}"
        )

    print()

    print("=" * 90)


def print_examples(relations, n=20):

    print()
    print("=" * 90)
    print(f"SAMPLE CAUSAL RELATIONS ({n})")
    print("=" * 90)

    shown = 0

    for relation in relations:

        if shown >= n:
            break

        print()
        print(f"Document: {relation['title']}")
        print(f"Type:     {relation['relation_type']}")

        print(
            f"Source:   {relation['source_trigger']} "
            f"({relation['source_type']})"
        )

        print(
            f"Target:   {relation['target_trigger']} "
            f"({relation['target_type']})"
        )

        print(
            f"Sentence: {relation['source_sentences']} -> "
            f"{relation['target_sentences']}"
        )

        for sent_id in sorted(
            set(
                relation["source_sentences"]
                + relation["target_sentences"]
            )
        ):

            if 0 <= sent_id < len(relation["sentences"]):

                print(
                    f"  [{sent_id}] "
                    f"{relation['sentences'][sent_id]}"
                )

        shown += 1

    print()


def main():

    parser = argparse.ArgumentParser(
        description="Calculate MAVEN-ERE causal relation statistics."
    )

    parser.add_argument(
        "input",
        type=str,
        help="Path to MAVEN-ERE JSONL file"
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=20,
        help="Number of causal examples to display"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path"
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {input_path}"
        )

    print(f"Loading: {input_path}")

    documents = load_jsonl(input_path)

    print(f"Loaded {len(documents):,} documents.")
    print()

    stats, relations = analyze(documents)

    print_stats(stats)

    print_examples(
        relations,
        n=args.examples
    )

    if args.output:

        output = {
            "documents": stats["documents"],
            "sentences": stats["sentences"],
            "events": stats["events"],
            "event_mentions": stats["event_mentions"],
            "causal_relations": stats["causal_relations"],
            "cause_relations": stats["cause_relations"],
            "precondition_relations": stats["precondition_relations"],
            "documents_with_causal": stats["documents_with_causal"],
            "documents_with_multiple_causal": stats[
                "documents_with_multiple_causal"
            ],
            "same_sentence_relations": stats[
                "same_sentence_relations"
            ],
            "cross_sentence_relations": stats[
                "cross_sentence_relations"
            ],
            "relation_distance": dict(
                stats["relation_distance"]
            ),
            "documents_with_chains": stats[
                "documents_with_chains"
            ],
            "documents_with_branching": stats[
                "documents_with_branching"
            ],
            "relations_per_document": stats[
                "relations_per_document"
            ],
        }

        with open(
            args.output,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                output,
                f,
                indent=2
            )

        print()
        print(f"Statistics saved to: {args.output}")


if __name__ == "__main__":
    main()