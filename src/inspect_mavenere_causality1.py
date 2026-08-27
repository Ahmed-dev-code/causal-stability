import json
import argparse
from collections import Counter


def load_jsonl(path):
    documents = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: invalid JSON on line {line_num}: {e}")

    return documents


def main(input_path):

    documents = load_jsonl(input_path)

    # ================================================================
    # GLOBAL STATISTICS
    # ================================================================

    total_documents = len(documents)

    documents_with_causal = 0

    total_relations = 0
    cause_relations = 0
    precondition_relations = 0

    same_sentence = 0
    cross_sentence = 0

    sentence_distance = Counter()

    relations_per_document = []

    # Event type statistics
    source_event_types = Counter()
    target_event_types = Counter()

    event_type_pairs = Counter()

    # Graph statistics
    documents_with_chains = 0
    documents_with_branching = 0

    # ================================================================
    # PROCESS DOCUMENTS
    # ================================================================

    for doc in documents:

        relations = doc.get("causal_relations", [])

        num_relations = len(relations)

        relations_per_document.append(num_relations)

        if num_relations > 0:
            documents_with_causal += 1

        total_relations += num_relations

        # ------------------------------------------------------------
        # Graph information already calculated by extraction script
        # ------------------------------------------------------------

        graph = doc.get("graph", {})

        if graph.get("has_chain", False):
            documents_with_chains += 1

        if graph.get("has_branching", False):
            documents_with_branching += 1

        # ------------------------------------------------------------
        # Causal relations
        # ------------------------------------------------------------

        for relation in relations:

            relation_type = relation.get("relation_type")

            # Only causal relation types
            if relation_type not in {"CAUSE", "PRECONDITION"}:
                continue

            # --------------------------------------------------------
            # Relation type
            # --------------------------------------------------------

            if relation_type == "CAUSE":
                cause_relations += 1

            elif relation_type == "PRECONDITION":
                precondition_relations += 1

            # --------------------------------------------------------
            # Sentence distance
            # --------------------------------------------------------

            distance = relation.get("sentence_distance")

            if distance is not None:

                sentence_distance[distance] += 1

                if distance == 0:
                    same_sentence += 1
                else:
                    cross_sentence += 1

            # --------------------------------------------------------
            # Event types
            # --------------------------------------------------------

            source_type = relation.get(
                "source_event_type",
                "UNKNOWN"
            )

            target_type = relation.get(
                "target_event_type",
                "UNKNOWN"
            )

            source_event_types[source_type] += 1
            target_event_types[target_type] += 1

            event_type_pairs[
                (source_type, target_type)
            ] += 1

    # ================================================================
    # OUTPUT
    # ================================================================

    print("=" * 90)
    print("MAVEN-ERE CAUSALITY-ONLY STATISTICS")
    print("=" * 90)

    # ---------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------

    print("\nDATASET")
    print("-" * 90)

    print(
        f"Documents:                         "
        f"{total_documents:,}"
    )

    print(
        f"Documents containing causal:      "
        f"{documents_with_causal:,}"
    )

    # ---------------------------------------------------------------
    # Causal relations
    # ---------------------------------------------------------------

    print("\nCAUSAL RELATIONS")
    print("-" * 90)

    print(
        f"Total causal relations:            "
        f"{total_relations:,}"
    )

    print(
        f"CAUSE:                             "
        f"{cause_relations:,}"
    )

    print(
        f"PRECONDITION:                      "
        f"{precondition_relations:,}"
    )

    if total_documents > 0:

        print(
            f"Average relations/document:       "
            f"{total_relations / total_documents:.2f}"
        )

    if documents_with_causal > 0:

        print(
            f"Average relations/causal doc:     "
            f"{total_relations / documents_with_causal:.2f}"
        )

    # ---------------------------------------------------------------
    # Sentence distance
    # ---------------------------------------------------------------

    print("\nSENTENCE DISTANCE")
    print("-" * 90)

    print(
        f"Same-sentence relations:          "
        f"{same_sentence:,}"
    )

    print(
        f"Cross-sentence relations:         "
        f"{cross_sentence:,}"
    )

    print("\nSentence distance distribution")
    print("-" * 90)

    for distance, count in sorted(sentence_distance.items()):

        percentage = (
            100 * count / total_relations
            if total_relations
            else 0
        )

        print(
            f"Distance {distance:2d}: "
            f"{count:7,} "
            f"({percentage:6.2f}%)"
        )

    # ---------------------------------------------------------------
    # Graph structure
    # ---------------------------------------------------------------

    print("\nCAUSAL GRAPH STRUCTURE")
    print("-" * 90)

    print(
        f"Documents with causal chains:     "
        f"{documents_with_chains:,}"
    )

    print(
        f"Documents with branching:         "
        f"{documents_with_branching:,}"
    )

    # ---------------------------------------------------------------
    # Relations per document
    # ---------------------------------------------------------------

    print("\nRELATIONS PER DOCUMENT")
    print("-" * 90)

    if relations_per_document:

        sorted_counts = sorted(relations_per_document)

        mean = (
            sum(sorted_counts)
            / len(sorted_counts)
        )

        median = sorted_counts[
            len(sorted_counts) // 2
        ]

        maximum = max(sorted_counts)

        print(
            f"Mean:                               "
            f"{mean:.2f}"
        )

        print(
            f"Median:                             "
            f"{median:.2f}"
        )

        print(
            f"Maximum:                            "
            f"{maximum:,}"
        )

        distribution = Counter(
            relations_per_document
        )

        print()

        for n in range(
            0,
            min(maximum, 20) + 1
        ):

            print(
                f"{n:2d} relations: "
                f"{distribution.get(n, 0):6,} documents"
            )

    # ---------------------------------------------------------------
    # Source event types
    # ---------------------------------------------------------------

    print("\nMOST COMMON CAUSE EVENT TYPES")
    print("-" * 90)

    for event_type, count in source_event_types.most_common(20):

        print(
            f"{event_type:40s} "
            f"{count:,}"
        )

    # ---------------------------------------------------------------
    # Target event types
    # ---------------------------------------------------------------

    print("\nMOST COMMON EFFECT EVENT TYPES")
    print("-" * 90)

    for event_type, count in target_event_types.most_common(20):

        print(
            f"{event_type:40s} "
            f"{count:,}"
        )

    # ---------------------------------------------------------------
    # Event type pairs
    # ---------------------------------------------------------------

    print("\nMOST COMMON CAUSAL EVENT-TYPE PAIRS")
    print("-" * 90)

    for (source, target), count in event_type_pairs.most_common(20):

        print(
            f"{source:35s} -> "
            f"{target:35s} "
            f"{count:,}"
        )

    print("\n" + "=" * 90)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Calculate causality-only statistics "
            "for the extracted MAVEN-ERE dataset."
        )
    )

    parser.add_argument(
        "input",
        help="Path to the extracted MAVEN-ERE JSONL file"
    )

    args = parser.parse_args()

    main(args.input)