import json
import csv
import argparse
from collections import Counter


# ================================================================
# LOAD JSONL
# ================================================================

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


# ================================================================
# MAIN
# ================================================================

def main(input_path, pairs_csv, percentages_csv):

    documents = load_jsonl(input_path)

    # ------------------------------------------------------------
    # Dataset statistics
    # ------------------------------------------------------------

    total_documents = len(documents)
    documents_with_causal = 0

    total_relations = 0
    cause_relations = 0
    precondition_relations = 0

    same_sentence = 0
    cross_sentence = 0

    sentence_distance = Counter()

    relations_per_document = []

    # ------------------------------------------------------------
    # Event statistics
    # ------------------------------------------------------------

    cause_event_types = Counter()
    effect_event_types = Counter()

    # (cause type, effect type)
    event_type_pairs = Counter()

    # ------------------------------------------------------------
    # Graph statistics
    # ------------------------------------------------------------

    documents_with_chains = 0
    documents_with_branching = 0

    # ================================================================
    # PROCESS DOCUMENTS
    # ================================================================

    for doc in documents:

        relations = doc.get("causal_relations", [])

        # Only CAUSE and PRECONDITION relations
        relations = [
            r for r in relations
            if r.get("relation_type") in {
                "CAUSE",
                "PRECONDITION"
            }
        ]

        num_relations = len(relations)

        relations_per_document.append(num_relations)

        if num_relations > 0:
            documents_with_causal += 1

        total_relations += num_relations

        # ------------------------------------------------------------
        # Graph
        # ------------------------------------------------------------

        graph = doc.get("graph", {})

        if graph.get("has_chain", False):
            documents_with_chains += 1

        if graph.get("has_branching", False):
            documents_with_branching += 1

        # ------------------------------------------------------------
        # Relations
        # ------------------------------------------------------------

        for relation in relations:

            relation_type = relation.get("relation_type")

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

            cause_event_types[source_type] += 1
            effect_event_types[target_type] += 1

            event_type_pairs[
                (source_type, target_type)
            ] += 1

    # ================================================================
    # WRITE PAIR CSV
    # ================================================================

    with open(
        pairs_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "cause_event_type",
            "effect_event_type",
            "count"
        ])

        for (cause, effect), count in event_type_pairs.most_common():

            writer.writerow([
                cause,
                effect,
                count
            ])

    # ================================================================
    # WRITE PERCENTAGE CSV
    # ================================================================

    with open(
        percentages_csv,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "cause_event_type",
            "effect_event_type",
            "count",
            "pct_all_causal_relations",
            "pct_of_cause_type"
        ])

        for (cause, effect), count in event_type_pairs.most_common():

            # Percentage among ALL causal relations
            pct_all = (
                100 * count / total_relations
                if total_relations
                else 0
            )

            # Percentage among relations originating
            # from this particular cause type
            total_from_cause = cause_event_types[cause]

            pct_cause = (
                100 * count / total_from_cause
                if total_from_cause
                else 0
            )

            writer.writerow([
                cause,
                effect,
                count,
                f"{pct_all:.4f}",
                f"{pct_cause:.4f}"
            ])

    # ================================================================
    # PRINT STATISTICS
    # ================================================================

    print()
    print("=" * 90)
    print("MAVEN-ERE CAUSALITY-ONLY STATISTICS")
    print("=" * 90)

    # ---------------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------------

    print()
    print("DATASET")
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

    print()
    print("CAUSAL RELATIONS")
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

    if total_documents:

        print(
            f"Average relations/document:       "
            f"{total_relations / total_documents:.2f}"
        )

    if documents_with_causal:

        print(
            f"Average relations/causal doc:     "
            f"{total_relations / documents_with_causal:.2f}"
        )

    # ---------------------------------------------------------------
    # Sentence distance
    # ---------------------------------------------------------------

    print()
    print("SENTENCE DISTANCE")
    print("-" * 90)

    print(
        f"Same-sentence relations:          "
        f"{same_sentence:,}"
    )

    print(
        f"Cross-sentence relations:         "
        f"{cross_sentence:,}"
    )

    print()
    print("Sentence distance distribution")
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

    print()
    print("CAUSAL GRAPH STRUCTURE")
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

    print()
    print("RELATIONS PER DOCUMENT")
    print("-" * 90)

    if relations_per_document:

        mean = (
            sum(relations_per_document)
            / len(relations_per_document)
        )

        sorted_counts = sorted(relations_per_document)

        n = len(sorted_counts)

        if n % 2 == 1:
            median = sorted_counts[n // 2]
        else:
            median = (
                sorted_counts[n // 2 - 1]
                + sorted_counts[n // 2]
            ) / 2

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

        for number in range(
            0,
            min(maximum, 20) + 1
        ):

            print(
                f"{number:2d} relations: "
                f"{distribution.get(number, 0):6,} documents"
            )

    # ---------------------------------------------------------------
    # Cause event types
    # ---------------------------------------------------------------

    print()
    print("MOST COMMON CAUSE EVENT TYPES")
    print("-" * 90)

    for event_type, count in cause_event_types.most_common(20):

        percentage = (
            100 * count / total_relations
            if total_relations
            else 0
        )

        print(
            f"{event_type:40s} "
            f"{count:7,} "
            f"({percentage:6.2f}%)"
        )

    # ---------------------------------------------------------------
    # Effect event types
    # ---------------------------------------------------------------

    print()
    print("MOST COMMON EFFECT EVENT TYPES")
    print("-" * 90)

    for event_type, count in effect_event_types.most_common(20):

        percentage = (
            100 * count / total_relations
            if total_relations
            else 0
        )

        print(
            f"{event_type:40s} "
            f"{count:7,} "
            f"({percentage:6.2f}%)"
        )

    # ---------------------------------------------------------------
    # Event type pairs
    # ---------------------------------------------------------------

    print()
    print("MOST COMMON CAUSAL EVENT-TYPE PAIRS")
    print("-" * 90)

    print(
        f"{'CAUSE':35s} -> "
        f"{'EFFECT':35s} "
        f"{'COUNT':>8s} "
        f"{'%':>8s}"
    )

    print("-" * 90)

    for (cause, effect), count in event_type_pairs.most_common(30):

        percentage = (
            100 * count / total_relations
            if total_relations
            else 0
        )

        print(
            f"{cause:35s} -> "
            f"{effect:35s} "
            f"{count:8,} "
            f"{percentage:7.2f}%"
        )

    # ---------------------------------------------------------------
    # Output files
    # ---------------------------------------------------------------

    print()
    print("=" * 90)
    print("OUTPUT FILES")
    print("=" * 90)

    print(
        f"Pair counts:        {pairs_csv}"
    )

    print(
        f"Pair percentages:   {percentages_csv}"
    )

    print("=" * 90)


# ================================================================
# CLI
# ================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Calculate causal event-pair statistics "
            "for MAVEN-ERE."
        )
    )

    parser.add_argument(
        "input",
        help="Input MAVEN-ERE JSONL file"
    )

    parser.add_argument(
        "--pairs-csv",
        default="data/mavenere_causal_event_pairs.csv",
        help="Output CSV containing causal event-type pair counts"
    )

    parser.add_argument(
        "--percentages-csv",
        default="data/mavenere_causal_event_pairs_percentages.csv",
        help="Output CSV containing pair percentages"
    )

    args = parser.parse_args()

    main(
        args.input,
        args.pairs_csv,
        args.percentages_csv
    )