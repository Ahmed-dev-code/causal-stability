import json
import os
from statistics import mean


INPUT_FILE = "data/mavenere_experiments.jsonl"
OUTPUT_FILE = "data/mavenere_base.json"


def load_jsonl(path):
    """Load a JSONL file: one JSON object per line."""

    documents = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                documents.append(json.loads(line))

            except json.JSONDecodeError as e:
                print(
                    f"Warning: invalid JSON on line "
                    f"{line_number}: {e}"
                )

    return documents


def get_event_info(doc):
    """
    Build a mapping:

        event_id -> event information

    We preserve MAVEN event identity.
    """

    events = {}

    for event in doc.get("events", []):

        event_id = event.get("event_id")

        if not event_id:
            continue

        mentions = event.get("mentions", [])

        if not mentions:
            continue

        # Use the first mention as the canonical mention.
        mention = mentions[0]

        events[event_id] = {
            "event_id": event_id,
            "event_type": event.get("type", ""),
            "type_id": event.get("type_id"),
            "mention_id": mention.get("mention_id", ""),
            "trigger": mention.get("trigger", ""),
            "sent_id": mention.get("sent_id", -1),
            "offset": mention.get("offset", [])
        }

    return events


def build_document(doc):

    events = get_event_info(doc)

    maven_relations = []

    sentence_distances = []

    for relation in doc.get("causal_relations", []):

        source_id = relation.get("source_event_id")
        target_id = relation.get("target_event_id")

        source = events.get(source_id)
        target = events.get(target_id)

        if source is None or target is None:
            continue

        relation_type = relation.get(
            "relation_type",
            ""
        )

        sentence_distance = relation.get(
            "sentence_distance",
            abs(
                source["sent_id"] -
                target["sent_id"]
            )
        )

        # --------------------------------------------------
        # Preserve the ORIGINAL MAVEN EVENT-LEVEL relation
        # --------------------------------------------------

        maven_relation = {
            "source_event_id": source_id,
            "target_event_id": target_id,

            "source_trigger": source["trigger"],
            "target_trigger": target["trigger"],

            "source_event_type": source["event_type"],
            "target_event_type": target["event_type"],

            "source_mention_id": source["mention_id"],
            "target_mention_id": target["mention_id"],

            "source_sent_id": source["sent_id"],
            "target_sent_id": target["sent_id"],

            "relation_type": relation_type,

            "sentence_distance": sentence_distance
        }

        maven_relations.append(maven_relation)

        sentence_distances.append(
            sentence_distance
        )

    # ------------------------------------------------------
    # Ignore documents without causal relations
    # ------------------------------------------------------

    if not maven_relations:
        return None

    # ------------------------------------------------------
    # Create NORMALIZED TEXTUAL GRAPH
    #
    # This is the graph your current align.py can use.
    #
    # Deduplication key:
    #
    #     cause + effect + relation_type
    #
    # We DO NOT deduplicate the MAVEN graph.
    # ------------------------------------------------------

    unique_relations = {}
    textual_relation_counts = {}

    for relation in maven_relations:

        cause = relation["source_trigger"]
        effect = relation["target_trigger"]
        relation_type = relation["relation_type"]

        key = (
            cause,
            effect,
            relation_type
        )

        textual_relation_counts[key] = (
            textual_relation_counts.get(key, 0) + 1
        )

        if key not in unique_relations:

            unique_relations[key] = {
                "cause": cause,
                "effect": effect,
                "relation_type": relation_type
            }

    gold_relations = list(
        unique_relations.values()
    )

    # ------------------------------------------------------
    # DUPLICATION STATISTICS
    # ------------------------------------------------------

    original_relation_count = len(
        maven_relations
    )

    unique_relation_count = len(
        gold_relations
    )

    duplicate_relation_count = (
        original_relation_count -
        unique_relation_count
    )

    # ------------------------------------------------------
    # Graph statistics from original MAVEN annotation
    # ------------------------------------------------------

    graph = doc.get("graph", {})

    causal_nodes = set()

    for relation in maven_relations:

        causal_nodes.add(
            relation["source_event_id"]
        )

        causal_nodes.add(
            relation["target_event_id"]
        )

    # ------------------------------------------------------
    # Relation type counts
    # ------------------------------------------------------

    num_cause = sum(
        r["relation_type"] == "CAUSE"
        for r in maven_relations
    )

    num_precondition = sum(
        r["relation_type"] == "PRECONDITION"
        for r in maven_relations
    )

    # ------------------------------------------------------
    # Build final story
    # ------------------------------------------------------

    story = {

        "story_id": (
            f"maven_{doc['document_id']}"
        ),

        "title": doc.get(
            "title",
            ""
        ),

        "text": " ".join(
            doc.get("sentences", [])
        ),

        # ==================================================
        # PRIMARY GOLD GRAPH
        #
        # Normalized textual causal propositions.
        # Used by align.py / evaluation.
        # ==================================================

        "gold_graph": {

            "relations": gold_relations

        },

        # ==================================================
        # ORIGINAL MAVEN GRAPH
        #
        # Nothing is deduplicated here.
        # ==================================================

        "maven_graph": {

            "relations": maven_relations

        },

        # ==================================================
        # METADATA
        # ==================================================

        "metadata": {

            "document_id": doc[
                "document_id"
            ],

            "title": doc.get(
                "title",
                ""
            ),

            "num_sentences": doc.get(
                "num_sentences",
                len(
                    doc.get(
                        "sentences",
                        []
                    )
                )
            ),

            "num_events": len(
                doc.get(
                    "events",
                    []
                )
            ),

            # Original MAVEN count
            "num_maven_relations":
                original_relation_count,

            # Normalized graph count
            "num_unique_textual_relations":
                unique_relation_count,

            # Number removed by normalization
            "num_duplicate_textual_relations":
                duplicate_relation_count,

            # Useful ratio
            "duplication_ratio": (
                duplicate_relation_count /
                original_relation_count
                if original_relation_count > 0
                else 0
            ),

            "num_cause_relations":
                num_cause,

            "num_precondition_relations":
                num_precondition,

            # Original MAVEN graph structure
            "num_causal_nodes":
                graph.get(
                    "num_causal_nodes",
                    len(causal_nodes)
                ),

            "num_branching_nodes":
                graph.get(
                    "num_branching_nodes",
                    0
                ),

            "num_convergence_nodes":
                graph.get(
                    "num_convergence_nodes",
                    0
                ),

            "num_chain_nodes":
                graph.get(
                    "num_chain_nodes",
                    0
                ),

            "has_branching":
                graph.get(
                    "has_branching",
                    False
                ),

            "has_chain":
                graph.get(
                    "has_chain",
                    False
                ),

            # Causal sentence information
            "num_causal_sentences":
                doc.get(
                    "num_causal_sentences",
                    0
                ),

            "causal_sentences":
                doc.get(
                    "causal_sentences",
                    []
                ),

            # Distance statistics
            "mean_sentence_distance": (
                mean(sentence_distances)
                if sentence_distances
                else 0
            ),

            "max_sentence_distance": (
                max(sentence_distances)
                if sentence_distances
                else 0
            ),

            "same_sentence_relations": sum(
                d == 0
                for d in sentence_distances
            ),

            "cross_sentence_relations": sum(
                d > 0
                for d in sentence_distances
            )
        },

        # Revisions will be added later.
        "revisions": []
    }

    return story


def main():

    print("=" * 70)
    print("MAVEN-ERE → CAUSAL STABILITY DATASET")
    print("=" * 70)

    print(f"\nInput : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")

    documents = load_jsonl(
        INPUT_FILE
    )

    print(
        f"\nInput documents: "
        f"{len(documents)}"
    )

    stories = []

    skipped = 0

    for doc in documents:

        story = build_document(doc)

        if story is None:
            skipped += 1

        else:
            stories.append(
                story
            )

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    output_directory = os.path.dirname(
        OUTPUT_FILE
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            stories,
            f,
            indent=2,
            ensure_ascii=False
        )

    # ------------------------------------------------------
    # Summary
    # ------------------------------------------------------

    maven_relations = sum(
        s["metadata"][
            "num_maven_relations"
        ]
        for s in stories
    )

    unique_relations = sum(
        s["metadata"][
            "num_unique_textual_relations"
        ]
        for s in stories
    )

    duplicates = sum(
        s["metadata"][
            "num_duplicate_textual_relations"
        ]
        for s in stories
    )

    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)

    print(
        f"Input documents:       "
        f"{len(documents)}"
    )

    print(
        f"Output stories:        "
        f"{len(stories)}"
    )

    print(
        f"Skipped documents:     "
        f"{skipped}"
    )

    print(
        f"MAVEN relations:       "
        f"{maven_relations}"
    )

    print(
        f"Unique textual edges:  "
        f"{unique_relations}"
    )

    print(
        f"Duplicate textual:     "
        f"{duplicates}"
    )

    if maven_relations:

        print(
            f"Duplicate ratio:       "
            f"{duplicates / maven_relations:.2%}"
        )

    print(
        f"\nOutput file: "
        f"{OUTPUT_FILE}"
    )

    # ------------------------------------------------------
    # Show first story
    # ------------------------------------------------------

    if stories:

        print("\n" + "=" * 70)
        print("FIRST STORY")
        print("=" * 70)

        print(
            json.dumps(
                stories[0],
                indent=2,
                ensure_ascii=False
            )
        )


if __name__ == "__main__":
    main()