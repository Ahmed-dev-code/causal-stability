import json
import random
import csv
from pathlib import Path
from collections import Counter, defaultdict
import statistics


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path("data/formatted_mavenere.json")
OUTPUT_JSON = Path("data/mavenere_subset_300.json")
OUTPUT_JSONL = Path("data/mavenere_subset_300.jsonl")
OUTPUT_CSV = Path("data/mavenere_subset_300_selection.csv")

SEED = 42
TARGET_SIZE = 300

MIN_TOKENS = 100
MAX_TOKENS = 500

MIN_RELATIONS = 3
MAX_RELATIONS = 30


# ============================================================
# HELPERS
# ============================================================

def load_json_or_jsonl(path):
    """
    Supports both:
      - JSON array
      - JSONL
    """

    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    # Normal JSON
    try:
        data = json.loads(content)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            return [data]

    except json.JSONDecodeError:
        pass

    # JSONL
    documents = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                documents.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: Could not parse line {line_number}: {e}")

    return documents


def get_relations(doc):
    """
    Extract relations from the simplified gold_graph.
    """

    graph = doc.get("gold_graph", {})
    relations = graph.get("relations", [])

    if not isinstance(relations, list):
        return []

    return relations


def get_token_count(doc):
    text = doc.get("text", "")
    return len(text.split())


def get_relation_counts(doc):
    relations = get_relations(doc)

    counts = Counter()

    for relation in relations:
        relation_type = relation.get("relation_type")

        if relation_type:
            counts[relation_type] += 1

    return counts


def get_relation_count(doc):
    return len(get_relations(doc))


def get_causal_composition(doc):
    """
    Categorize story according to CAUSE / PRECONDITION composition.

    Categories:
      cause_only
      precondition_only
      mixed
    """

    counts = get_relation_counts(doc)

    cause = counts.get("CAUSE", 0)
    precondition = counts.get("PRECONDITION", 0)

    if cause > 0 and precondition > 0:
        return "mixed"

    if cause > 0:
        return "cause_only"

    if precondition > 0:
        return "precondition_only"

    return "none"


def get_max_distance(doc):
    """
    Prefer metadata max distance if available.
    Otherwise calculate from maven_graph.
    """

    metadata = doc.get("metadata", {})

    if "max_sentence_distance" in metadata:
        return metadata["max_sentence_distance"]

    maven_graph = doc.get("maven_graph", {})
    relations = maven_graph.get("relations", [])

    distances = []

    for relation in relations:
        distance = relation.get("sentence_distance")

        if isinstance(distance, (int, float)):
            distances.append(distance)

    return max(distances) if distances else 0


def get_graph_structure(doc):
    metadata = doc.get("metadata", {})

    branching = metadata.get("has_branching", False)
    chain = metadata.get("has_chain", False)
    convergence = (
        metadata.get("num_convergence_nodes", 0) > 0
    )

    structures = []

    if branching:
        structures.append("branching")

    if chain:
        structures.append("chain")

    if convergence:
        structures.append("convergence")

    if not structures:
        return "simple"

    return "+".join(structures)


def length_bucket(tokens):
    if tokens <= 150:
        return "<=150"

    if tokens <= 300:
        return "151-300"

    return "301-500"


def complexity_bucket(relations):
    if relations <= 5:
        return "3-5"

    if relations <= 10:
        return "6-10"

    if relations <= 20:
        return "11-20"

    return "21-30"


def distance_bucket(distance):
    if distance <= 2:
        return "0-2"

    if distance <= 5:
        return "3-5"

    if distance <= 10:
        return "6-10"

    return "11+"


# ============================================================
# PROFILE
# ============================================================

def profile_documents(documents):

    relation_counts = Counter()
    composition_counts = Counter()

    length_counts = Counter()
    complexity_counts = Counter()
    distance_counts = Counter()
    structure_counts = Counter()

    for doc in documents:

        tokens = get_token_count(doc)
        relations = get_relation_count(doc)
        distance = get_max_distance(doc)

        relation_counts.update(get_relation_counts(doc))

        composition_counts[
            get_causal_composition(doc)
        ] += 1

        length_counts[
            length_bucket(tokens)
        ] += 1

        complexity_counts[
            complexity_bucket(relations)
        ] += 1

        distance_counts[
            distance_bucket(distance)
        ] += 1

        structure_counts[
            get_graph_structure(doc)
        ] += 1

    return {
        "relations": relation_counts,
        "composition": composition_counts,
        "length": length_counts,
        "complexity": complexity_counts,
        "distance": distance_counts,
        "structure": structure_counts,
    }


# ============================================================
# CANDIDATE FILTER
# ============================================================

def create_candidates(documents):

    candidates = []

    for doc in documents:

        tokens = get_token_count(doc)
        relations = get_relation_count(doc)

        if not (
            MIN_TOKENS <= tokens <= MAX_TOKENS
            and MIN_RELATIONS <= relations <= MAX_RELATIONS
        ):
            continue

        # Add temporary selection information
        item = {
            "doc": doc,
            "tokens": tokens,
            "relations": relations,
            "distance": get_max_distance(doc),
            "length_bucket": length_bucket(tokens),
            "complexity_bucket": complexity_bucket(relations),
            "distance_bucket": distance_bucket(
                get_max_distance(doc)
            ),
            "structure": get_graph_structure(doc),
            "composition": get_causal_composition(doc),
        }

        candidates.append(item)

    return candidates


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def proportional_targets(counter, total):

    if not counter:
        return {}

    total_available = sum(counter.values())

    raw = {}

    for key, count in counter.items():
        raw[key] = count / total_available * total

    targets = {
        key: int(value)
        for key, value in raw.items()
    }

    # Distribute remaining stories according to largest remainder
    remaining = total - sum(targets.values())

    remainders = sorted(
        counter.keys(),
        key=lambda k: raw[k] - targets[k],
        reverse=True
    )

    for key in remainders[:remaining]:
        targets[key] += 1

    return targets


# ============================================================
# MULTI-DIMENSIONAL STRATIFIED SAMPLING
# ============================================================

def select_subset(candidates, target_size):

    random.seed(SEED)

    # --------------------------------------------------------
    # 1. Preserve causal composition
    # --------------------------------------------------------

    composition_counter = Counter(
        item["composition"]
        for item in candidates
    )

    composition_targets = proportional_targets(
        composition_counter,
        target_size
    )

    print("\nCAUSAL COMPOSITION TARGETS")
    print("-" * 80)

    for key, target in composition_targets.items():
        available = composition_counter[key]

        print(
            f"{key:20s} : "
            f"{available:4d} available -> "
            f"{target:4d} target"
        )

    # --------------------------------------------------------
    # 2. For each causal composition, preserve:
    #
    #    length
    #    complexity
    #    distance
    #    graph structure
    # --------------------------------------------------------

    selected = []

    composition_groups = defaultdict(list)

    for item in candidates:
        composition_groups[
            item["composition"]
        ].append(item)

    for composition, target in composition_targets.items():

        group = composition_groups[composition]

        if not group:
            continue

        random.shuffle(group)

        # Desired distribution inside this composition
        #
        # We create a score based on how rare a candidate's
        # combination of properties is.
        #
        # This encourages broad coverage instead of randomly
        # taking only easy/simple examples.

        selected_group = []

        # Build strata
        strata = defaultdict(list)

        for item in group:

            key = (
                item["length_bucket"],
                item["complexity_bucket"],
                item["distance_bucket"],
                item["structure"],
            )

            strata[key].append(item)

        # Randomize every stratum
        for values in strata.values():
            random.shuffle(values)

        # ----------------------------------------------------
        # Round-robin over strata.
        #
        # This prevents one particular combination from
        # dominating the subset.
        # ----------------------------------------------------

        strata_items = list(strata.items())
        random.shuffle(strata_items)

        while len(selected_group) < target:

            progress = False

            for _, values in strata_items:

                if len(selected_group) >= target:
                    break

                if values:

                    selected_group.append(
                        values.pop()
                    )

                    progress = True

            if not progress:
                break

        selected.extend(selected_group)

    # --------------------------------------------------------
    # Safety fallback
    # --------------------------------------------------------

    if len(selected) < target_size:

        selected_ids = {
            item["doc"].get("story_id")
            or item["doc"].get("id")
            for item in selected
        }

        remaining = [
            item
            for item in candidates
            if (
                item["doc"].get("story_id")
                or item["doc"].get("id")
            ) not in selected_ids
        ]

        random.shuffle(remaining)

        selected.extend(
            remaining[:target_size - len(selected)]
        )

    # If somehow too many
    selected = selected[:target_size]

    # Shuffle final dataset
    random.shuffle(selected)

    return selected


# ============================================================
# FINAL STATISTICS
# ============================================================

def print_final_statistics(selected):

    print("\n" + "=" * 80)
    print("FINAL SUBSET")
    print("=" * 80)

    print(f"Stories: {len(selected)}")

    tokens = [
        x["tokens"]
        for x in selected
    ]

    relations = [
        x["relations"]
        for x in selected
    ]

    distances = [
        x["distance"]
        for x in selected
    ]

    print("\nTEXT")
    print("-" * 80)

    print(
        f"Tokens:\n"
        f"  Mean:   {statistics.mean(tokens):.2f}\n"
        f"  Median: {statistics.median(tokens):.2f}\n"
        f"  Min:    {min(tokens)}\n"
        f"  Max:    {max(tokens)}"
    )

    print("\nCAUSAL COMPLEXITY")
    print("-" * 80)

    print(
        f"Relations:\n"
        f"  Mean:   {statistics.mean(relations):.2f}\n"
        f"  Median: {statistics.median(relations):.2f}\n"
        f"  Min:    {min(relations)}\n"
        f"  Max:    {max(relations)}"
    )

    print("\nMAXIMUM CAUSAL DISTANCE")
    print("-" * 80)

    print(
        f"Mean: {statistics.mean(distances):.2f}\n"
        f"Min:  {min(distances)}\n"
        f"Max:  {max(distances)}"
    )

    # --------------------------------------------------------
    # Relation types
    # --------------------------------------------------------

    relation_counter = Counter()

    for item in selected:
        relation_counter.update(
            get_relation_counts(item["doc"])
        )

    total_relations = sum(
        relation_counter.values()
    )

    print("\nRELATION TYPES")
    print("-" * 80)

    for relation_type in [
        "CAUSE",
        "PRECONDITION"
    ]:

        count = relation_counter.get(
            relation_type,
            0
        )

        percentage = (
            count / total_relations * 100
            if total_relations
            else 0
        )

        print(
            f"{relation_type:15s}: "
            f"{count:6d} "
            f"({percentage:5.2f}%)"
        )

    print(
        f"{'TOTAL':15s}: "
        f"{total_relations:6d}"
    )

    # --------------------------------------------------------
    # Causal composition
    # --------------------------------------------------------

    composition = Counter(
        item["composition"]
        for item in selected
    )

    print("\nCAUSAL COMPOSITION")
    print("-" * 80)

    for key in [
        "cause_only",
        "precondition_only",
        "mixed"
    ]:

        count = composition.get(key, 0)

        print(
            f"{key:20s}: "
            f"{count:4d} "
            f"({count / len(selected) * 100:5.2f}%)"
        )

    # --------------------------------------------------------
    # Length
    # --------------------------------------------------------

    print("\nLENGTH DISTRIBUTION")
    print("-" * 80)

    length_counter = Counter(
        x["length_bucket"]
        for x in selected
    )

    for key in [
        "<=150",
        "151-300",
        "301-500"
    ]:

        count = length_counter[key]

        print(
            f"{key:12s} "
            f"{count:4d} "
            f"({count / len(selected) * 100:5.2f}%)"
        )

    # --------------------------------------------------------
    # Complexity
    # --------------------------------------------------------

    print("\nCAUSAL COMPLEXITY DISTRIBUTION")
    print("-" * 80)

    complexity_counter = Counter(
        x["complexity_bucket"]
        for x in selected
    )

    for key in [
        "3-5",
        "6-10",
        "11-20",
        "21-30"
    ]:

        count = complexity_counter[key]

        print(
            f"{key:12s} "
            f"{count:4d} "
            f"({count / len(selected) * 100:5.2f}%)"
        )

    # --------------------------------------------------------
    # Distance
    # --------------------------------------------------------

    print("\nCAUSAL DISTANCE DISTRIBUTION")
    print("-" * 80)

    distance_counter = Counter(
        x["distance_bucket"]
        for x in selected
    )

    for key in [
        "0-2",
        "3-5",
        "6-10",
        "11+"
    ]:

        count = distance_counter[key]

        print(
            f"{key:12s} "
            f"{count:4d} "
            f"({count / len(selected) * 100:5.2f}%)"
        )

    # --------------------------------------------------------
    # Graph structure
    # --------------------------------------------------------

    print("\nGRAPH STRUCTURE")
    print("-" * 80)

    structure_counter = Counter(
        x["structure"]
        for x in selected
    )

    for key, count in structure_counter.most_common():

        print(
            f"{key:35s} "
            f"{count:4d} "
            f"({count / len(selected) * 100:5.2f}%)"
        )


# ============================================================
# SAVE
# ============================================================

def save_outputs(selected):

    documents = [
        item["doc"]
        for item in selected
    ]

    # JSON
    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            documents,
            f,
            indent=2,
            ensure_ascii=False
        )

    # JSONL
    with open(
        OUTPUT_JSONL,
        "w",
        encoding="utf-8"
    ) as f:

        for doc in documents:

            f.write(
                json.dumps(
                    doc,
                    ensure_ascii=False
                )
                + "\n"
            )

    # CSV
    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "story_id",
                "title",
                "tokens",
                "relations",
                "cause_relations",
                "precondition_relations",
                "composition",
                "max_distance",
                "length_bucket",
                "complexity_bucket",
                "distance_bucket",
                "graph_structure",
            ]
        )

        writer.writeheader()

        for item in selected:

            relation_counts = get_relation_counts(
                item["doc"]
            )

            writer.writerow({
                "story_id":
                    item["doc"].get("story_id"),

                "title":
                    item["doc"].get("title"),

                "tokens":
                    item["tokens"],

                "relations":
                    item["relations"],

                "cause_relations":
                    relation_counts.get(
                        "CAUSE", 0
                    ),

                "precondition_relations":
                    relation_counts.get(
                        "PRECONDITION", 0
                    ),

                "composition":
                    item["composition"],

                "max_distance":
                    item["distance"],

                "length_bucket":
                    item["length_bucket"],

                "complexity_bucket":
                    item["complexity_bucket"],

                "distance_bucket":
                    item["distance_bucket"],

                "graph_structure":
                    item["structure"],
            })


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("MAVEN-ERE SUBSET CREATION")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("LOADING DATASET")
    print("=" * 80)

    documents = load_json_or_jsonl(INPUT_FILE)

    print(f"Loaded documents: {len(documents)}")

    # --------------------------------------------------------
    # Candidate pool
    # --------------------------------------------------------

    candidates = create_candidates(documents)

    print("\n" + "=" * 80)
    print("CANDIDATE POOL")
    print("=" * 80)

    print("Criteria:")
    print(f"  Tokens:             {MIN_TOKENS}-{MAX_TOKENS}")
    print(
        f"  Causal relations:   "
        f"{MIN_RELATIONS}-{MAX_RELATIONS}"
    )

    print(
        f"\nCandidate documents: "
        f"{len(candidates)} / {len(documents)}"
    )

    print(
        f"Candidate percentage: "
        f"{len(candidates) / len(documents) * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Candidate relation distribution
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("CANDIDATE RELATION DISTRIBUTION")
    print("=" * 80)

    candidate_relations = Counter()

    for item in candidates:

        candidate_relations.update(
            get_relation_counts(item["doc"])
        )

    total = sum(candidate_relations.values())

    for relation_type in [
        "CAUSE",
        "PRECONDITION"
    ]:

        count = candidate_relations.get(
            relation_type,
            0
        )

        print(
            f"{relation_type:15s}: "
            f"{count:6d} "
            f"({count / total * 100:5.2f}%)"
        )

    # --------------------------------------------------------
    # Select
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("STRATIFIED SELECTION")
    print("=" * 80)

    selected = select_subset(
        candidates,
        TARGET_SIZE
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print_final_statistics(selected)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FILES CREATED")
    print("=" * 80)

    save_outputs(selected)

    print(f"JSON:  {OUTPUT_JSON}")
    print(f"JSONL: {OUTPUT_JSONL}")
    print(f"CSV:   {OUTPUT_CSV}")

    print("\n" + "=" * 80)
    print("SUBSET CREATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()