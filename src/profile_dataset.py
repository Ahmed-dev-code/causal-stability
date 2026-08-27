import json
import os
import statistics
from collections import Counter


INPUT_FILE = r"data\formatted_mavenere.json"

# Candidate subset constraints
MAX_TOKENS = 500
MIN_RELATIONS = 3
MAX_RELATIONS = 30


# ============================================================
# LOADING
# ============================================================

def load_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        return []

    # Regular JSON
    if content.startswith("[") or content.startswith("{"):
        try:
            data = json.loads(content)

            if isinstance(data, list):
                return data

            if isinstance(data, dict):
                return [data]

        except json.JSONDecodeError:
            pass

    # JSONL fallback
    documents = []

    for line_number, line in enumerate(content.splitlines(), 1):
        line = line.strip()

        if not line:
            continue

        try:
            documents.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"ERROR on line {line_number}: {e}")
            raise

    return documents


# ============================================================
# HELPERS
# ============================================================

def token_count(text):
    return len(text.split())


def get_relations(doc):
    graph = doc.get("gold_graph", {})

    if isinstance(graph, dict):
        return graph.get("relations", [])

    return []


def get_relation_type_counts(doc):
    counts = Counter()

    for relation in get_relations(doc):
        relation_type = relation.get("relation_type")

        if relation_type:
            counts[relation_type] += 1

    return counts


def percentile(values, p):

    if not values:
        return 0

    values = sorted(values)

    k = (len(values) - 1) * (p / 100)

    f = int(k)
    c = min(f + 1, len(values) - 1)

    if f == c:
        return values[f]

    return values[f] + (values[c] - values[f]) * (k - f)


def print_distribution(name, values):

    print(f"\n{name}")
    print("-" * 90)

    if not values:
        print("No data.")
        return

    print(f"Mean:       {statistics.mean(values):.2f}")
    print(f"Median:     {statistics.median(values):.2f}")

    for p in [25, 50, 75, 90, 95, 99]:
        print(f"P{p:<3}:        {percentile(values, p):.2f}")

    print(f"Minimum:    {min(values):.2f}")
    print(f"Maximum:    {max(values):.2f}")


# ============================================================
# PROFILE DOCUMENTS
# ============================================================

def profile_documents(documents):

    profiles = []

    for doc in documents:

        text = doc.get("text", "")

        metadata = doc.get("metadata", {})

        relations = get_relations(doc)

        relation_types = get_relation_type_counts(doc)

        num_tokens = token_count(text)

        num_sentences = metadata.get(
            "num_sentences",
            len(doc.get("sentences", []))
        )

        num_relations = len(relations)

        profile = {
            "story_id": doc.get("story_id"),
            "title": doc.get("title"),

            "tokens": num_tokens,
            "sentences": num_sentences,

            "relations": num_relations,

            "cause": relation_types.get("CAUSE", 0),

            "precondition": relation_types.get(
                "PRECONDITION", 0
            ),

            "causal_nodes": metadata.get(
                "num_causal_nodes", 0
            ),

            "mean_distance": metadata.get(
                "mean_sentence_distance", 0
            ),

            "max_distance": metadata.get(
                "max_sentence_distance", 0
            ),

            "same_sentence": metadata.get(
                "same_sentence_relations", 0
            ),

            "cross_sentence": metadata.get(
                "cross_sentence_relations", 0
            ),

            "branching": metadata.get(
                "has_branching", False
            ),

            "chain": metadata.get(
                "has_chain", False
            ),

            "convergence": (
                metadata.get(
                    "num_convergence_nodes", 0
                ) > 0
            ),
        }

        profiles.append(profile)

    return profiles


# ============================================================
# CANDIDATE FILTER
# ============================================================

def is_candidate(profile):

    return (
        profile["tokens"] <= MAX_TOKENS
        and
        MIN_RELATIONS <= profile["relations"] <= MAX_RELATIONS
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("MAVEN-ERE DATASET PROFILING + CANDIDATE ANALYSIS")
    print("=" * 90)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    if not os.path.exists(INPUT_FILE):

        print(f"\nFile not found: {INPUT_FILE}")
        return

    documents = load_documents(INPUT_FILE)

    print(f"\nLoaded documents: {len(documents)}")

    # --------------------------------------------------------
    # Profile
    # --------------------------------------------------------

    profiles = profile_documents(documents)

    # --------------------------------------------------------
    # Candidate pool
    # --------------------------------------------------------

    candidates = [
        p for p in profiles
        if is_candidate(p)
    ]

    print("\n" + "=" * 90)
    print("CANDIDATE POOL")
    print("=" * 90)

    print(
        f"\nCriteria:"
        f"\n  Tokens:             <= {MAX_TOKENS}"
        f"\n  Causal relations:   {MIN_RELATIONS}-{MAX_RELATIONS}"
    )

    print(
        f"\nCandidate documents: "
        f"{len(candidates)} / {len(profiles)}"
    )

    print(
        f"Candidate percentage: "
        f"{100 * len(candidates) / len(profiles):.2f}%"
    )

    # --------------------------------------------------------
    # Candidate length
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("CANDIDATE TEXT LENGTH")
    print("=" * 90)

    print_distribution(
        "Tokens",
        [p["tokens"] for p in candidates]
    )

    print_distribution(
        "Sentences",
        [p["sentences"] for p in candidates]
    )

    # --------------------------------------------------------
    # Candidate causal complexity
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("CANDIDATE CAUSAL COMPLEXITY")
    print("=" * 90)

    print_distribution(
        "Causal relations",
        [p["relations"] for p in candidates]
    )

    print_distribution(
        "Causal nodes",
        [p["causal_nodes"] for p in candidates]
    )

    # --------------------------------------------------------
    # Relation types
    # --------------------------------------------------------

    total_cause = sum(
        p["cause"] for p in candidates
    )

    total_precondition = sum(
        p["precondition"] for p in candidates
    )

    total_relations = (
        total_cause +
        total_precondition
    )

    print("\nRelation types")
    print("-" * 90)

    print(
        f"CAUSE:         {total_cause:,} "
        f"({100 * total_cause / total_relations:.2f}%)"
    )

    print(
        f"PRECONDITION:  {total_precondition:,} "
        f"({100 * total_precondition / total_relations:.2f}%)"
    )

    print(
        f"TOTAL:         {total_relations:,}"
    )

    # --------------------------------------------------------
    # Candidate distance
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("CANDIDATE CAUSAL DISTANCE")
    print("=" * 90)

    print_distribution(
        "Mean sentence distance",
        [p["mean_distance"] for p in candidates]
    )

    print_distribution(
        "Maximum sentence distance",
        [p["max_distance"] for p in candidates]
    )

    total_same = sum(
        p["same_sentence"] for p in candidates
    )

    total_cross = sum(
        p["cross_sentence"] for p in candidates
    )

    total_distance = total_same + total_cross

    print("\nRelation distance")
    print("-" * 90)

    print(
        f"Same sentence:   {total_same:,} "
        f"({100 * total_same / total_distance:.2f}%)"
    )

    print(
        f"Cross sentence:  {total_cross:,} "
        f"({100 * total_cross / total_distance:.2f}%)"
    )

    # --------------------------------------------------------
    # Graph structure
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("CANDIDATE GRAPH STRUCTURE")
    print("=" * 90)

    branching = sum(
        p["branching"] for p in candidates
    )

    chains = sum(
        p["chain"] for p in candidates
    )

    convergence = sum(
        p["convergence"] for p in candidates
    )

    n = len(candidates)

    print(
        f"Branching:     {branching:,} "
        f"({100 * branching / n:.2f}%)"
    )

    print(
        f"Chains:        {chains:,} "
        f"({100 * chains / n:.2f}%)"
    )

    print(
        f"Convergence:   {convergence:,} "
        f"({100 * convergence / n:.2f}%)"
    )

    # --------------------------------------------------------
    # Length buckets
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("CANDIDATE LENGTH DISTRIBUTION")
    print("=" * 90)

    length_buckets = [
        ("<=150", 0, 150),
        ("151-200", 151, 200),
        ("201-300", 201, 300),
        ("301-400", 301, 400),
        ("401-500", 401, 500),
    ]

    for name, lower, upper in length_buckets:

        subset = [
            p for p in candidates
            if lower <= p["tokens"] <= upper
        ]

        print(
            f"{name:<12}"
            f"{len(subset):>5} documents "
            f"({100 * len(subset) / n:>6.2f}%)"
        )

    # --------------------------------------------------------
    # Relation buckets
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("CANDIDATE CAUSAL COMPLEXITY DISTRIBUTION")
    print("=" * 90)

    relation_buckets = [
        ("3-5", 3, 5),
        ("6-10", 6, 10),
        ("11-20", 11, 20),
        ("21-30", 21, 30),
    ]

    for name, lower, upper in relation_buckets:

        subset = [
            p for p in candidates
            if lower <= p["relations"] <= upper
        ]

        print(
            f"{name:<12}"
            f"{len(subset):>5} documents "
            f"({100 * len(subset) / n:>6.2f}%)"
        )

    # --------------------------------------------------------
    # Causal distance buckets
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("CANDIDATE MAXIMUM CAUSAL DISTANCE")
    print("=" * 90)

    distance_buckets = [
        ("0-2", 0, 2),
        ("3-5", 3, 5),
        ("6-10", 6, 10),
        ("11+", 11, float("inf")),
    ]

    for name, lower, upper in distance_buckets:

        subset = [
            p for p in candidates
            if lower <= p["max_distance"] <= upper
        ]

        print(
            f"{name:<12}"
            f"{len(subset):>5} documents "
            f"({100 * len(subset) / n:>6.2f}%)"
        )

    # --------------------------------------------------------
    # Graph combinations
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("CANDIDATE GRAPH COMBINATIONS")
    print("=" * 90)

    combinations = Counter()

    for p in candidates:

        structure = []

        if p["branching"]:
            structure.append("branching")

        if p["chain"]:
            structure.append("chain")

        if p["convergence"]:
            structure.append("convergence")

        if not structure:
            key = "simple"
        else:
            key = " + ".join(structure)

        combinations[key] += 1

    for key, count in combinations.most_common():

        print(
            f"{key:<35}"
            f"{count:>5} "
            f"({100 * count / n:>6.2f}%)"
        )

    # --------------------------------------------------------
    # Candidate quality warnings
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("SUBSET DESIGN CHECK")
    print("=" * 90)

    if len(candidates) < 300:

        print(
            f"\nWARNING: Only {len(candidates)} candidates."
            "\nA 300-document subset is not possible."
        )

    else:

        print(
            f"\nOK: {len(candidates)} candidates available "
            f"for a 300-document subset."
        )

    # --------------------------------------------------------
    # Suggested target
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("PROVISIONAL 300-STORY TARGET")
    print("=" * 90)

    print("""
Target:

    <=150 tokens       ~60 stories
    151-300 tokens     ~150 stories
    301-500 tokens     ~90 stories

Within these groups we will balance:

    - causal relation complexity
    - causal distance
    - graph structure

We will NOT select the stories yet.
""")

    print("=" * 90)
    print("CANDIDATE ANALYSIS COMPLETE")
    print("=" * 90)


if __name__ == "__main__":
    main()