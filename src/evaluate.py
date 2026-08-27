from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL = SentenceTransformer("all-MiniLM-L6-v2")


def event_similarity(event1, event2):
    """
    Calculate semantic similarity between two event descriptions.
    """

    embeddings = MODEL.encode(
        [event1, event2],
        normalize_embeddings=True
    )

    return float(cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0])


def align_events(original_events, revised_events, threshold=0.70):
    """
    One-to-one greedy event alignment.

    Each revised event can be matched to at most one
    original event.
    """

    candidates = []

    # Calculate every possible pair
    for original in original_events:
        for revised in revised_events:

            score = event_similarity(original, revised)

            if score >= threshold:
                candidates.append(
                    (score, original, revised)
                )

    # Highest similarities first
    candidates.sort(reverse=True)

    alignments = []

    matched_original = set()
    matched_revised = set()

    for score, original, revised in candidates:

        if original in matched_original:
            continue

        if revised in matched_revised:
            continue

        alignments.append({
            "original": original,
            "revision": revised,
            "similarity": round(score, 3)
        })

        matched_original.add(original)
        matched_revised.add(revised)

    return alignments

def extract_events(graph):
    """
    Extract unique event descriptions from a causal graph.
    """

    events = []

    for relation in graph["relations"]:

        cause = relation["cause"]
        effect = relation["effect"]

        if cause not in events:
            events.append(cause)

        if effect not in events:
            events.append(effect)

    return events


def calculate_event_preservation(original_graph, revised_graph):

    original_events = extract_events(original_graph)
    revised_events = extract_events(revised_graph)

    alignments = align_events(
        original_events,
        revised_events
    )

    preservation = len(alignments) / len(original_events)

    return preservation, alignments
def calculate_edge_preservation(original_graph, revised_graph, alignments):

    # Map original event → revised event
    mapping = {
        item["original"]: item["revision"]
        for item in alignments
    }

    original_edges = original_graph["relations"]
    revised_edges = revised_graph["relations"]

    preserved = 0

    for edge in original_edges:

        original_cause = edge["cause"]
        original_effect = edge["effect"]

        if original_cause not in mapping:
            continue

        if original_effect not in mapping:
            continue

        predicted_cause = mapping[original_cause]
        predicted_effect = mapping[original_effect]

        for revised_edge in revised_edges:

            if (
                revised_edge["cause"] == predicted_cause
                and
                revised_edge["effect"] == predicted_effect
            ):
                preserved += 1
                break

    if len(original_edges) == 0:
        return 0.0

    return preserved / len(original_edges)

def evaluate_graph_pair(original_graph, revised_graph):

    event_preservation, alignments = calculate_event_preservation(
        original_graph,
        revised_graph
    )

    edge_preservation = calculate_edge_preservation(
        original_graph,
        revised_graph,
        alignments
    )

    stability = (
        event_preservation +
        edge_preservation
    ) / 2

    return {
        "event_preservation": round(event_preservation, 3),
        "edge_preservation": round(edge_preservation, 3),
        "causal_stability": round(stability, 3),
        "alignments": alignments
    }

if __name__ == "__main__":

    original = {
        "relations": [
            {
                "cause": "Heavy rain",
                "effect": "Flooding in the town"
            },
            {
                "cause": "Flooding",
                "effect": "Blocked the main road"
            }
        ]
    }

    revision = {
        "relations": [
            {
                "cause": "Intense rainfall",
                "effect": "flooding in the town"
            },
            {
                "cause": "resulting flood",
                "effect": "main road inaccessible"
            }
        ]
    }

    result = evaluate_graph_pair(
        original,
        revision
    )

    print("\nRESULT")
    print("=" * 50)

    print(f"Event preservation: {result['event_preservation']}")
    print(f"Edge preservation:  {result['edge_preservation']}")
    print(f"Causal stability:   {result['causal_stability']}")

    print("\nALIGNMENTS")

    for alignment in result["alignments"]:
        print(
            f"{alignment['original']} "
            f"<-> "
            f"{alignment['revision']} "
            f"= "
            f"{alignment['similarity']}"
        )