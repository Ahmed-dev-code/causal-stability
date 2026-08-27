from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")


def similarity(text1, text2):

    embeddings = model.encode(
        [text1, text2],
        normalize_embeddings=True
    )

    return cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]


if __name__ == "__main__":

    pairs = [
    # Expected SAME
    ("Heavy rain", "Intense rainfall"),
    ("Flooding", "flooding in the town"),
    ("Blocked the main road", "main road inaccessible"),
    ("resulting flood", "Flooding"),

    # Expected DIFFERENT
    ("Heavy rain", "flooding"),
    ("Flooding", "main road"),
    ("Heavy rain", "car accident"),
    ("road blockage", "flooding"),
    ]

    for a, b in pairs:

        score = similarity(a, b)

        print(
            f"{a:30} <-> {b:30} = {score:.3f}"
        )