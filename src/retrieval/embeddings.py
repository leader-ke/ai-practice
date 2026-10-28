MODEL_NAME = "all-MiniLM-L6-v2"


def load_model(name=MODEL_NAME):
    # Imported here so the rest of the package works without the large embedding library.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def text_to_embed(chunk, add_prefix=False):
    """The text that gets embedded: the chunk, optionally prefixed with its document name."""
    if add_prefix:
        return f"Document: {chunk['document']}\n{chunk['text']}"
    return chunk["text"]


def embed(model, texts):
    """Normalized vectors, so a dot product between two of them is their cosine similarity."""
    return model.encode(texts, normalize_embeddings=True)
