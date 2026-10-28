def chunk_documents(documents, max_chars):
    chunks = []
    for document in documents:
        current_paragraphs = []
        current_length = 0

        for paragraph in document["paragraphs"]:
            text = paragraph["text"]
            paragraph_id = paragraph["id"]

            added_length = len(text) + (2 if current_paragraphs else 0)

            if current_paragraphs and current_length + added_length > max_chars:
                chunks.append(
                    {
                        "document": document["name"],
                        "paragraph_ids": [p["id"] for p in current_paragraphs],
                        "text": "\n\n".join(p["text"] for p in current_paragraphs),
                    }
                )
                current_paragraphs = []
                current_length = 0

            if len(text) > max_chars:
                words = text.split()
                pieces = []
                piece = ""

                for word in words:
                    candidate = f"{piece} {word}".strip()

                    if len(candidate) > max_chars and piece:
                        pieces.append(piece)
                        piece = word
                    else:
                        piece = candidate

                if piece:
                    pieces.append(piece)

                for piece in pieces:
                    chunks.append(
                        {
                            "document": document["name"],
                            "paragraph_ids": [paragraph_id],
                            "text": piece,
                        }
                    )

                continue

            current_paragraphs.append(paragraph)
            current_length += added_length

        if current_paragraphs:
            chunks.append(
                {
                    "document": document["name"],
                    "paragraph_ids": [p["id"] for p in current_paragraphs],
                    "text": "\n\n".join(p["text"] for p in current_paragraphs),
                }
            )

    return chunks
