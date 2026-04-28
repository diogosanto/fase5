from sklearn.feature_extraction.text import HashingVectorizer


class LocalHashEmbeddings:
    def __init__(self, n_features: int = 256) -> None:
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        matrix = self.vectorizer.transform([text])
        return matrix.toarray()[0].tolist()


def get_embedding_model() -> LocalHashEmbeddings:
    return LocalHashEmbeddings()
