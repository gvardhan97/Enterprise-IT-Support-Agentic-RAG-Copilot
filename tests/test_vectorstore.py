from types import SimpleNamespace

import pytest

from app.rag import vectorstore


class FakePinecone:
    def __init__(self, indexes):
        self.indexes = dict(indexes)  # name -> dimension
        self.deleted = []
        self.created = []

    def list_indexes(self):
        return [{"name": name} for name in self.indexes]

    def describe_index(self, name):
        return SimpleNamespace(dimension=self.indexes[name], status={"ready": True})

    def delete_index(self, name):
        del self.indexes[name]
        self.deleted.append(name)

    def create_index(self, name, dimension, metric, spec):
        self.indexes[name] = dimension
        self.created.append((name, dimension, metric))

    def Index(self, name):
        return f"index:{name}"


@pytest.fixture
def pinecone(monkeypatch):
    def install(indexes):
        fake = FakePinecone(indexes)
        monkeypatch.setattr(vectorstore, "Pinecone", lambda api_key: fake)
        monkeypatch.setattr(vectorstore.time, "sleep", lambda s: None)
        monkeypatch.setattr(vectorstore.settings, "pinecone_api_key", "test-key")
        monkeypatch.setattr(vectorstore.settings, "embedding_model", "text-embedding-3-small")
        return fake
    return install


@pytest.mark.parametrize("model, dimension", [
    ("text-embedding-3-small", 1536),
    ("TEXT-EMBEDDING-3-LARGE", 3072),
    ("text-embedding-ada-002", 1536),
    ("all-MiniLM-L6-v2", 384),
    ("azure/text-embedding-3-large-eu", 3072),
    ("sentence-transformers/all-minilm-l12-v2", 384),
])
def test_embedding_dimension(model, dimension):
    assert vectorstore.get_embedding_dimension(model) == dimension


def test_unknown_embedding_model_is_rejected():
    with pytest.raises(ValueError, match="Unsupported embedding model 'bge-large'"):
        vectorstore.get_embedding_dimension("bge-large")


def test_blank_embedding_model_is_rejected(monkeypatch):
    monkeypatch.setattr(vectorstore.settings, "embedding_model", "  ")

    with pytest.raises(RuntimeError, match="not configured"):
        vectorstore.get_embedding_dimension()


def test_creates_missing_index(pinecone):
    fake = pinecone({})

    assert vectorstore.ensure_index() == "index:fde-it-support-rag"
    assert fake.created == [("fde-it-support-rag", 1536, "cosine")]
    assert fake.deleted == []


def test_keeps_index_with_matching_dimension(pinecone):
    fake = pinecone({"fde-it-support-rag": 1536})

    vectorstore.ensure_index()

    assert fake.created == []
    assert fake.deleted == []


def test_recreates_index_when_embedding_dimension_changes(pinecone):
    fake = pinecone({"fde-it-support-rag": 384, "other-team-index": 768})

    vectorstore.ensure_index()

    assert fake.deleted == ["fde-it-support-rag"]
    assert fake.created == [("fde-it-support-rag", 1536, "cosine")]
    assert fake.indexes["other-team-index"] == 768


def test_missing_keys_fail_loudly(monkeypatch):
    monkeypatch.setattr(vectorstore, "_embeddings", None)

    with pytest.raises(RuntimeError, match="PINECONE_API_KEY"):
        vectorstore.ensure_index()
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        vectorstore.get_embeddings()
