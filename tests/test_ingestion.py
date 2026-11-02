import pytest
from docx import Document as DocxDocument
from langchain_core.documents import Document
from pypdf import PdfWriter

from app.services.ingestion import chunk_documents, load_file


@pytest.mark.parametrize("name", ["vpn.txt", "VPN.MD"])
def test_loads_text_and_markdown(tmp_path, name):
    path = tmp_path / name
    path.write_text("Connect with GlobalProtect.", encoding="utf-8")

    docs = load_file(path)

    assert len(docs) == 1
    assert docs[0].page_content == "Connect with GlobalProtect."
    assert docs[0].metadata["source"] == str(path)


def test_loads_docx_and_drops_blank_paragraphs(tmp_path):
    path = tmp_path / "policy.docx"
    doc = DocxDocument()
    for text in ["Password rules", "   ", "At least 14 characters", ""]:
        doc.add_paragraph(text)
    doc.save(path)

    docs = load_file(path)

    assert docs[0].page_content == "Password rules\nAt least 14 characters"
    assert docs[0].metadata == {"source": str(path)}


def test_loads_pdf_one_document_per_page(tmp_path):
    path = tmp_path / "runbook.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    writer.write(path)

    assert len(load_file(path)) == 2


def test_rejects_unsupported_type(tmp_path):
    path = tmp_path / "notes.csv"
    path.write_text("a,b")

    with pytest.raises(ValueError, match="Unsupported file type: .csv"):
        load_file(path)


def test_chunks_long_documents_with_overlap_and_offsets():
    words = [f"word{i}" for i in range(600)]
    doc = Document(page_content=" ".join(words), metadata={"source": "kb.md"})

    chunks = chunk_documents([doc])

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 900 for c in chunks)
    assert all(c.metadata["source"] == "kb.md" for c in chunks)
    # Each chunk records where it starts, and neighbours overlap.
    for prev, cur in zip(chunks, chunks[1:]):
        assert cur.metadata["start_index"] > prev.metadata["start_index"]
        assert prev.page_content.split()[-1] in cur.page_content


def test_short_document_stays_one_chunk():
    chunks = chunk_documents([Document(page_content="Short note.")])

    assert [c.page_content for c in chunks] == ["Short note."]
    assert chunks[0].metadata["start_index"] == 0
