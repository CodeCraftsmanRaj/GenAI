from src.documents import chunk_documents

def test_chunking_preserves_source_and_text():
    docs = [{"source": "a.txt", "source_path": "a.txt", "text": "one two three four five six seven eight nine ten"}]
    chunks = chunk_documents(docs, chunk_size_words=5, overlap_words=2)
    assert chunks
    assert chunks[0]["source"] == "a.txt"
    assert chunks[0]["text"] == "one two three four five"
    assert chunks[1]["text"].startswith("four five")
