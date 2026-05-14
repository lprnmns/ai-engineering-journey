import pytest

from labs.rag.chunking import ChunkSearchResult
from labs.rag.context_builder import (
    build_context,
    build_context_and_prompt,
    build_rag_prompt,
    format_context_block,
)
from labs.rag.vector_store import build_vector_store


def make_result(
    chunk_id: str = "doc_001_chunk_001",
    text: str = "Python sanal ortam oluşturmak için venv kullanılır.",
    score: float = 0.9,
) -> ChunkSearchResult:
    return ChunkSearchResult(
        chunk_id=chunk_id,
        doc_id="doc_001",
        title="Python Virtual Environment",
        text=text,
        source="toolbox/python",
        chunk_index=1,
        score=score,
    )


def test_format_context_block_includes_metadata_and_text() -> None:
    result = make_result()

    block = format_context_block(index=1, result=result)

    assert "[1]" in block
    assert "Source: toolbox/python" in block
    assert "Document: doc_001" in block
    assert "Chunk: doc_001_chunk_001" in block
    assert "Score: 0.9000" in block
    assert "Python sanal ortam" in block


def test_build_context_joins_results_in_order() -> None:
    results = [
        make_result(chunk_id="chunk_001", text="Birinci parça.", score=0.8),
        make_result(chunk_id="chunk_002", text="İkinci parça.", score=0.7),
    ]

    context = build_context(results)

    assert "[1]" in context
    assert "[2]" in context
    assert context.index("Birinci parça.") < context.index("İkinci parça.")


def test_build_context_can_truncate_long_context() -> None:
    result = make_result(text="x" * 500)

    context = build_context([result], max_chars=100)

    assert len(context) < 120
    assert "[TRUNCATED]" in context


def test_build_context_rejects_invalid_max_chars() -> None:
    with pytest.raises(ValueError):
        build_context([make_result()], max_chars=0)


def test_build_rag_prompt_contains_question_context_and_rules() -> None:
    query = "Sanal ortam nasıl kurulur?"
    context = build_context([make_result()])

    prompt = build_rag_prompt(query=query, context=context)

    assert query in prompt
    assert context in prompt
    assert "using only the context" in prompt
    assert "Do not invent facts" in prompt


def test_build_context_and_prompt_returns_structured_result() -> None:
    query = "Sanal ortam nasıl kurulur?"
    results = [make_result()]

    output = build_context_and_prompt(query=query, results=results)

    assert output.query == query
    assert output.used_chunk_count == 1
    assert "Python sanal ortam" in output.context
    assert "Question:" in output.prompt


def test_context_builder_works_with_vector_store_results() -> None:
    store = build_vector_store()
    query = "RAG sisteminde ilgili doküman parçaları nasıl bulunur?"
    results = store.search(query=query, top_k=2)

    output = build_context_and_prompt(query=query, results=results)

    assert output.used_chunk_count == 2
    assert "doc_005" in output.context
    assert query in output.prompt
