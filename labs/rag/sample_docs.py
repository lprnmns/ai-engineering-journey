from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    source: str


SAMPLE_DOCUMENTS: list[Document] = [
    Document(
        doc_id="doc_001",
        title="Python Virtual Environment",
        text=(
            "Python projelerinde sanal ortam oluşturmak için python -m venv .venv "
            "komutu kullanılır. Sanal ortam aktif edilince bağımlılıklar projeye özel kurulur."
        ),
        source="toolbox/python",
    ),
    Document(
        doc_id="doc_002",
        title="Git Branch Workflow",
        text=(
            "Git branch, ana kodu bozmadan yeni özellik geliştirmek için kullanılır. "
            "Yeni branch açılır, değişiklikler commit edilir ve sonra pull request ile main branch'e alınır."
        ),
        source="toolbox/git",
    ),
    Document(
        doc_id="doc_003",
        title="Cosine Similarity",
        text=(
            "Cosine similarity iki vektörün yön benzerliğini ölçer. "
            "Değer 1'e yaklaştıkça iki metin veya vektör daha benzer kabul edilir."
        ),
        source="toolbox/math",
    ),
    Document(
        doc_id="doc_004",
        title="Pandas CSV Loading",
        text=(
            "Pandas ile CSV okumak için read_csv fonksiyonu kullanılır. "
            "Veri analizi yaparken satır sayısı, eksik değerler ve sütun tipleri incelenir."
        ),
        source="toolbox/data",
    ),
    Document(
        doc_id="doc_005",
        title="RAG Retrieval",
        text=(
            "Retrieval augmented generation sistemlerinde önce kullanıcının sorusuyla ilgili doküman parçaları bulunur. "
            "Sonra bu parçalar dil modeline bağlam olarak verilir."
        ),
        source="toolbox/rag",
    ),
]
