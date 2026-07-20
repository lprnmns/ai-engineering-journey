# 9 Aylık AI Engineering Yol Haritası — Revize 2026

**Güncelleme tarihi:** 20 Temmuz 2026

**Hedef profil:** Production-oriented AI Engineer / Applied AI Engineer / RAG Engineer

**Planın başlangıç kanıtı:** 16 Mayıs 2026 tarihli “Becoming an AI Engineer — Month 1/9” paylaşımı

**Önerilen tempo:** Üniversite döneminde haftada 10–15 saat, tatilde 15–20 saat

## 1. Kısa karar

Bu plan sıfırdan Python veya bilgisayar bilimi öğretmek için değildir. Bilgisayar mühendisliği eğitimi ve mevcut repo, temel altyapının önemli bölümünün zaten bulunduğunu gösteriyor.

Yol haritası başlamadan önce şu alanlarda çalışma yapılmıştı:

- Python, OOP, type hints, mypy ve pytest
- Git, branch, PR ve Conventional Commits
- Pandas, SQL, EDA ve veri temizleme
- Baseline ML, cross-validation ve experiment logging
- Titanic üzerinde feature engineering, ablation ve model comparison

Bu çalışmalar 9 aylık programın “Ay 1–2”si olarak sayılmayacak. Kamuya açık başlangıç paylaşımında mini RAG sistemi açıkça **Month 1/9** olarak tanımlandığı için ay numaralandırmasında bu paylaşım esas alınacaktır.

## 2. Üniversiteden gelen altyapı

Transkript, aşağıdaki alanlarda sıfırdan tekrar gerektirmeyecek bir temel olduğunu gösteriyor:

- Programlama teknikleri, yapısal programlama ve nesne yönelimli programlama
- Veri yapıları ve algoritmalar
- Ayrık matematik, lineer cebir, olasılık ve nümerik analiz
- İşletim sistemleri ve bilgisayar mimarisi
- Veritabanı sistemleri, veritabanı uygulamaları ve dosya yapıları
- Web teknolojileri
- Java ve programlama dilleri
- Biçimsel diller ve otomata
- Proje dersleri ve staj deneyimi

Bu nedenle ayrı aylar boyunca temel OOP, temel veri yapıları, temel SQL veya temel Git çalışılmayacak. Yalnızca projede ihtiyaç duyulan konular hedefli şekilde tekrar edilecek.

Güçlendirilmesi gereken bilgisayar mühendisliği alanları:

- HTTP, REST, ağ temelleri ve API tasarımı
- Async I/O, concurrency ve background jobs
- İleri SQL, transaction, index ve query planı
- Retry, timeout, idempotency, queue ve distributed workflow kavramları
- Software architecture ve system design anlatımı
- Authentication, authorization, secret yönetimi ve rate limiting
- Integration, contract ve end-to-end testleri
- Teknik İngilizce dokümantasyon ve proje sunumu

## 3. Repoda kanıtlanan mevcut yetkinlikler

- Strict `mypy` kullanılan Python kod tabanı
- Geniş `pytest` test seti
- Git ve PR tabanlı çalışma düzeni
- Pandas, SQL, EDA ve preprocessing
- Baseline model ve doğru validation yaklaşımı
- Feature engineering, ablation ve model comparison
- Repeated CV ve visual error analysis
- Deney sonuçlarını ve başarısız denemeleri kayıt altında tutma
- Chunking ve metadata
- In-memory vector store
- Term-frequency ve TF-IDF
- Hybrid retrieval
- Retrieval benchmarkları
- Weak evidence filtering
- No-answer detection
- Answerability evaluation, threshold ve profiles
- Sentence Transformers tabanlı dense embedding
- Dense in-memory vector store
- Türkçe paraphrase retrieval benchmarkı
- TF-IDF ve dense retrieval karşılaştırması
- Dense + lexical hybrid retrieval ve ağırlık deneyi
- Cross-encoder reranker bileşeni

16 Mayıs paylaşımında 239 test bulunurken repo daha sonra 293 teste; dense embedding, hybrid retrieval ve cross-encoder reranker katmanlarına ilerledi. Bu, Month 2 hedeflerinden hem “dense embeddings” hem de “stronger evals” bölümlerinde gerçek ilerleme olduğunu gösteriyor.

## 4. Dışarıdan gelen teknik geri bildirimin anlamı

Paylaşıma verilen cevapta iki önemli nokta doğru şekilde fark edilmiş:

1. Framework kullanmadan “sıkıcı” temel bileşenleri kurmak, RAG sisteminin nerede bozulduğunu görmeyi sağlar.
2. No-answer yolu, similarity skorunu yalnızca teknik bir değer olmaktan çıkarıp ürün kararına dönüştürür.

Bu geri bildirim önemlidir çünkü çalışmanın değeri yalnızca TF-IDF veya vector store yazmak değildir. Asıl güçlü taraf, sistemin **ne zaman cevap vermemesi gerektiğini** tasarlamaya başlamaktır.

Portföy anlatımında kullanılabilecek ana fikir:

> I built a mini RAG pipeline from first principles and treated retrieval confidence as a product decision: answer with evidence or reject when the available context is insufficient.

## 5. 2026 güncel ihtiyaçlardan çıkan sonuç

Stanford AI Index 2026, 2025 AI iş ilanlarında Python, bilgisayar bilimi, scalability, automation, workflow management, data analysis, SQL, project management, data science ve AWS'nin öne çıktığını gösteriyor.

Aynı rapora göre 2024–2025 arasında:

- Generative AI geçen ilanlar %111 arttı.
- Large language modeling %102 arttı.
- Prompt engineering %261 arttı.
- Retrieval augmented generation %337 arttı.
- Context engineering güçlü şekilde yükseldi.
- Agentic AI, AI agents ve LangGraph gibi terimlerde hızlı büyüme görüldü.

Modern AI Engineer profili şu birleşime ihtiyaç duyuyor:

1. Python, SQL ve yazılım mühendisliği
2. Veri, ML ve doğru evaluation sezgisi
3. LLM, RAG, context engineering ve tool use
4. API, Docker, cloud, automation ve scalability
5. Güvenlik, tracing, maliyet ve reliability

Anthropic'in agent rehberi karmaşık frameworklerden önce basit ve birleştirilebilir workflow kalıplarını öneriyor. Bu yüzden plan, agent framework ezberinden önce tool contract, state, retry, approval ve evaluation öğretir.

OWASP GenAI riskleri; prompt injection, sensitive information disclosure, supply chain, data/model poisoning, improper output handling, excessive agency, system prompt leakage, vector/embedding weakness, misinformation ve unbounded consumption konularını kapsıyor. Bunlar flagship projede test edilmeden proje production-ready sayılmayacak.

NIST Generative AI Profile nedeniyle risk yönetimi ayrı bir son rapor değil; veri, ölçüm, deployment ve ürün kararlarının sürekli parçası olacak.

MCP güncel bir araç bağlantı standardıdır; ancak temel tool-calling mantığı anlaşılmadan kullanılmayacaktır.

## 6. Düzeltilmiş mevcut konum ve yüzde

| Aşama | Durum | Kanıt / eksik |
|---|---:|---|
| Yol haritası öncesi CS/Python/ML hazırlığı | Tamamlandı | Üniversite dersleri, Python engineering ve Titanic çalışmaları |
| Month 1 — Mini RAG from scratch | Tamamlandı | Chunking, vector store, TF-IDF, hybrid retrieval, no-answer ve answerability evals |
| Month 2 — Embeddings, stronger evals, real LLM, deployment | Yaklaşık %35–40 | Dense embedding, lexical/dense karşılaştırması, dense hybrid ve reranker bileşeni var; büyük eval seti, gerçek LLM ve deployment yok |
| Month 3–9 | Başlanmadı | Production RAG, transformer, agents, MLOps/cloud, gerçek ürün ve kariyer paketi |

**9 aylık kamuya açık programdaki katı ilerleme:** yaklaşık **%15–16**.

Hesap:

```text
Month 1 tamamlandı       = 1.00 ay
Month 2 yaklaşık %35–40  = 0.35–0.40 ay
Toplam                   = 1.35–1.40 / 9
                         ≈ %15–16
```

Bu oran bilgi seviyesinin yalnızca %15 olduğu anlamına gelmez. Bilgisayar mühendisliği ve yol haritası öncesi çalışmalar dahil edildiğinde junior yazılım/AI hazırlığı yaklaşık %40 seviyesindedir. İki sayı farklı şeyleri ölçer:

- `%15–16`: İlan ettiğin dokuz aylık programın tamamlanma oranı
- `yaklaşık %40`: Mevcut akademik ve teknik altyapının junior AI Engineer hazırlığına katkısı

Şu an doğru başlangıç noktası **Month 2'ye mentorun verdiği ilk hafta staj sprintiyle devam etmektir**.

## 7. Çalışma sistemi

### Haftalık zaman dağılımı

- %55 proje geliştirme
- %20 hedefli öğrenme ve resmi dokümantasyon
- %15 test, evaluation ve hata analizi
- %10 README, teknik İngilizce ve portföy anlatımı

### Sınav haftası modu

- Haftada 4–6 saat
- Yeni büyük konu açılmaz.
- Küçük refactor, test, eval case veya dokümantasyon yapılır.
- Takvim değil kabul kriteri esas alınır.

### Her hafta zorunlu kanıt

En az biri:

- Çalışan özellik
- Negatif veya regression testi
- Benchmark/eval sonucu
- Architecture Decision Record
- Hata analizi
- Deploy edilmiş sürüm
- İngilizce teknik dokümantasyon

### AI ile kod yazma kuralı

- Açıklayamadığın kod merge edilmez.
- AI tarafından yazılan kritik fonksiyon için negatif test yazılır.
- Dependency ve mimari karar resmi dokümanla doğrulanır.
- Güvenlik, veri silme ve dış sistem işlemlerinde human approval korunur.
- Özellik bittikten sonra akış kod kapalıyken anlatılabilmelidir.

## 8. Revize 9 aylık plan

### Month 1 — Mini RAG Sistemini Temelden Kurma

**Durum:** Tamamlandı.

Tamamlananlar:

- Document chunking
- Metadata taşıma
- In-memory vector store
- TF ve TF-IDF vectorization
- Cosine similarity
- Hybrid retrieval
- Context building
- Extractive answerer
- Weak evidence filtering
- No-answer detection
- Answerability evals ve threshold denemeleri
- Hard benchmarklar ve answerability profiles

Öğrenilen ana ders:

> RAG yalnızca top-k chunk bulup cevap üretmek değildir. Sistem, bulunan kanıtın yeterli olup olmadığına ve cevap vermenin güvenli olup olmadığına karar vermelidir.

**Kabul kriteri:** Month 1 sonunda mevcut testler, temiz mypy sonucu ve answerable/unanswerable örnekleri ayıran benchmark nedeniyle tamamlandı. Repo sonraki Month 2 çalışmalarıyla 293 teste ulaştı.

### Month 2 — Embeddings, Stronger Evals, Real LLM ve İlk Deployment

**Durum:** Devam ediyor; yaklaşık %35–40.

20 Temmuz 2026 itibarıyla tamamlanan veya ilerleyen parçalar:

- Sentence Transformers tabanlı dense vectorizer
- Dense in-memory vector store
- Türkçe paraphrase benchmarkı
- TF-IDF ve dense retrieval karşılaştırması
- Dense + lexical hybrid retrieval
- Hybrid ağırlık deneyi
- Dense answerability threshold deneyi
- Bağımsız cross-encoder reranker bileşeni

Ana eksikler:

- En az 50 versioned eval case, BM25, nDCG ve bootstrap/confidence interval
- Dense/hybrid retrieval → reranker birleşik pipeline ve karşılaştırmalı benchmark
- Gerçek PDF ingestion ve kalıcı vector database
- Gerçek LLM, structured/source-mapped cevap ve provider abstraction
- FastAPI, Docker, CI, cloud deployment, latency ve maliyet ölçümü

#### Hafta 1 — Dense embeddings

- Sentence Transformers veya eşdeğer açık embedding modeli
- Embedding batch üretimi
- FAISS ile lokal dense index
- Lexical ve dense retrieval karşılaştırması
- Türkçe/İngilizce query davranışı

#### Hafta 2 — Stronger retrieval evaluation

- En az 50 versioned golden query
- Answerable, unanswerable, near-miss ve adversarial kategorileri
- Recall@k, MRR ve nDCG
- BM25 baseline
- Dense, lexical ve hybrid karşılaştırması
- Cross-encoder reranker deneyi
- Confidence interval veya bootstrap

#### Hafta 3 — Gerçek LLM entegrasyonu

- Bir ana LLM provider
- Provider-independent interface
- Structured output ve schema validation
- Citation/source mapping
- Context engineering
- No-answer kararının LLM cevabından önce uygulanması
- Timeout, retry ve hata yönetimi

#### Hafta 4 — İlk servis ve deployment

- FastAPI iskeleti
- `/health`, `/ready` ve `/query` endpoint'leri
- Dockerfile
- GitHub Actions: Ruff, mypy ve pytest
- Basit cloud deployment
- Token, cost ve p50/p95 latency ölçümü

**Ay çıktısı:** Dense retrieval ve gerçek LLM kullanan, kaynak gösteren ve gerektiğinde cevap vermeyen deploy edilmiş mini RAG servisi.

**Kabul kriteri:** En az 50 eval case, kaynaklı cevap, no-answer precision/recall, yeşil CI, Docker image ve çalışan demo endpoint'i.

#### Staj sprinti — Mentor 1. hafta programı

Mentor programı Month 2'ye doğrudan entegre edilmiştir. Ayrıntılı eşleme ve teslim checklist'i için [Staj 1. Hafta Programı — Mevcut Durum ve Yol Haritası Entegrasyonu](staj_1_hafta_program_karsilastirmasi.md) belgesine bakılmalıdır.

Sıra:

1. Transformer, attention, token, context window, prompt katmanları ve açık model aileleri
2. Mevcut dense embedding kodunu kullanan en az 10 cümlelik açıklamalı deney
3. Gerçek PDF, iki chunk ayarı, kalıcı vector DB, dense/hybrid → rerank ve gerçek LLM içeren RAG
4. Ollama veya LM Studio üzerinde iki yerel model benchmarkı
5. Gerçek kurumsal problem ve ilk ürün fikri
6. Teknik raporların, ölçümlerin, mimari diyagramın ve demonun 15 dakikalık sunumda birleştirilmesi

Bu sprint Month 4'teki transformer/open-model konularının girişini ve Month 7–8'deki gerçek problem seçimini erkene çeker. İlerideki ayların daha derin PyTorch, serving, ürün geliştirme ve hardening hedefleri korunur.

### Month 3 — Production RAG, PostgreSQL ve Ingestion

#### Hafta 1 — Veri ingestion

- Markdown, PDF ve gerçek bir domain kaynağı
- Parse, normalize ve deduplication
- Document/chunk versioning
- Incremental re-index
- Bozuk dosya ve encoding hata yönetimi

#### Hafta 2 — PostgreSQL ve pgvector

- PostgreSQL schema
- pgvector index
- Alembic migration
- Metadata filter
- Index, transaction ve query planı
- Connection pooling

#### Hafta 3 — Async ve reliability

- Async endpoint
- Background ingestion job
- Retry, timeout ve idempotency
- Cache stratejisi
- Rate limit
- Structured logging

#### Hafta 4 — Production RAG evaluation

- Retrieval ve answer regression suite
- Citation accuracy
- Groundedness/faithfulness rubric
- Human review örneklemi
- Prompt/retriever/model version karşılaştırması
- Kullanıcı feedback kaydı

**Ay çıktısı:** Versioned ingestion, PostgreSQL/pgvector ve evaluation pipeline içeren production-minded RAG backend.

**Kabul kriteri:** Incremental ingestion, kalıcı vector store, integration test, en az 75 eval case ve reproducible benchmark.

### Month 4 — PyTorch, Transformer ve Open Model Temeli

#### Hafta 1 — PyTorch

- Tensor ve autograd
- Dataset/DataLoader
- Training ve validation loop
- Loss, optimizer ve scheduler
- Checkpoint, seed ve reproducibility

#### Hafta 2 — Transformer

- Tokenization
- Embedding katmanı
- Self-attention
- Encoder/decoder ayrımı
- Context window ve positional information
- Hugging Face inference

#### Hafta 3 — Uygulamalı NLP

- Text classification
- Embedding fine-tuning mantığı
- Baseline ile karşılaştırma
- Error analysis
- Calibration ve threshold

#### Hafta 4 — Open-weight model serving

- Küçük bir açık modelle lokal inference
- Quantization kavramı
- GPU/CPU latency karşılaştırması
- Memory ve throughput ölçümü
- Fine-tuning gerekip gerekmediğine dair karar raporu

**Ay çıktısı:** PyTorch eğitim projesi, transformer açıklama notu ve ölçülmüş küçük model inference deneyi.

**Kabul kriteri:** Training loop'u açıklayabilmek, modeli yeniden üretmek ve model seçimini kalite/latency/maliyet üzerinden savunmak.

### Month 5 — Tool Calling, Agent Workflows, MCP ve Güvenlik

#### Hafta 1 — Tool calling

- JSON schema tabanlı tool contract
- Input/output validation
- Read-only ve write tool ayrımı
- Timeout ve retry
- Kritik işlemlerde human approval

#### Hafta 2 — Stateful workflow

- Önce frameworksüz state machine
- Checkpoint ve resumability
- Deterministic routing
- Idempotency
- Sonra yalnızca bir orchestration aracı, tercihen LangGraph

#### Hafta 3 — MCP

- MCP client/server mimarisi
- Lokal MCP server
- Repo arama, doküman okuma veya güvenli DB sorgu aracı
- Tool allowlist ve permission
- MCP'nin gereksiz olduğu durumların analizi

#### Hafta 4 — OWASP odaklı red-team

- Prompt injection
- Sensitive information disclosure
- Improper output handling
- Excessive agency
- System prompt leakage
- Vector/embedding weakness
- Unbounded cost ve loop

**Ay çıktısı:** Onay mekanizması bulunan, izlenebilir ve test edilen tool-using workflow.

**Kabul kriteri:** En az üç tool, state/checkpoint, human approval, adversarial test seti ve failure analysis.

### Month 6 — MLOps, LLMOps, Cloud ve Observability

#### Hafta 1 — Versioning ve tracking

- MLflow veya eşdeğer tek araç
- Dataset, model, prompt ve eval version bağlantısı
- Artifact storage
- Reproducible run ID
- Release notes

#### Hafta 2 — Observability

- OpenTelemetry trace yaklaşımı
- Request → retrieval → rerank → LLM → tool zinciri
- Latency, error, token ve cost metrikleri
- Quality dashboard
- PII log masking

#### Hafta 3 — Cloud ve CI/CD

- AWS önerilir; GCP veya Azure da kabul edilir
- Container registry
- Managed PostgreSQL
- Secret manager
- Staging/production ayrımı
- CI/CD ve rollback

#### Hafta 4 — Reliability ve scale

- Load test
- Queue/background worker
- Cache
- Circuit breaker ve fallback
- Backup/recovery
- SLO ve production readiness checklist

**Ay çıktısı:** Cloud üzerinde gözlemlenebilir, sürümlenebilir ve geri alınabilir AI servisi.

**Kabul kriteri:** CI/CD, trace, dashboard, cost/latency ölçümü, rollback ve production readiness raporu.

### Month 7 — İkinci Gerçek Problem, Data Engineering ve Multimodal Extension

#### Ana amaç

Titanic ve kişisel notlar dışındaki gerçek bir domain probleminde bağımsız uygulama geliştirmek.

Proje staj yerindeki erişilebilir ve gizlilik açısından uygun bir problemden seçilirse daha değerlidir.

Zorunlu parçalar:

- Gerçek kullanıcı veya paydaş
- Data contract ve schema evolution
- Batch veya queue tabanlı ingestion
- Baseline ve başarı metriği
- Deployment
- Feedback loop
- Privacy değerlendirmesi

Ana proje zamanında ilerlerse yalnızca bir modern extension seçilir:

- PDF tablo/görsel/metin için multimodal document understanding
- Speech-to-text tabanlı bilgi erişimi
- Open-weight model serving
- Cost-aware multi-model routing

**Ay çıktısı:** İkinci, gerçek domain odaklı AI case study.

**Kabul kriteri:** En az birkaç gerçek kullanıcı geri bildirimi, ölçülebilir başarı metriği ve çalışan deployment.

### Month 8 — Flagship AI Ürünü ve Hardening

Önerilen proje: **AI Engineering Knowledge OS** veya staj alanındaki daha güçlü gerçek problem.

Teknik kapsam:

- FastAPI
- PostgreSQL/pgvector
- Hybrid retrieval ve reranker
- Kaynaklı RAG cevapları
- Tool-using workflow
- Versioned eval set
- Feedback loop
- Docker ve cloud deployment
- Tracing, cost ve latency dashboard
- OWASP odaklı güvenlik testleri

Haftalık sıra:

1. Scope, kullanıcı, başarı metriği ve threat model
2. End-to-end integration ve adversarial eval
3. Performance, reliability ve UX hardening
4. README, architecture diagram, demo ve case study

**Ay çıktısı:** Kullanılabilir ve teknik olarak savunulabilir flagship AI ürünü.

**Kabul kriteri:** Proje kalite kapısından en az 75/100, çalışan demo ve gerçek kullanıcı geri bildirimi.

### Month 9 — Açık Kaynak, Sistem Tasarımı ve İşe Hazırlık

#### Teknik hazırlık

- Python ve SQL mülakatları
- ML evaluation ve leakage
- RAG system design
- Agent/tool safety
- API, Docker ve cloud
- Cost/latency/reliability trade-off

#### Portföy

- İngilizce GitHub README
- İki dakikalık ve on dakikalık proje anlatımı
- Demo videoları
- Architecture Decision Records
- Başarısız deney ve öğrenilen dersler
- Ölçülebilir CV maddeleri

#### Görünürlük

- En az bir açık kaynak katkısı
- Teknik yazı veya case study
- Düzenli fakat abartısız proje paylaşımı
- Hedefli staj/junior başvuruları
- Başvuru ve geri bildirim takibi

**Ay çıktısı:** Savunulabilir portföy, CV, açık kaynak kanıtı ve işe başvuru paketi.

**Kabul kriteri:** İki güçlü proje, çalışan demo, teknik case study ve mock interviewlarda açıklanabilen mimari kararlar.

## 9. Dokuz ay boyunca devam edecek alışkanlıklar

### Matematik

Haftada 1–2 saat hedefli tekrar:

- Olasılık ve istatistik
- Confidence interval ve hypothesis testing
- Vektör uzayı ve cosine similarity
- Gradient descent ve optimizasyon
- Calibration ve uncertainty

Tam ders tekrarı yapılmaz; projede kullanılan matematik derinleştirilir.

### SQL ve data engineering

- Her ay en az beş orta seviye SQL sorusu
- Index ve query planı
- Transaction ve isolation
- JSON/vector kolonları
- ETL, data contract ve schema evolution

### System design

Month 3'ten itibaren her ay bir tasarım notu:

- RAG ingestion sistemi
- Model serving sistemi
- Agent/tool execution sistemi
- Evaluation platformu
- Cost-aware model routing

### Teknik İngilizce

- README ve ADR'ler İngilizce
- Ayda bir kısa teknik yazı
- Demo anlatımı İngilizce prova
- Resmi dokümanları İngilizce okuma

### Kariyer

- Ayda bir mock interview
- Ayda bir teknik paylaşım
- Month 6'dan itibaren seçici başvurular
- Month 9'u beklemeden CV/GitHub geri bildirimi

## 10. Ana teknoloji stack'i

- Python 3.12+
- `uv`, Ruff, mypy, pytest
- FastAPI ve Pydantic
- PostgreSQL, pgvector ve Alembic
- Redis yalnız gerçek cache/queue ihtiyacı varsa
- scikit-learn, PyTorch ve Hugging Face
- Sentence Transformers veya eşdeğer embedding modeli
- Bir ana LLM provider ve provider abstraction
- Docker, Docker Compose ve GitHub Actions
- MLflow veya eşdeğer tek experiment tracker
- OpenTelemetry tabanlı trace/metric yaklaşımı
- AWS, GCP veya Azure'dan yalnız biri

Bilinçli olarak ertelenecekler:

- Kubernetes: düşük trafik ve tek servis için gereksiz
- Multi-agent: tek workflow yetersiz kalmadıkça kullanılmaz
- Fine-tuning: ihtiyaç eval ile kanıtlanmadan yapılmaz
- Birden fazla agent framework
- Birden fazla production vector database
- AI backend'i gölgeleyen büyük frontend çalışması

## 11. Proje kalite kapısı

| Alan | Puan |
|---|---:|
| Problem ve veri tanımı | 10 |
| Baseline ve evaluation | 20 |
| Kod kalitesi ve test | 15 |
| API, container ve deployment | 15 |
| Reliability ve observability | 15 |
| Güvenlik ve privacy | 10 |
| Dokümantasyon ve demo | 10 |
| Kullanıcı/iş değeri | 5 |

Flagship proje en az 75/100 almalı.

Ek zorunluluklar:

- Başarısız örnekler saklanmalı.
- Metric seçimi açıklanmalı.
- Maliyet ve latency ölçülmeli.
- Negatif ve adversarial test bulunmalı.
- “Neden bu teknoloji?” sorusu cevaplanmalı.

## 12. Sıradaki çalışma sırası — mentor programıyla bütünleştirilmiş

### Aşama A — Staj 1. hafta teslim paketi

1. Transformer/token/attention/context window teknik notunu, prompt deneyini ve açık model karşılaştırmasını tamamla.
2. Mevcut Sentence Transformers katmanıyla en az 10 cümlelik embedding deneyini ve yorum tablosunu üret.
3. Gerçek PDF üzerinde iki chunk ayarını karşılaştır; dense/hybrid retrieval'ı reranker'a bağla.
4. ChromaDB, Qdrant, Milvus ve Pinecone karar notundan sonra sprint için tek bir kalıcı vector DB seç.
5. Gerçek LLM ile kaynaklı cevap ve no-answer akışını birleştir.
6. Ollama veya LM Studio üzerinde iki yerel modeli ortak test setiyle ölç.
7. Kurumsal problemi, kullanıcıyı, veriyi, başarı metriğini ve riskleri tanımla.
8. Tüm çıktıları 15 dakikalık teknik sunum ve kısa demoda birleştir.

### Aşama B — Month 2'yi production-minded şekilde kapatma

1. En az 50 versioned golden query oluştur.
2. BM25 baseline, Recall@k ve nDCG ekle.
3. Dense, lexical, hybrid ve reranked sonuçları tek benchmarkta karşılaştır.
4. Bootstrap veya confidence interval ile küçük eval seti belirsizliğini göster.
5. Provider-independent LLM interface, structured output, citation mapping, timeout ve retry ekle.
6. FastAPI `/health`, `/ready` ve `/query` endpoint'lerini oluştur.
7. Ruff + mypy + pytest CI, Docker image ve basit cloud deployment ekle.
8. Token, maliyet ve p50/p95 latency ölçümünü raporla.

Month 2 sonunda beklenen çıktı:

> Gerçek PDF kullanan; dense/hybrid retrieval ve reranking yapan; kaynak gösteren, gerektiğinde cevap vermeyen; test, CI, Docker ve API ile tekrar üretilebilir mini RAG servisi.

## 13. Kaynaklar

- Stanford HAI, **The 2026 AI Index Report**: https://hai.stanford.edu/ai-index/2026-ai-index-report
- Stanford HAI, 2026 rapor PDF'i: https://hai.stanford.edu/assets/files/ai_index_report_2026.pdf
- Anthropic, **Building Effective Agents**: https://www.anthropic.com/engineering/building-effective-agents
- OWASP GenAI Security Project: https://genai.owasp.org/llm-top-10/
- NIST, **AI RMF Generative AI Profile**: https://www.nist.gov/publications/artificial-intelligence-risk-management-framework-generative-artificial-intelligence
- Model Context Protocol: https://modelcontextprotocol.io/docs/getting-started/intro
- OpenTelemetry, **Generative AI Semantic Conventions**: https://opentelemetry.io/docs/specs/semconv/gen-ai/
- Hugging Face, **LLM Course**: https://huggingface.co/learn/llm-course/chapter1/1
- FastAPI documentation: https://fastapi.tiangolo.com/
- Docker documentation: https://docs.docker.com/
- PostgreSQL documentation: https://www.postgresql.org/docs/

## 14. Son değerlendirme

Paylaşım ve Git geçmişi birlikte değerlendirildiğinde doğru konum şudur:

- Kamuya açık 9 aylık programda Month 1 tamamlandı.
- Month 2'nin dense embeddings ve stronger evals bölümleri ilerledi.
- Dense/hybrid retrieval ve reranker bileşeni var; gerçek PDF, birleşik rerank pipeline, gerçek LLM ve deployment henüz tamamlanmadı.
- Program ilerlemesi bu nedenle yaklaşık %15–16'dır.
- Mentorun ilk hafta programı mevcut yol haritasıyla uyumludur ve Month 2 içinde staj sprinti olarak yürütülecektir.
- Bilgisayar mühendisliği ve yol haritası öncesi çalışmalar nedeniyle gerçek başlangıç seviyen bundan daha yüksektir.

En büyük risk temel eksikliği değildir. En büyük risk, çok sayıda küçük deney üretip bunları deploy edilmiş, ölçülmüş ve kullanıcı değeri olan bir üründe birleştirememektir.

Başarı ölçütü:

> Kaç konu gördüğün değil; çalışan sistemi ne kadar iyi ölçtüğün, ne zaman cevap vermemesi gerektiğini ne kadar doğru tasarladığın, güvenli şekilde deploy ettiğin ve kararlarını ne kadar net savunabildiğindir.
