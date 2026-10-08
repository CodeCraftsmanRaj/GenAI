# Experiment 8 — Development of a Domain-Specific RAG Chatbot for SPIT

**Student:** Raj Kalpesh Mathuria  
**UID:** 2023300139  
**Batch:** PE-C  
**Course:** Generative AI Lab (CE413), Semester VII

## Aim
Build a domain-specific Retrieval-Augmented Generation (RAG) chatbot for SPIT using approved institutional documents and, when supplied, anonymized/mock ERP exports. Evaluate how retrieval quality affects answer quality and groundedness.

## What is included
- Explicit PDF/HTML/TXT/Markdown/CSV/JSON text extraction.
- Word-based chunking with overlap and source metadata.
- Sentence-Transformers embeddings (`all-MiniLM-L6-v2`) and a FAISS inner-product index over normalized embeddings.
- Visible Top-k retrieval results and similarity scores.
- Grounded generation through Ollama (default) or an OpenAI-compatible chat-completions endpoint.
- A Streamlit UI, CLI, top-k retrieval evaluation script, and a small JSONL QA set.
- Privacy reminders and clearly labeled synthetic demo documents.

> **Important:** The included knowledge base is only a demonstration corpus, not a verified source of SPIT institutional facts. Replace/add approved official public documents and instructor-approved anonymized/mock ERP data before reporting domain-specific results. Do not use real student personal data.

## Project structure
```text
Experiment_8_SPIT_RAG/
├── app.py
├── main.py
├── config.yaml
├── requirements.txt
├── src/
│   ├── config.py
│   ├── documents.py
│   ├── retriever.py
│   ├── generator.py
│   ├── pipeline.py
│   └── evaluate.py
├── knowledge_base/
│   ├── website_pages/
│   ├── institutional_pdfs/
│   └── erp_export/
├── evaluation/qa.jsonl
└── results/
```

## Setup (uv on Linux)
```bash
cd Experiment_8_SPIT_RAG
uv venv
uv pip install -r requirements.txt
```
The first embedding-model run downloads `sentence-transformers/all-MiniLM-L6-v2` weights. Ensure network access on first run.

## Configure the generator
### Option A — Ollama (default)
Install and start Ollama, then pull the model configured in `config.yaml`:
```bash
ollama pull llama3.2:3b
ollama serve
```
If Ollama is already running as a service, do not start a second instance. Keep `llm.provider: ollama`.

### Option B — OpenAI-compatible API
Set `llm.provider: openai_compatible`, set `llm.base_url` and `llm.model` in `config.yaml`, and set the environment variable named by `llm.api_key_env`. Do not commit API keys.

## Add your corpus
- Save official public SPIT website pages as `.html`, `.md` or `.txt` under `knowledge_base/website_pages/`.
- Place instructor-provided institutional PDFs in `knowledge_base/institutional_pdfs/`.
- Place only instructor-provided anonymized/mock ERP data in `knowledge_base/erp_export/`.
- Record source URLs and dates in a metadata CSV of your own; keep original filenames for traceability.
- Remove the included synthetic demo file if you want the index to contain only official sources.

Supported extensions: PDF, TXT, Markdown, HTML, CSV and JSON.

## Run
### 1. Extract, chunk, embed and index
```bash
uv run main.py ingest
```

### 2. Inspect retrieval without calling the LLM
```bash
uv run main.py retrieve "What does the knowledge base say about missing evidence?" --top-k 3
```

### 3. Ask a question
```bash
uv run main.py ask "What does the knowledge base say about missing evidence?" --top-k 3
```

### 4. Launch the web UI
```bash
uv run streamlit run app.py
```

### 5. Run Top-k evaluation
```bash
uv run main.py evaluate
```
Results are written to `results/topk_evaluation_details.csv` and `results/topk_evaluation_summary.csv`. Hit rate is computed only for QA records that specify an expected source filename. The starter QA set is small; replace/expand it to the required 30 questions and populate expected sources from your actual corpus before using it for coursework conclusions.

## How the pipeline works
1. **Extraction:** reads supported file types and cleans HTML navigation/script/style content.
2. **Chunking:** splits extracted text into approximately 450-word chunks with 60-word overlap by default.
3. **Embedding:** creates normalized sentence embeddings.
4. **Indexing:** stores vectors in FAISS `IndexFlatIP`; with normalized vectors, inner product corresponds to cosine similarity.
5. **Retrieval:** embeds the question and returns Top-k chunks with similarity scores and source metadata.
6. **Prompt construction:** passes retrieved evidence to the LLM and instructs it to abstain when evidence is insufficient.
7. **Answer generation:** returns a response with source filenames, while the UI/CLI displays the actual retrieved chunks for auditing.

## Experiments aligned with the lab manual
- **Experiment A — Top-k:** run `evaluate` with k = 1, 3, 5 and 10; inspect Hit@k and retrieval latency. Extend the QA file to 30 questions with expected sources.
- **Experiment B — Retrieval failure:** temporarily remove/move the document containing the answer, re-ingest, and ask the same question. Check whether the system abstains. Restore the source afterwards.
- **Experiment C — Irrelevant/misleading context:** add clearly labeled irrelevant/conflicting test documents, re-ingest, and compare results. Do not add misinformation to the official corpus.
- **Experiment D — RAG vs ICL:** compare the same questions using an ICL prompt without retrieved documents and RAG with retrieval. Record answer accuracy, groundedness and unsupported claims manually or with a documented rubric.

## Metrics
- **Hit@k:** relevant source appears among the top-k retrieved chunks.
- **Precision@k:** relevant retrieved chunks / k (requires relevance labels per chunk).
- **MRR:** reciprocal rank of the first relevant result (requires relevance labels).
- **Answer accuracy:** generated answer matches the expected answer.
- **Faithfulness/groundedness:** claims are supported by retrieved evidence.
- **Hallucination rate:** fraction of answers containing unsupported claims.
- **Latency:** time to retrieve and generate an answer. The provided evaluation currently measures retrieval latency; record full generation latency separately for the full experiment.

## Limitations / honest reporting
- This starter project cannot claim measured answer accuracy or faithfulness until it is run against a curated corpus and labeled QA set.
- Similarity scores are retrieval relevance signals, not calibrated probabilities.
- The threshold in `config.yaml` is a tunable heuristic and should be evaluated rather than assumed optimal.
- A correct retrieval can still lead to an incorrect generated answer; inspect evidence and output together.
- The included demonstration documents are not official SPIT policy.

## Repository
Add your code to: https://github.com/CodeCraftsmanRaj/GenAI/tree/main/Experiment_8_RAG (update the repository path if you create it elsewhere).
