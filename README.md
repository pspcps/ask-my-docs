# Ask My Docs

Ask questions across your documents, read the passages behind each answer, and export the conversation. Ask My Docs is a Streamlit workspace for local document Q&A with Ollama.

![Ask My Docs showing an annual leave answer, its source excerpt, and the active library](docs/demo.png)

## Try the demo

The app opens with a sample company handbook. Select **How much annual leave do employees get?**, then open **Source excerpts** to check the answer against the handbook. You can also type your own query. The demo uses preset answers for the suggested questions and keyword matching for other queries, so it runs without Ollama.

```bash
git clone https://github.com/pspcps/ask-my-docs.git
cd ask-my-docs
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Use Python 3.11 or 3.12. On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`. Open <http://localhost:8501>.

## Use your own documents

1. Install [Ollama](https://ollama.com/download) and start it with `ollama serve` if it is not already running.
2. Download the chat and embedding models.

   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

3. Select **Refresh engine status** in the sidebar. Upload one or more files and select **Index uploads**. You can also put files in `documents/` and select **Index folder**.
4. Use **Search within** to scope a question to specific files. Open **Source excerpts** to inspect the retrieved text. PDF sources show page numbers and CSV sources show row numbers.
5. Export the conversation as Markdown, including the source excerpts, from the active library panel.

The app accepts PDF, UTF-8 TXT and Markdown, DOCX, and CSV. Browser uploads are capped at 20 MB per file. Image-only PDFs require OCR before indexing. Indexing a new set replaces the active library and starts a new conversation. Each session keeps its index and conversation in memory. Export before closing the session if you want to keep the conversation.

## How it works

1. **Parse and split.** Extract text and keep source metadata, then split longer sections into overlapping passages.
2. **Embed and retrieve.** Ollama embeds the passages and the question. NumPy ranks the selected documents’ passages by cosine similarity and returns up to four.
3. **Answer and inspect.** The chat model receives numbered passages and is prompted to cite them. Open the excerpts to verify important claims against the source. Each question retrieves evidence independently.

The default setup sends document text and questions from Streamlit to the Ollama server on the same machine. If you set `OLLAMA_HOST` to a remote server, that server receives the text and questions. Keep Streamlit and Ollama on your own machine for a fully local setup.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` | Ollama API address |
| `ASK_DOCS_CHAT_MODEL` | `llama3.2` | Preferred installed chat model |
| `ASK_DOCS_EMBED_MODEL` | `nomic-embed-text` | Model used for passage and question embeddings |

## Project layout

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit workspace, indexing controls, and conversation |
| `workspace.py` | File parsing, passage retrieval, and Ollama integration |
| `workspace.css` | Workspace styling |
| `documents/sample_document.txt` | Sample handbook used by the demo |
| `docs/demo.png` | Screenshot of the running demo |
| `api.py`, `data_ingest.py` | Original optional Flask/Redis API, with separate `requirements-legacy-api.txt` dependencies |

Run `python -m unittest -v` for the workspace checks. The tests run without an Ollama server.
