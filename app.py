"""Streamlit interface for the Ask My Docs workspace."""

from pathlib import Path

import requests
import streamlit as st

from workspace import (
    CHAT_MODEL, DEMO, DEMO_PROMPTS, EMBED_MODEL, FILE_TYPES, ROOT,
    answer_question, demo_answer, demo_passages, embed, ollama_request,
    passages_from_file,
)

st.set_page_config(page_title="Ask My Docs", page_icon=":material/library_books:", layout="wide")
st.html(ROOT / "workspace.css")

state = st.session_state
for key, value in {
    "mode": "demo",
    "passages": demo_passages(),
    "vectors": None,
    "messages": [],
    "workspace_revision": 0,
}.items():
    state.setdefault(key, value)


@st.cache_data(ttl=15, show_spinner=False)
def installed_models():
    try:
        return [item["name"] for item in ollama_request("tags").get("models", [])]
    except (requests.RequestException, ValueError, KeyError):
        return None


def set_library(mode, passages, vectors=None):
    state.mode = mode
    state.passages = passages
    state.vectors = vectors
    state.messages = []
    state.workspace_revision += 1


def index_files(files):
    passages = []
    problems = []
    indexed_names = set()
    for name, read in files:
        if name in indexed_names:
            problems.append(f"{name}: duplicate filename")
            continue
        try:
            passages.extend(passages_from_file(name, read()))
            indexed_names.add(name)
        except Exception as error:
            problems.append(f"{name}: {error}")
    if problems:
        for problem in problems:
            st.warning(problem)
    if not passages:
        st.error("No readable files were indexed. Your current library is still active.")
        return
    try:
        with st.spinner(f"Embedding {len(passages)} passages locally…"):
            vectors = embed([passage.text for passage in passages])
    except (requests.RequestException, ValueError, KeyError) as error:
        st.error(f"Indexing failed. Your current library is still active. {error}")
        return
    set_library("local", passages, vectors)
    state.index_notices = problems
    st.rerun()


def show_message(message):
    with st.chat_message(message["role"], avatar=":material/person:" if message["role"] == "user" else ":material/auto_stories:"):
        st.markdown(message["text"])
        sources = message.get("sources", [])
        if sources:
            with st.expander(f"Source excerpts ({len(sources)})"):
                for number, passage in enumerate(sources, 1):
                    st.text(f"[{number}] {passage.name} · {passage.location}")
                    if state.mode == "demo":
                        st.markdown(passage.text)
                    else:
                        st.text(passage.text)


models = installed_models()
embed_tag = EMBED_MODEL if ":" in EMBED_MODEL else f"{EMBED_MODEL}:latest"
chat_tag = CHAT_MODEL if ":" in CHAT_MODEL else f"{CHAT_MODEL}:latest"
chat_models = [model for model in models or [] if model.split(":")[0] != embed_tag.split(":")[0]]
can_index = embed_tag in (models or [])
pending_index = None

with st.sidebar:
    st.html("""<div class="brand"><span class="brand-mark">A</span><div><strong>Ask My Docs</strong>
    <small>Document workspace</small></div></div>""")
    st.subheader("Library")
    uploaded = st.file_uploader(
        "Add documents", type=FILE_TYPES, accept_multiple_files=True,
        help="PDF, TXT, Markdown, DOCX or CSV. Each upload can be up to 20 MB.",
    )
    if st.button("Index uploads", icon=":material/add_to_photos:", width="stretch",
                 disabled=not (uploaded and can_index)):
        pending_index = [(Path(file.name).name, file.getvalue) for file in uploaded]

    with st.expander("Use the local documents folder"):
        paths = sorted(
            path for path in (ROOT / "documents").glob("*")
            if path.is_file() and path.suffix.lower().lstrip(".") in FILE_TYPES and path != DEMO
        )
        st.caption(f"{len(paths)} supported files in the project’s documents/ folder.")
        if st.button("Index folder", width="stretch", disabled=not (paths and can_index)):
            pending_index = [(path.name, path.read_bytes) for path in paths]

    if st.button("Open sample handbook", icon=":material/auto_stories:", width="stretch",
                 disabled=state.mode == "demo"):
        set_library("demo", demo_passages())
        state.pop("index_notices", None)
        st.rerun()

    st.divider()
    st.subheader("Local engine")
    if models is None:
        st.caption("Ollama offline · sample mode available")
        st.code("ollama serve", language="bash")
    else:
        st.caption("Ollama connected")
        if not can_index:
            st.code(f"ollama pull {EMBED_MODEL}", language="bash")
        if not chat_models:
            st.code(f"ollama pull {CHAT_MODEL}", language="bash")
    model = None
    if chat_models:
        model = st.selectbox("Chat model", chat_models,
                             index=chat_models.index(chat_tag) if chat_tag in chat_models else 0)
    if st.button("Refresh engine status", width="stretch"):
        installed_models.clear()
        st.rerun()

if pending_index:
    index_files(pending_index)

names = list(dict.fromkeys(passage.name for passage in state.passages))
demo = state.mode == "demo"
st.html(
    '<div class="hero">'
    '<h1>Ask My Docs<span>.</span></h1>'
    '<p>Ask a question, then go straight to the passage behind the answer.</p></div>'
)

chat_column, context_column = st.columns([2.4, 1], gap="large")
with context_column:
    with st.container(border=True):
        st.subheader("Active library")
        st.caption("Sample handbook · no model needed" if demo else "Local documents · this session")
        st.markdown(f"**{len(names)}** {'document' if len(names) == 1 else 'documents'}"
                    f"  ·  **{len(state.passages)}** passages")
        if demo:
            st.text("sample_document.txt")
            st.caption("Try the example questions or search the handbook with your own wording.")
            selected = names
        else:
            selected = st.multiselect("Search within", names, default=names,
                                      key=f"scope_{state.workspace_revision}")
            for problem in state.get("index_notices", []):
                st.warning(problem)

    if state.messages:
        if st.button("Clear conversation", icon=":material/delete_sweep:", width="stretch"):
            state.messages = []
            st.rerun()
        transcript = []
        for message in state.messages:
            transcript.append(f"## {message['role'].title()}\n\n{message['text']}")
            for number, source in enumerate(message.get("sources", []), 1):
                transcript.append(f"[{number}] {source.name} · {source.location}\n\n{source.text}")
        st.download_button("Export conversation", "\n\n".join(transcript), "ask-my-docs.md",
                           "text/markdown", icon=":material/download:", width="stretch")

with chat_column:
    st.subheader("Conversation")
    if not state.messages:
        if demo:
            st.caption("Start with a question about the sample company handbook.")
        else:
            st.caption("Ask about the selected files. Answers include the passages retrieved for each question.")
    prompt = None
    if demo:
        if not state.messages:
            for example in DEMO_PROMPTS:
                if st.button(example, icon=":material/arrow_outward:", width="stretch"):
                    prompt = example

    for message in state.messages:
        show_message(message)

    if not demo and not model:
        st.info("Start Ollama and install a chat model to ask questions about this library.")
    if not selected:
        st.info("Select at least one document in the active library.")
    typed = st.chat_input("Ask a question about your documents…", max_chars=1500,
                          disabled=not selected or (not demo and model is None))
    question = (typed or prompt or "").strip()
    if question:
        try:
            with st.spinner("Finding relevant passages…"):
                answer, sources = (
                    demo_answer(question, state.passages) if demo else
                    answer_question(question, state.passages, state.vectors, model, selected)
                )
        except (requests.RequestException, ValueError, KeyError) as error:
            st.error(f"Couldn’t get an answer. Check Ollama and try again. {error}")
        else:
            state.messages.extend([
                {"role": "user", "text": question},
                {"role": "assistant", "text": answer, "sources": sources},
            ])
            st.rerun()
