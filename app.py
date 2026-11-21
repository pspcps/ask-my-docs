"""Streamlit interface for the shared document service."""

from pathlib import Path

import streamlit as st

from service import DocumentService, ServiceError
from workspace import DEMO_PROMPTS, FILE_TYPES, ROOT, demo_answer, demo_passages

st.set_page_config(page_title="Ask My Docs", page_icon=":material/library_books:", layout="wide")
st.html(ROOT / "workspace.css")
state = st.session_state
state.setdefault("messages", [])
state.setdefault("mode", "Sample handbook")


@st.cache_resource
def document_service():
    return DocumentService()


service = document_service()


def run_action(action, *args):
    try:
        result = action(*args)
    except ServiceError as error:
        notice = ("error", str(error))
    else:
        if result is None:
            notice = ("info", "Document removed.")
        elif result.get("duplicate"):
            notice = ("info", f"{result['name']} is already in the library or queued.")
        else:
            notice = ("info", f"{result['name']} · version {result['version']} queued.")
    state.setdefault("notices", []).append(notice)


def library_snapshot(documents):
    return [(item["id"], item["latest_job"]["id"], item["latest_job"]["status"],
             item["latest_job"]["progress"], item["latest_job"]["attempts"]) for item in documents]


@st.fragment(run_every=2)
def job_status(snapshot):
    current = service.documents()
    if library_snapshot(current) != snapshot:
        st.rerun()
    for item in current:
        job = item["latest_job"]
        if job["status"] in ("queued", "running"):
            st.progress(job["progress"], text=f"{job['name']} · v{job['version']} · {job['status']}")


def display_message(message):
    avatar = ":material/person:" if message["role"] == "user" else ":material/auto_stories:"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["text"])
        sources = message.get("sources", [])
        if sources:
            with st.expander(f"Source excerpts ({len(sources)})"):
                for source in sources:
                    st.text(source["label"])
                    if source.get("sample"):
                        st.markdown(source["excerpt"])
                    else:
                        st.text(source["excerpt"])


documents = service.documents()
with st.sidebar:
    st.html("""<div class="brand"><span class="brand-mark">A</span><div><strong>Ask My Docs</strong>
    <small>Document workspace</small></div></div>""")
    st.radio("Workspace", ["Sample handbook", "Document library"], key="mode")
    demo = state.mode == "Sample handbook"
    if not demo:
        st.subheader("Manage documents")
        for level, notice in state.pop("notices", []):
            if level == "error":
                st.error(notice)
            else:
                st.info(notice)
        labels = {item["id"]: f"{item['name']} · {item['id'][:6]}" for item in documents}
        target = st.selectbox(
            "Action", [None, *labels], format_func=lambda item: labels.get(item, "Add a document"),
        )
        files = st.file_uploader(
            "Replacement file" if target else "Add documents", type=FILE_TYPES,
            accept_multiple_files=target is None, key=f"files-{target}",
        )
        if st.button("Queue replacement" if target else "Queue uploads", disabled=not files, width="stretch"):
            for file in ([files] if target else files):
                run_action(service.enqueue, file.name, file.getvalue(), target)
            st.rerun()
        if target:
            item = next(item for item in documents if item["id"] == target)
            job = item["latest_job"]
            if job["status"] == "failed":
                st.error(job["error"])
                if st.button("Retry failed job", width="stretch"):
                    run_action(service.retry, job["id"])
                    st.rerun()
            if item["active"] and job["status"] not in ("queued", "running"):
                if st.button("Rebuild embeddings", width="stretch"):
                    run_action(service.rebuild, target)
                    st.rerun()
            with st.popover("Delete document", width="stretch"):
                st.write("Remove this document, its active passages, and its job history.")
                if st.button("Confirm deletion"):
                    run_action(service.delete, target)
                    st.rerun()
        job_status(library_snapshot(documents))
        st.caption("Run `python worker.py` to process queued jobs.")

active = {item["id"]: item for item in documents if item["active"]}
history_key = (demo, () if demo else tuple((key, item["active_version"]) for key, item in active.items()))
if state.get("history_key") != history_key:
    state.history_key = history_key
    state.messages = []

st.html(
    '<div class="hero"><h1>Ask My Docs<span>.</span></h1>'
    '<p>Ask a question, then go straight to the passage behind the answer.</p></div>'
)
chat_column, context_column = st.columns([2.4, 1], gap="large")
with context_column:
    with st.container(border=True):
        st.subheader("Active library")
        if demo:
            st.caption("Sample handbook · no model needed")
            st.markdown(f"**1** document · **{len(demo_passages())}** passages")
            st.text("sample_document.txt")
            st.caption("Try the example questions or search the handbook with your own wording.")
            selected = []
        else:
            st.caption("Persistent library · shared with the API")
            count = sum(len(item["chunk_ids"]) for item in active.values())
            st.markdown(f"**{len(active)}** active · **{count}** {'passage' if count == 1 else 'passages'}")
            selected = st.multiselect(
                "Search within", list(active), default=list(active), format_func=lambda item: labels[item],
                key=f"scope-{history_key[1]}",
            )
            for item in documents:
                job = item["latest_job"]
                current = f"v{item['active_version']} active · " if item["active"] else ""
                st.text(f"{item['name']} · {current}v{job['version']} {job['status']}")
    if state.messages:
        if st.button("Clear conversation", icon=":material/delete_sweep:", width="stretch"):
            state.messages = []
            st.rerun()
        transcript = []
        for message in state.messages:
            transcript.append(f"## {message['role'].title()}\n\n{message['text']}")
            for source in message.get("sources", []):
                transcript.append(f"{source['label']}\n\n{source['excerpt']}")
        st.download_button("Export conversation", "\n\n".join(transcript), "ask-my-docs.md",
                           "text/markdown", icon=":material/download:", width="stretch")

with chat_column:
    st.subheader("Conversation")
    prompt = None
    if not state.messages and demo:
        st.caption("Start with a question about the sample company handbook.")
        for example in DEMO_PROMPTS:
            if st.button(example, icon=":material/arrow_outward:", width="stretch"):
                prompt = example
    if not demo and not selected:
        st.info("Queue a document, then select a ready version from the active library.")
    for message in state.messages:
        display_message(message)
    typed = st.chat_input("Ask a question about your documents…", max_chars=1500, disabled=not demo and not selected)
    question = (typed or prompt or "").strip()
    if question:
        try:
            with st.spinner("Finding relevant passages…"):
                if demo:
                    answer, passages = demo_answer(question, demo_passages())
                    sources = [
                        {"label": f"[{index}] {item.name} · {item.location}", "excerpt": item.text, "sample": True}
                        for index, item in enumerate(passages, 1)
                    ]
                else:
                    result = service.ask(question, selected)
                    answer = result["answer"]
                    sources = [
                        {"label": f"[{item['number']}] {item['document_name']} · {item['location']} · v{item['version']}",
                         "excerpt": item["excerpt"]}
                        for item in result["citations"]
                    ]
        except ServiceError as error:
            st.error(str(error))
        else:
            state.messages.extend([
                {"role": "user", "text": question},
                {"role": "assistant", "text": answer, "sources": sources},
            ])
            st.rerun()
