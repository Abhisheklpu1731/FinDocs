import re
import time

import streamlit as st

from findocs_qa.generation.answer import Answerer
from findocs_qa.ingest.pipeline import ingest_pdf
from findocs_qa.library import Library

MAX_UPLOAD_MB = 100
CITATION = re.compile(r"\[([^\[\],]+),\s*page\s*(\d+)\]")
EXAMPLES = [
    "What was the total revenue for the year?",
    "How many employees does the company have?",
    "Who is the CEO and managing director?",
    "What dividend was declared per share?",
]
# How much of the overall progress bar each stage owns
STAGE_WEIGHTS = {"parse": 0.35, "chunk": 0.05, "embed": 0.55, "index": 0.05}
STAGE_TITLES = {
    "parse": "Read the PDF",
    "chunk": "Split into passages",
    "embed": "Embed passages",
    "index": "Build search index",
}

st.set_page_config(page_title="FinDocs QA", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600&display=swap');
    .stApp, .stApp p, .stApp label, .stApp li, .stApp button, .stApp input, .stApp textarea,
    .stApp [data-testid="stCaptionContainer"] {font-family: 'Inter', sans-serif;}
    .block-container {padding-top: 2.5rem; max-width: 1100px;}
    .hero {margin-bottom: 1.2rem;}
    .hero .eyebrow {text-transform: uppercase; letter-spacing: .16em; font-size: .75rem;
                    font-weight: 600; opacity: .6; margin-bottom: .3rem;}
    .hero h1 {font-family: 'Fraunces', serif; font-weight: 700; font-size: 3.4rem; line-height: 1.05;
              letter-spacing: -.02em; margin: 0; padding: 0;}
    .hero p {font-size: 1.1rem; opacity: .7; margin: .6rem 0 0 0;}
    .stApp h2, .stApp h3, [data-testid="stSidebar"] h2 {font-family: 'Fraunces', serif; font-weight: 600;
                                                         letter-spacing: -.01em;}
    .stTabs [data-baseweb="tab"] {font-size: 1.05rem; font-weight: 500;}
    .pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem;
           background: rgba(120,120,255,.15); margin-right: 6px;}
    .pill.warn {background: rgba(240,170,40,.22);}
    .stage {padding: 4px 0; font-size: .95rem;}
    .stage.pending {opacity: .45;}
    .dot {display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 10px;
          border: 2px solid currentColor; opacity: .6; box-sizing: border-box;}
    .dot.done {background: #2eb872; border-color: #2eb872; opacity: 1;}
    .dot.active {background: #4f6bed; border-color: #4f6bed; opacity: 1;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading search models (first run only)...")
def get_library() -> Library:
    return Library()


@st.cache_resource
def get_answerer() -> Answerer:
    return Answerer(retriever=get_library().retriever)


@st.cache_data(show_spinner=False, max_entries=64)
def page_png(doc_id: str, page: int, version: int) -> bytes | None:
    return get_library().page_image(doc_id, page)


library = get_library()
state = st.session_state
state.setdefault("messages", [])
state.setdefault("pending", None)
state.setdefault("last_ingest", None)
state.setdefault("version", 0)   # bumps whenever documents change, to refresh cached images


# ---------------- sidebar: library ----------------

with st.sidebar:
    st.header("Library")
    st.caption("Tick the reports you want to search.")
    for doc_id, doc in library.docs.items():
        left, right = st.columns([5, 1])
        left.checkbox(
            doc["label"], value=True, key=f"scope_{doc_id}",
            help=f"{doc['n_pages']} pages · {doc['n_chunks']} passages",
        )
        if not doc["builtin"] and right.button("✕", key=f"del_{doc_id}", help="Remove this report"):
            library.delete(doc_id)
            state.version += 1
            st.rerun()
        kind = "uploaded" if not doc["builtin"] else "built-in"
        st.caption(f"{kind} · {doc['n_pages']} pages · {doc['n_chunks']} passages")
    if not library.docs:
        st.info("No reports yet. Add one in the **Add a report** tab.")
    st.divider()
    if st.button("Clear conversation", use_container_width=True, disabled=not state.messages):
        state.messages = []
        st.rerun()

selected = {d for d in library.docs if state.get(f"scope_{d}", True)}


# ---------------- helpers ----------------

def pretty_citations(text: str) -> str:
    """[doc_id, page 7] -> **[Infosys · p.7]**"""
    return CITATION.sub(lambda m: f"**[{library.label(m.group(1).strip())} · p.{m.group(2)}]**", text)


def render_sources(result: dict, key: str) -> None:
    cited = {(d.strip(), int(p)) for d, p in CITATION.findall(result["answer"])}
    st.markdown(f"**Sources** ({len(result['chunks'])} passages used)")
    for i, c in enumerate(result["chunks"]):
        is_cited = (c["doc_id"], c["page"]) in cited
        mark = "  |  cited" if is_cited else ""
        title = f"{library.label(c['doc_id'])} · page {c['page']}{mark}"
        with st.expander(title):
            moved = c.get("hybrid_rank")
            st.markdown(
                f"<span class='pill'>rerank score {c['score']:.2f}</span>"
                f"<span class='pill'>search rank #{moved}</span>"
                + ("<span class='pill warn'>scanned page (OCR)</span>" if c.get("scanned") else ""),
                unsafe_allow_html=True,
            )
            st.text(c["text"])
            if st.toggle("Show the original page", value=True, key=f"{key}_img_{i}"):
                png = page_png(c["doc_id"], c["page"], state.version)
                if png:
                    st.image(png, use_container_width=True)
                else:
                    st.caption("Original PDF not available.")


def render_message(msg: dict, idx: int) -> None:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
        return
    with st.chat_message("assistant"):
        if msg.get("error"):
            st.error(msg["error"])
            return
        result = msg["result"]
        st.markdown(pretty_citations(result["answer"]))
        st.caption(
            f"{msg['seconds']:.1f}s · {result['tokens']} tokens · "
            f"searched {', '.join(library.label(d) for d in msg['scope'])}"
        )
        render_sources(result, key=f"m{idx}")


def ask(question: str) -> None:
    """Run the full question pipeline with live progress, then store the answer."""
    state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    started = time.time()
    steps = {"hybrid": "Searching (keywords + meaning)", "rerank": "Re-ranking passages",
             "generate": "Writing the answer"}
    with st.chat_message("assistant"):
        status = st.status("Working on it...", expanded=True)

        def on_step(name: str, detail: str) -> None:
            status.write(f"{steps[name]} — {detail}")

        try:
            answerer = get_answerer()
            answerer.retriever = library.retriever
            result = answerer.answer(question, doc_ids=selected, on_step=on_step)
        except Exception as e:   # surface API/key problems in the chat instead of a stack trace
            status.update(label="Something went wrong", state="error")
            state.messages.append({"role": "assistant", "error": f"{type(e).__name__}: {e}"})
            return
        seconds = time.time() - started
        status.update(label=f"Done in {seconds:.1f}s", state="complete", expanded=False)

    state.messages.append({"role": "assistant", "result": result, "seconds": seconds,
                           "scope": sorted(selected)})
    st.rerun()


def stage_html(current: str | None, done: set[str]) -> str:
    rows = []
    for key, title in STAGE_TITLES.items():
        if key in done:
            rows.append(f"<div class='stage'><span class='dot done'></span>{title}</div>")
        elif key == current:
            rows.append(f"<div class='stage'><span class='dot active'></span><b>{title}</b></div>")
        else:
            rows.append(f"<div class='stage pending'><span class='dot'></span>{title}</div>")
    return "".join(rows)


def process_upload(uploaded, label: str) -> None:
    doc_id = library.unique_id(uploaded.name)
    pdf_path = library.add_upload(doc_id, label, uploaded.getvalue())

    status = st.status(f"Processing {label}...", expanded=True)
    with status:
        bar = st.progress(0.0)
        stages_box = st.empty()
        message_box = st.empty()
        done_stages: set[str] = set()
        offsets = {}
        acc = 0.0
        for key, w in STAGE_WEIGHTS.items():
            offsets[key] = acc
            acc += w
        started = time.time()
        result = None
        try:
            for ev in ingest_pdf(pdf_path, doc_id, library.embedder):
                if ev["stage"] == "done":
                    result = ev["result"]
                    done_stages = set(STAGE_WEIGHTS)
                    bar.progress(1.0)
                    stages_box.markdown(stage_html(None, done_stages), unsafe_allow_html=True)
                    break
                stage = ev["stage"]
                done_stages = {s for s in STAGE_WEIGHTS if offsets[s] < offsets[stage]}
                frac = ev["done"] / max(ev["total"], 1)
                bar.progress(min(offsets[stage] + STAGE_WEIGHTS[stage] * frac, 0.99))
                stages_box.markdown(stage_html(stage, done_stages), unsafe_allow_html=True)
                message_box.caption(ev["message"])
            message_box.empty()
            library.commit_upload(doc_id, label, result)
        except Exception as e:
            library.discard_upload(doc_id)
            status.update(label="Processing failed", state="error")
            st.error(f"{type(e).__name__}: {e}")
            return
    status.update(label=f"Finished in {time.time() - started:.1f}s", state="complete", expanded=False)
    state.last_ingest = {"doc_id": doc_id, "label": label, "stats": result["stats"],
                         "sample": result["chunks"][:3]}
    state.version += 1
    st.rerun()


# ---------------- main ----------------

st.markdown(
    "<div class='hero'><div class='eyebrow'>Annual report intelligence</div><h1>FinDocs QA</h1>"
    "<p>Ask questions about company annual reports. Every answer cites the page it came from.</p></div>",
    unsafe_allow_html=True,
)

tab_ask, tab_add = st.tabs(["Ask", "Add a report"])

with tab_ask:
    if library.retriever is None:
        st.info("Add a report first, then come back to ask questions.")
    elif not selected:
        st.warning("Select at least one report in the sidebar.")
    else:
        names = ", ".join(library.label(d) for d in sorted(selected))
        st.caption(f"Searching: **{names}**")

    for i, m in enumerate(state.messages):
        render_message(m, i)

    if not state.messages and library.retriever is not None and selected:
        st.markdown("**Try one of these**")
        cols = st.columns(2)
        for i, q in enumerate(EXAMPLES):
            if cols[i % 2].button(q, key=f"ex_{i}", use_container_width=True):
                state.pending = q
                st.rerun()

    typed = st.chat_input("Ask about the selected reports...",
                          disabled=library.retriever is None or not selected)
    question = typed or state.pending
    state.pending = None
    if question:
        ask(question)

with tab_add:
    st.markdown(
        "Upload any company's annual report (PDF). It is read, split into passages, "
        "embedded and indexed on this machine, then you can ask questions about it right away."
    )
    uploaded = st.file_uploader("PDF report", type="pdf", accept_multiple_files=False)
    if uploaded is not None:
        size_mb = uploaded.size / 1_000_000
        default_label = re.sub(r"[_\-]+", " ", uploaded.name.rsplit(".", 1)[0]).strip().title()
        label = st.text_input("Name shown in the app", value=default_label)
        st.caption(f"{uploaded.name} · {size_mb:.1f} MB")
        if size_mb > MAX_UPLOAD_MB:
            st.error(f"File is larger than {MAX_UPLOAD_MB} MB.")
        elif st.button("Process report", type="primary", disabled=not label.strip()):
            process_upload(uploaded, label.strip())

    if state.last_ingest:
        info = state.last_ingest
        s = info["stats"]
        st.success(f"**{info['label']}** is ready. Switch to the **Ask** tab to query it.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pages", s["total_pages"])
        c2.metric("Pages with text", s["text_pages"])
        c3.metric("Passages", s["chunks"])
        c4.metric("Avg words / passage", s["avg_words"])
        if s["scanned_pages"]:
            st.warning(
                f"{s['scanned_pages']} page(s) are scanned images read through the PDF's OCR layer. "
                "Numbers on those pages may be garbled, so double-check answers that cite them."
            )
        if s["empty_pages"]:
            st.caption(f"{s['empty_pages']} page(s) had no extractable text and were skipped "
                       "(images, blank pages).")
        with st.expander("Preview of the first passages"):
            for c in info["sample"]:
                st.caption(f"page {c['page']}")
                st.text(c["text"][:500])
