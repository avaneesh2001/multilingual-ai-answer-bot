import streamlit as st

from src.language import LanguageDetector
from src.search import Search
from src.vectorstore import VectorStore
from src.reranker import Reranker
from src.summarizer import SummarizeAnswer


st.set_page_config(page_title="Multilingual RAG Demo", page_icon="🔎", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Archivo+Black&display=swap');

.stApp {
    background:
        radial-gradient(circle at 18% 22%, rgba(20,20,20,0.12) 0 6px, transparent 7px),
        radial-gradient(circle at 82% 78%, rgba(20,20,20,0.10) 0 4px, transparent 5px),
        #f3eddf;
    color: #1f1f1f;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.4rem;
}

html, body, [class*="css"] {
    font-family: 'Space Mono', monospace;
}

h1 {
    font-family: 'Archivo Black', sans-serif !important;
    font-size: 3.2rem !important;
    letter-spacing: -2px;
    color: #1c1c1c !important;
}

h2, h3 {
    font-family: 'Archivo Black', sans-serif !important;
    color: #1c1c1c !important;
}

p, label, span, div {
    color: #1f1f1f;
}

.retro-topline {
    display: flex;
    align-items: center;
    gap: 18px;
    margin-bottom: 40px;
    font-size: 1.05rem;
}

.retro-logo {
    width: 54px;
    height: 54px;
    border: 3px solid #1c1c1c;
    border-radius: 50%;
    display: grid;
    place-items: center;
    font-size: 24px;
}

.retro-hero {
    border-top: 3px solid #1c1c1c;
    border-bottom: 3px solid #1c1c1c;
    padding: 42px 0 34px 0;
    margin-bottom: 32px;
}

.retro-kicker {
    font-weight: 700;
    text-transform: lowercase;
    border-bottom: 3px solid #1c1c1c;
    display: inline-block;
    margin-bottom: 6px;
}

.stTextInput > div > div > input {
    background: #f8f1df;
    color: #1c1c1c;
    border: 3px solid #1c1c1c;
    border-radius: 0;
    padding: 14px 16px;
    font-size: 1rem;
    box-shadow: 6px 6px 0 #1c1c1c;
}

.stButton > button {
    background: #1c1c1c;
    color: #f8f1df;
    border: 3px solid #1c1c1c;
    border-radius: 0;
    padding: 0.75rem 1.6rem;
    font-weight: 700;
    box-shadow: 6px 6px 0 #d34f2f;
}

.stButton > button:hover {
    background: #d34f2f;
    color: #f8f1df;
    border-color: #1c1c1c;
}

[data-testid="stStatusWidget"] {
    background: #f8f1df;
    border: 3px solid #1c1c1c;
    border-radius: 0;
    box-shadow: 8px 8px 0 #1c1c1c;
}

[data-testid="stExpander"] {
    background: #f8f1df;
    border: 3px solid #1c1c1c;
    border-radius: 0;
    box-shadow: 6px 6px 0 #1c1c1c;
    margin-bottom: 18px;
}

[data-testid="stExpander"] summary {
    font-weight: 700;
}

.stAlert {
    background: #f8f1df;
    border: 3px solid #1c1c1c;
    border-radius: 0;
}

.answer-card {
    background: #f8f1df;
    color: #1c1c1c;
    border: 3px solid #1c1c1c;
    padding: 28px;
    box-shadow: 8px 8px 0 #1c1c1c;
    line-height: 1.7;
    font-size: 1.05rem;
}

.source-card {
    background: #f8f1df;
    border: 3px solid #1c1c1c;
    padding: 16px 18px;
    margin-bottom: 14px;
    box-shadow: 6px 6px 0 #1c1c1c;
}

.source-card a {
    color: #1c1c1c;
    font-weight: 700;
    text-decoration: underline;
}

hr {
    border: none;
    border-top: 3px solid #1c1c1c;
}
.stButton > button {
    background: #d34f2f;
    color: #f8f1df;
    border: 3px solid #1c1c1c;
    box-shadow: 5px 5px 0 #1c1c1c;
}
[data-testid="stStatusWidget"] {
    background: #f8f1df !important;
    border: 3px solid #1c1c1c !important;
    box-shadow: 6px 6px 0 #1c1c1c !important;
}

[data-testid="stStatusWidget"] * {
    color: #1c1c1c !important;
}
.stAlert {
    background: #f8f1df !important;
    color: #1c1c1c !important;
    border: 3px solid #1c1c1c !important;
    border-radius: 0 !important;
    box-shadow: 4px 4px 0 #1c1c1c !important;
}
            
            /* Fix dark JSON/code blocks */
pre, code, .stCodeBlock {
    background: #f8f1df !important;
    color: #1c1c1c !important;
    border: 3px solid #1c1c1c !important;
    border-radius: 0 !important;
    box-shadow: 5px 5px 0 #1c1c1c !important;
    font-family: 'Space Mono', monospace !important;
}

/* Streamlit JSON viewer */
[data-testid="stJson"] {
    background: #f8f1df !important;
    color: #1c1c1c !important;
    border: 3px solid #1c1c1c !important;
    box-shadow: 5px 5px 0 #1c1c1c !important;
            
/* Fix status container */
[data-testid="stStatusWidget"] {
    background: #f8f1df !important;
    border: 3px solid #1c1c1c !important;
    box-shadow: 6px 6px 0 #1c1c1c !important;
}

/* THIS is the black header */
[data-testid="stStatusWidget"] > div:first-child {
    background: #f8f1df !important;
    color: #1c1c1c !important;
    border-bottom: 3px solid #1c1c1c !important;
}

/* text inside header */
[data-testid="stStatusWidget"] * {
    color: #1c1c1c !important;
}
            
}
.source-card {
border: 3px solid #1c1c1c;
padding: 14px 18px;
margin-bottom: 12px;
background: #f8f1df;
box-shadow: 5px 5px 0 #1c1c1c;
}
.answer-card {
    background: #f8f1df;
    border: 3px solid #1c1c1c;
    padding: 26px;
    margin-top: 10px;
    box-shadow: 8px 8px 0 #1c1c1c;
    line-height: 1.8;
    font-size: 1.05rem;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_components():
    return {
        "lang_detector": LanguageDetector(),
        "search": Search(),
        "reranker": Reranker(),
        "summarizer": SummarizeAnswer(),
    }


components = load_components()

lang_detector = components["lang_detector"]
search = components["search"]
reranker = components["reranker"]
summarize_n_answer = components["summarizer"]


st.title("🔎 Multilingual RAG Demo")
st.write(
    "Ask a question in any language. The system searches, retrieves, reranks, summarizes, and answers with sources."
)

query = st.text_input("Enter your query", value="माउंट फूजी का महत्व समझाइए")

run = st.button("Generate Answer")


if run and query.strip():

    with st.status("Running pipeline...", expanded=True) as status:

        st.write("Detecting language...")
        language = lang_detector.detect_language(query)
        st.info(f"Detected language: ` {language} `")

        st.write("Translating query...")
        translated_query = lang_detector.translate(query, language)
        st.info(f"Translated query: {translated_query}")

        st.write("Searching web and wikis...")
        search_res = search.search(translated_query)

        with st.expander("Search Results"):
            with st.expander("See search results"):
                for item in search_res:
                    st.markdown(
                        f"""
                    <div class="source-card">
                    <strong>{item.get("title","No title")}</strong><br>
                    <a href="{item.get("url","#")}" target="_blank">{item.get("url","")}</a>
                    <p>{item.get("snippet","")}</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

        st.write("Creating vector store...")
        vector_store = VectorStore()
        vector_store.create_vector_store(search_res)

        st.write("Retrieving top documents...")
        texts_top_10 = vector_store.retrieval(translated_query)

        with st.expander("Retrieved Context"):
            for i, (text, metadata) in enumerate(texts_top_10, start=1):
                st.markdown(f"### Result {i}")
                st.write(text)
                st.caption(metadata)

        st.write("Reranking retrieved documents...")
        texts = [text for text, _ in texts_top_10]

        reranked_context = reranker.rerank(translated_query, texts)

        text_to_metadata = {text: metadata for text, metadata in texts_top_10}

        reranked_metadata = [
            text_to_metadata[text]
            for text in reranked_context
            if text in text_to_metadata
        ]

        with st.expander("Reranked Context"):
            for i, text in enumerate(reranked_context, start=1):
                st.markdown(f"### Reranked Chunk {i}")
                st.write(text)

        st.write("Summarizing context...")
        context_summary = summarize_n_answer.summarizer(
            reranked_context, translated_query
        )

        with st.expander("Context Summary"):
            st.write(context_summary)

        st.write("Generating final answer...")
        answer = summarize_n_answer.answer(context_summary, translated_query)

        st.write("Translating answer back...")
        translated_answer = lang_detector.retranslate(answer, language)

        status.update(label="Pipeline completed", state="complete", expanded=False)

    st.subheader("Final Answer")
    st.markdown(
        f"""
                <div class="answer-card">
                {translated_answer}
                </div>
                """,
        unsafe_allow_html=True,
    )

    st.markdown("### Sources")

    for source in reranked_metadata:
        url = source.get("url", "")
        title = source.get("title", "Source")

        st.markdown(
            f"""
        <div class="source-card">
            <strong>{title}</strong><br>
            <a href="{url}" target="_blank">{url}</a>
        </div>
        """,
            unsafe_allow_html=True,
        )
