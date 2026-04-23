import streamlit as st
import requests
import json

st.set_page_config(page_title="IR Retrieval UI", layout="wide")

# Configuration
API_BASE = st.sidebar.text_input(
    "API Base URL",
    value="http://localhost:8000",
    help="Address of the running FastAPI service"
)
st.sidebar.markdown("---")
show_raw_json = st.sidebar.checkbox("Show raw JSON response")
score_type = st.sidebar.selectbox(
    "Score to display",
    options=["qdrant_fusion_score", "doc_score", "cross_encoder_score"],
    index=0,
    help="Which score field to use for sorting display"
)

st.title("IR Retrieval Service UI")
st.caption("Local two-level pipeline: dense + sparse -> RRF -> optional Cross-Encoder")

tab_search, tab_files, tab_upload, tab_delete, tab_status = st.tabs([
    "Search",
    "List Files",
    "Upload File",
    "Delete File",
    "System Status",
])

# 1. SEARCH
with tab_search:
    st.header("Search Parameters")
    query = st.text_area("Enter search query", height=80,
                         placeholder="Example: mechanisms of drug resistance in bacteria")

    col1, col2, col3 = st.columns(3)
    with col1:
        top_k_doc = st.number_input("top_k_doc (Doc-level)", min_value=1, value=5)
        top_k_dense = st.number_input("top_k_dense (Chunk)", min_value=1, value=10)
    with col2:
        top_k_sparse = st.number_input("top_k_sparse (Chunk)", min_value=1, value=10)
        final_top_k = st.number_input("final_top_k", min_value=1, value=5)
    with col3:
        use_cross_encoder = st.toggle("Cross-Encoder Reranking", value=False)

    if st.button("Run Search", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a search query.")
        else:
            payload = {
                "query": query,
                "top_k_doc": int(top_k_doc),
                "top_k_dense": int(top_k_dense),
                "top_k_sparse": int(top_k_sparse),
                "final_top_k": int(final_top_k),
                "use_cross_encoder": use_cross_encoder,
            }
            try:
                with st.spinner("Searching..."):
                    res = requests.post(f"{API_BASE}/search", json=payload, timeout=60)
                    res.raise_for_status()
                    data = res.json()

                if show_raw_json:
                    st.expander("Raw Response").code(
                        json.dumps(data, indent=2, ensure_ascii=False), language="json"
                    )

                results = data.get("results", []) if isinstance(data, dict) else data
                if not results:
                    st.info("No results found.")
                else:
                    st.success(f"Found {len(results)} chunks")
                    for i, item in enumerate(results, 1):
                        scores = item.get("scores", {})
                        display_score = scores.get(score_type, "N/A")
                        doc_id = item.get("doc_id", "unknown")
                        with st.expander(f"Result {i} | {score_type}: {display_score} | Doc: {doc_id}"):
                            st.markdown(f"**File:** `{item.get('file_path', '')}`")
                            st.markdown(f"**Chunk ID:** `{item.get('chunk_id', '')}` | **Index:** {item.get('chunk_index', -1)}")
                            st.markdown("**Content:**")
                            st.code(item.get("chunk_text", ""), language="markdown")
                            st.markdown("**All Scores:**")
                            st.json({k: f"{v:.6f}" if isinstance(v, (int, float)) else v for k, v in scores.items()})

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Check if the service is running.")
            except requests.exceptions.HTTPError as e:
                st.error(f"HTTP error: {e}")
                if e.response is not None:
                    st.code(e.response.text)
            except Exception as e:
                st.error(f"Request error: {type(e).__name__}: {e}")

# 2. LIST FILES
with tab_files:
    st.header("Files on Disk")

    col_search, col_page_size = st.columns([3, 1])
    with col_search:
        filename_filter = st.text_input("Filter by filename", placeholder="e.g. 10009")
    with col_page_size:
        page_size = st.selectbox("Per page", [25, 50, 100, 200], index=1)

    if "files_page" not in st.session_state:
        st.session_state.files_page = 1

    if filename_filter:
        st.session_state.files_page = 1

    try:
        params = {
            "page": st.session_state.files_page,
            "page_size": page_size,
            "search": filename_filter,
        }
        res = requests.get(f"{API_BASE}/list-files", params=params, timeout=30)
        res.raise_for_status()
        data = res.json()

        files = data.get("files", [])
        total = data.get("total", 0)
        total_pages = max(1, -(-total // page_size))

        st.caption(f"{total} file(s) found — page {st.session_state.files_page} of {total_pages}")

        if files:
            rows = [{"Filename": f["filename"], "Size (bytes)": f["size_bytes"]} for f in files]
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.info("No files found.")

        col_prev, col_indicator, col_next = st.columns([1, 2, 1])
        with col_prev:
            if st.button("← Prev", disabled=st.session_state.files_page <= 1):
                st.session_state.files_page -= 1
                st.rerun()
        with col_indicator:
            st.markdown(f"<div style='text-align:center'>Page {st.session_state.files_page} / {total_pages}</div>",
                        unsafe_allow_html=True)
        with col_next:
            if st.button("Next →", disabled=st.session_state.files_page >= total_pages):
                st.session_state.files_page += 1
                st.rerun()

        if show_raw_json:
            st.expander("Raw Response").code(json.dumps(data, indent=2), language="json")

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API.")
    except Exception as e:
        st.error(f"Error: {type(e).__name__}: {e}")

# 3. UPLOAD FILE
with tab_upload:
    st.header("Upload & Index Document")
    st.info("File will be saved to the documents volume and immediately indexed.")

    uploaded = st.file_uploader("Choose a .txt or .md file", type=["txt", "md"])

    if uploaded is not None:
        st.markdown(f"**Filename:** `{uploaded.name}` — **Size:** {len(uploaded.getvalue())} bytes")
        preview = uploaded.getvalue().decode("utf-8", errors="ignore")[:500]
        with st.expander("Preview (first 500 chars)"):
            st.code(preview, language="markdown")

        if st.button("Upload & Index", type="primary"):
            try:
                with st.spinner("Uploading and indexing..."):
                    res = requests.post(
                        f"{API_BASE}/upload-file",
                        files={"file": (uploaded.name, uploaded.getvalue(), "text/plain")},
                        timeout=120,
                    )
                    res.raise_for_status()
                    data = res.json()

                status = data.get("status", "unknown")
                if status == "indexed":
                    st.success(f"`{data['filename']}` indexed successfully.")
                elif status == "reindexed":
                    st.warning(f"`{data['filename']}` already existed and was re-indexed.")
                elif status == "already_indexed":
                    st.info(f"`{data['filename']}` is already indexed and unchanged.")
                else:
                    st.info(f"Status: {status}")

                if show_raw_json:
                    st.json(data)

            except requests.exceptions.HTTPError as e:
                st.error(f"HTTP error: {e}")
                if e.response is not None:
                    st.code(e.response.text)
            except Exception as e:
                st.error(f"Error: {type(e).__name__}: {e}")

# 4. DELETE FILE
with tab_delete:
    st.header("Delete Document from Index")
    st.warning("This removes the file from both the index and the volume.")
    del_path = st.text_input("Relative file path", placeholder="e.g. my_doc.txt")

    if st.button("Delete File", type="secondary", use_container_width=True):
        if not del_path.strip():
            st.warning("Please enter a file path.")
        else:
            try:
                with st.spinner("Deleting..."):
                    full_path = f"/app/data/documents/{del_path.lstrip('/')}"
                    res = requests.post(f"{API_BASE}/delete-file", json={"path": full_path}, timeout=30)
                    res.raise_for_status()
                    data = res.json()
                    if data.get("status") == "deleted":
                        st.success(f"`{del_path}` removed from index and disk.")
                    else:
                        st.info(f"`{del_path}` was not found in the index.")
                    if show_raw_json:
                        st.json(data)
            except Exception as e:
                st.error(f"Error: {type(e).__name__}: {e}")

# 5. SYSTEM STATUS
with tab_status:
    st.header("System Status")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh Status", type="primary", use_container_width=True):
            try:
                health = requests.get(f"{API_BASE}/health", timeout=10)
                health.raise_for_status()
                st.subheader("Health Check")
                st.json(health.json())
            except requests.exceptions.Timeout:
                st.error("Health check timed out.")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to API.")
            except Exception as e:
                st.error(f"Health check failed: {type(e).__name__}: {e}")

            st.markdown("---")

            try:
                with st.spinner("Loading index information..."):
                    idx = requests.get(f"{API_BASE}/index-info", timeout=30)
                    idx.raise_for_status()
                    st.subheader("Index Information")
                    st.json(idx.json())
            except requests.exceptions.Timeout:
                st.error("Index info timed out.")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to API.")
            except Exception as e:
                st.error(f"Index info failed: {type(e).__name__}: {e}")

    with col2:
        st.markdown("""
        ### System Information
        - Dense: Qwen3-Embedding-0.6B
        - Sparse: BM25 (fastembed)
        - Fusion: RRF (Reciprocal Rank Fusion)
        - Rerank: bge-reranker-v2-m3 (optional)
        - State: state/index_state.json
        """)
        st.markdown("---")
        st.markdown("### Quick Links")
        st.markdown("- [Health Check](http://localhost:8000/health)")
        st.markdown("- [Index Info](http://localhost:8000/index-info)")
        st.markdown("- [API Docs](http://localhost:8000/docs)")

st.divider()
st.caption("IR Retrieval Service UI")
