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

# Tabs
tab_search, tab_add, tab_delete, tab_status = st.tabs([
    "Search",
    "Add File",
    "Delete File",
    "System Status"
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
                "use_cross_encoder": use_cross_encoder
            }
            try:
                with st.spinner("Searching..."):
                    res = requests.post(f"{API_BASE}/search", json=payload, timeout=60)
                    res.raise_for_status()
                    data = res.json()

                    if show_raw_json:
                        st.expander("Raw Response").code(json.dumps(data, indent=2, ensure_ascii=False),
                                                         language="json")

                    # Parse response: {"query": "...", "results": [...]}
                    results = data.get("results", []) if isinstance(data, dict) else data

                    if not results:
                        st.info("No results found.")
                    else:
                        st.success(f"Found {len(results)} chunks")
                        for i, item in enumerate(results, 1):
                            # Extract fields according to actual API response
                            chunk_text = item.get("chunk_text", "")
                            doc_id = item.get("doc_id", "unknown")
                            file_path = item.get("file_path", "")
                            chunk_id = item.get("chunk_id", "")
                            chunk_index = item.get("chunk_index", -1)

                            # Scores are nested in "scores" object
                            scores = item.get("scores", {})
                            display_score = scores.get(score_type, "N/A")

                            with st.expander(f"Result {i} | {score_type}: {display_score} | Doc: {doc_id}"):
                                st.markdown(f"**File:** `{file_path}`")
                                st.markdown(f"**Chunk ID:** `{chunk_id}` | **Index:** {chunk_index}")
                                st.markdown("**Content:**")
                                st.code(chunk_text, language="markdown")

                                # Show all scores in a table
                                st.markdown("**All Scores:**")
                                score_data = {k: f"{v:.6f}" if isinstance(v, (int, float)) else v for k, v in
                                              scores.items()}
                                st.json(score_data)

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Check if the service is running.")
            except requests.exceptions.HTTPError as e:
                st.error(f"HTTP error: {e}")
                if e.response is not None:
                    st.code(e.response.text)
            except Exception as e:
                st.error(f"Request error: {type(e).__name__}: {e}")

# 2. ADD FILE
with tab_add:
    st.header("Index New Document")
    st.info("The file must be located in data/documents/ inside the container or mounted via Docker volume.")
    file_path = st.text_input("File Path", value="/app/data/documents/new_doc.txt")

    if st.button("Index File", type="primary"):
        try:
            with st.spinner("Sending to index..."):
                res = requests.post(f"{API_BASE}/add-file", json={"path": file_path}, timeout=120)
                res.raise_for_status()
                data = res.json()
                status = data.get("status", "unknown")
                if status == "indexed":
                    st.success("File successfully indexed.")
                elif status == "reindexed":
                    st.warning("File modified. Re-indexing completed.")
                elif status == "already_indexed":
                    st.info("File already indexed, no changes.")
                else:
                    st.info(f"Status: {status}")
                if show_raw_json:
                    st.json(data)
        except Exception as e:
            st.error(f"Error: {type(e).__name__}: {e}")

# 3. DELETE FILE
with tab_delete:
    st.header("Delete Document from Index")
    del_path = st.text_input("File Path to Delete", value="/app/data/documents/old_doc.txt")

    if st.button("Delete File", type="secondary", use_container_width=True):
        try:
            with st.spinner("Deleting..."):
                res = requests.post(f"{API_BASE}/delete-file", json={"path": del_path}, timeout=30)
                res.raise_for_status()
                st.success("File removed from index (or already missing).")
                if show_raw_json:
                    st.json(res.json())
        except Exception as e:
            st.error(f"Error: {type(e).__name__}: {e}")

# 4. SYSTEM STATUS
with tab_status:
    st.header("System Status")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh Status", type="primary", use_container_width=True):
            # Try health check
            try:
                health = requests.get(f"{API_BASE}/health", timeout=10)
                health.raise_for_status()
                st.subheader("Health Check")
                st.json(health.json())
            except requests.exceptions.Timeout:
                st.error("Health check timed out (10s). Service might be busy.")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to API. Check if service is running.")
            except Exception as e:
                st.error(f"Health check failed: {type(e).__name__}: {e}")

            st.markdown("---")

            # Try index info
            try:
                with st.spinner("Loading index information..."):
                    idx = requests.get(f"{API_BASE}/index-info", timeout=30)
                    idx.raise_for_status()
                    st.subheader("Index Information")
                    st.json(idx.json())
            except requests.exceptions.Timeout:
                st.error("Index info timed out (30s). Large index may take longer.")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to API.")
            except Exception as e:
                st.error(f"Index info failed: {type(e).__name__}: {e}")

    with col2:
        st.markdown("""
        ### System Information
        - Dense: BAAI/bge-small-en-v1.5
        - Sparse: TF-weighted BoW
        - Fusion: RRF (Reciprocal Rank Fusion)
        - Rerank: ms-marco-MiniLM-L-6-v2 (optional)
        - State: state/index_state.json
        """)

        # Add quick stats
        st.markdown("---")
        st.markdown("### Quick Links")
        st.markdown("- [Health Check](http://localhost:8000/health)")
        st.markdown("- [Index Info](http://localhost:8000/index-info)")
        st.markdown("- [API Docs](http://localhost:8000/docs)")

# Footer
st.divider()
st.caption("IR Retrieval Service UI")