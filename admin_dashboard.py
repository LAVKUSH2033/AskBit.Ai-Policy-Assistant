import streamlit as st
import os
from metrics import Metrics
from document_ingestion import DocumentIngester
from model_router import ModelRouter

st.title("Admin Dashboard")

# Initialize components
metrics = Metrics()
ingester = DocumentIngester()
router = ModelRouter()

# Metrics Overview
st.header("Metrics Overview")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("P95 Latency (s)", f"{metrics.get_p95_latency():.2f}")
with col2:
    st.metric("Cache Hit Rate", f"{metrics.get_cache_hit_rate():.2%}")
with col3:
    st.metric("Clarification Rate", f"{metrics.get_clarification_rate():.2%}")
with col4:
    st.metric("Total Queries", metrics.total_queries)
with col5:
    st.metric("PII Redactions", metrics.pii_redactions)

st.subheader("Model Costs")
for model, cost in metrics.model_costs.items():
    st.write(f"{model}: ${cost:.4f}")

# Top Questions
st.header("Top Questions")
top_questions = metrics.get_top_questions(10)
for question, count in top_questions:
    st.write(f"{count}: {question}")

# Document Management
st.header("Document Management")
uploaded_file = st.file_uploader("Upload a new document (PDF, DOCX, MD)", type=['pdf', 'docx', 'md'])
if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if st.button("Ingest Document"):
        try:
            ingester.ingest_document(temp_path)
            st.success("Document ingested successfully!")
            # Clear cache on document update
            router.clear_cache_on_document_update()
            st.info("Cache invalidated due to document update.")
        except Exception as e:
            st.error(f"Error ingesting document: {e}")
        finally:
            os.remove(temp_path)

# Cache Management
st.header("Cache Management")
if st.button("Invalidate Cache"):
    router.clear_cache_on_document_update()
    st.success("Cache invalidated!")

if st.button("Reset Metrics"):
    metrics.reset()
    st.success("Metrics reset!")
