import streamlit as st
import pandas as pd
from model import MultiAntibioticPredictor
import ollama
import time
from PIL import Image
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="AmPy - Genome-based Antibiotic Advisor", layout="wide")

# -------------------------------
# Enforce light theme
# -------------------------------
st.markdown(
    """
    <style>
    html, body, .css-18e3th9, .css-1v3fvcr {
        background-color: #ffffff;
        color: #000000;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Header: Logo top-left + centered title
# -------------------------------
header_col1, header_col2, _ = st.columns([1, 4, 1])

with header_col1:
    try:
        logo = Image.open("image.png")
        st.image(logo, width=80)
    except:
        pass

with header_col2:
    st.markdown("<h1 style='text-align:center; font-size:60px; margin-bottom:0;'>AmPy</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align:center; margin-top:0;'>Genome-based Antibiotic Advisor</h3>", unsafe_allow_html=True)

# -------------------------------
# Load ML Predictor & Knowledge Base
# -------------------------------
@st.cache_resource
def load_predictor():
    predictor = MultiAntibioticPredictor.load_models("multi_antibiotic_models.pkl")
    metrics_df = pd.DataFrame(predictor.performance_metrics).T
    top10 = metrics_df.sort_values("auc", ascending=False).head(10)
    top10_antibiotics = top10.index.tolist()
    return predictor, top10_antibiotics

@st.cache_resource
def load_knowledge_base():
    try:
        with open("antibiotic_docs.pkl", "rb") as f:
            docs = pickle.load(f)
        with open("antibiotic_embeddings.pkl", "rb") as f:  # Fixed typo from .pdk to .pkl
            embeddings = pickle.load(f)
        return docs, embeddings
    except Exception as e:
        st.warning(f"Could not load knowledge base: {e}")
        return None, None

def get_relevant_context(query, docs, embeddings, top_k=3):
    """Fast context retrieval using precomputed embeddings"""
    if docs is None or embeddings is None:
        return ""
    
    try:
        # Get query embedding (simplified - you may need to adjust based on your embedding method)
        # For now, we'll do a simple keyword search as fallback
        query_lower = query.lower()
        relevant_docs = []
        
        for i, doc in enumerate(docs):
            if any(word in doc.lower() for word in query_lower.split()[:5]):  # Quick keyword match
                relevant_docs.append(doc)
            if len(relevant_docs) >= top_k:
                break
                
        return " | ".join(relevant_docs[:top_k])
    except:
        return ""

predictor, top10_antibiotics = load_predictor()
docs, embeddings = load_knowledge_base()

# -------------------------------
# FASTA parsers
# -------------------------------
def extract_fasta_sequences(fasta_text: str):
    sequences = {}
    current_header = None
    for line in fasta_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            current_header = line[1:].strip()
            sequences[current_header] = []
        elif current_header:
            sequences[current_header].append(line)
    for header in sequences:
        sequences[header] = "".join(sequences[header])
    return sequences

def concat_fasta_sequences(fasta_text: str):
    sequences = []
    for line in fasta_text.splitlines():
        line = line.strip()
        if not line or line.startswith(">"):
            continue
        sequences.append(line)
    return "".join(sequences)

# -------------------------------
# Resistance prediction
# -------------------------------
def predict_resistance(sequences: dict):
    results = {}
    for header, sequence in sequences.items():
        try:
            kmer_seq = predictor.extract_kmers(sequence)
            X_new = predictor.vectorizer.transform([kmer_seq])
            predictions = {}
            for ab in top10_antibiotics:
                model = predictor.models[ab]
                prob = model.predict_proba(X_new)[0, 1]
                predictions[ab] = {
                    "prediction": "Resistant" if prob > 0.5 else "Susceptible",
                    "probability": float(prob),
                    "confidence": float(max(prob, 1 - prob))
                }
            results[header] = dict(sorted(predictions.items(), key=lambda x: x[1]["probability"], reverse=True))
        except Exception as e:
            results[header] = {"error": str(e)}
    return results

# -------------------------------
# Session state
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "results" not in st.session_state:
    st.session_state.results = None
if "genome_summary" not in st.session_state:
    st.session_state.genome_summary = ""
if "uploaded_fasta_text" not in st.session_state:
    st.session_state.uploaded_fasta_text = None
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

# -------------------------------
# Upload FASTA
# -------------------------------
uploaded_file = st.file_uploader("Upload bacterial genome FASTA", type=["fasta", "fa"])
concat_option = st.checkbox("Concatenate all sequences into one", value=True)

if uploaded_file is not None:
    if st.session_state.uploaded_fasta_text != uploaded_file:
        st.session_state.uploaded_fasta_text = uploaded_file
        st.session_state.prediction_done = False

    if not st.session_state.prediction_done:
        fasta_text = uploaded_file.read().decode("utf-8")

        if concat_option:
            concat_sequence = concat_fasta_sequences(fasta_text)
            fasta_data = {"Concatenated_sequence": concat_sequence}
            st.write("All sequences concatenated into one.")
        else:
            fasta_data = extract_fasta_sequences(fasta_text)
            st.write(f"Found {len(fasta_data)} sequences.")
            st.json(list(fasta_data.keys())[:5])

        # Prediction running indicator
        prediction_placeholder = st.empty()
        prediction_placeholder.info("🔬 Running genome resistance prediction...")
        time.sleep(0.5)

        st.session_state.results = predict_resistance(fasta_data)

        # Summarize results
        summary_lines = []
        for i, (header, preds) in enumerate(st.session_state.results.items()):
            if i >= 5:
                break
            if "error" in preds:
                summary_lines.append(f"{header}: Error - {preds['error']}")
                continue
            resistant = [ab for ab, res in preds.items() if res["prediction"] == "Resistant"]
            susceptible = [ab for ab, res in preds.items() if res["prediction"] == "Susceptible"]
            summary_lines.append(
                f"{header}: Resistant to {', '.join(resistant) if resistant else 'none'}; "
                f"Susceptible to {', '.join(susceptible) if susceptible else 'none'}"
            )
        st.session_state.genome_summary = " | ".join(summary_lines)
        st.session_state.prediction_done = True
        prediction_placeholder.success("✅ Prediction complete. You can now ask questions to AmPy.")

# -------------------------------
# Chat Interface (only after prediction)
# -------------------------------
if st.session_state.prediction_done:
    st.divider()
    st.header("💬 Chat with AmPy about the results")

    # Display conversation
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask a question...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Trim genome summary for LLM
        genome_summary_trimmed = st.session_state.genome_summary
        if len(genome_summary_trimmed) > 1000:
            genome_summary_trimmed = genome_summary_trimmed[:1000] + "..."

        # Get relevant WHO guidelines context
        relevant_context = get_relevant_context(user_input, docs, embeddings)
        context_text = f"\nWHO Guidelines Context: {relevant_context}" if relevant_context else ""

        # System message with concise instruction
        system_message = {
            "role": "system",
            "content": (
                "You are a highly knowledgeable physician assistant specializing in infectious diseases and antibiotics. "
                "Keep your responses concise and to the point, but if you are not provided enough information about the case, ask for more info before suggesting a treatment. Citing WHO guidelines when relevant. "
                f"Genome prediction summary: {genome_summary_trimmed}{context_text}"
            )
        }

        # LLM response with streaming
        with st.chat_message("assistant"):
            placeholder = st.empty()
            
            # Stream the response
            full_response = ""
            try:
                stream = ollama.chat(
                    model="gemma3",
                    messages=[system_message] + st.session_state.messages,
                    stream=True,
                )
                
                for chunk in stream:
                    if chunk['message']['content']:
                        full_response += chunk['message']['content']
                        placeholder.markdown(full_response + "▊")  # Add cursor effect
                        
                # Remove cursor and show final response
                placeholder.markdown(full_response)
                
            except Exception as e:
                placeholder.error(f"Error generating response: {str(e)}")
                full_response = f"Sorry, I encountered an error: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": full_response})