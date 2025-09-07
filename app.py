import streamlit as st
import requests

st.set_page_config(page_title="Document Q&A System", layout="wide")
st.title("📄 Document Q&A System Using RAG")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    if st.button("Process Document"):
        files = {"file": (uploaded_file.name, uploaded_file.getbuffer(), uploaded_file.type)}
        with st.spinner("Uploading and processing document..."):
            try:
                response = requests.post("http://localhost:8000/upload-pdf/", files=files)
                if response.status_code == 200:
                    st.success("Document processed successfully! You can now ask questions.")
                else:
                    st.error(f"Failed to process document: {response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Error connecting to backend: {e}")

# Text input for questions
question = st.text_input("Ask a question about the document:")

if question:
    with st.spinner("Fetching answer..."):
        try:
            resp = requests.post("http://localhost:8000/ask/", json={"question": question})
            if resp.status_code == 200:
                data = resp.json()
                if "answer" in data:
                    st.markdown(f"**Answer:** {data['answer']}")
                else:
                    st.error(data.get("error", "No answer returned."))
            else:
                st.error(f"Error from backend: {resp.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to backend: {e}")
