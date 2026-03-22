import streamlit as st
from pypdf import PdfReader
import ollama

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

# -------- CONFIG --------
st.set_page_config(page_title="EduBridge", layout="wide")

# -------- DESIGN CSS --------
st.markdown("""
<style>

body {
    background-color: #f5f7fb;
}

h1 {
    color: #1f4e79;
    text-align: center;
}

.stButton>button {
    background-color: #1f4e79;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    border: none;
}

.stButton>button:hover {
    background-color: #163a5f;
}

.stTextInput>div>div>input {
    border-radius: 10px;
    padding: 10px;
}

.stSelectbox>div>div {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# -------- HEADER --------
st.markdown("<h1>🎓 EduBridge</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Your AI-powered academic assistant</p>", unsafe_allow_html=True)
st.markdown("---")

# -------- SIDEBAR --------
st.sidebar.markdown("## 🎓 EduBridge")
st.sidebar.markdown("### Navigation")
st.sidebar.info("Upload a PDF and explore AI features 🚀")

page = st.sidebar.selectbox(
    "",
    ["Upload", "Chatbot", "Summary", "Questions", "Translate"]
)

# -------- UPLOAD --------
uploaded_file = st.file_uploader("Upload your course PDF")

if uploaded_file:

    # -------- READ PDF --------
    pdf = PdfReader(uploaded_file)
    text = ""

    for page_pdf in pdf.pages:
        if page_pdf.extract_text():
            text += page_pdf.extract_text()

    st.success("📄 Course loaded successfully!")

    # -------- SPLIT --------
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_text(text)

    # -------- EMBEDDINGS --------
    embeddings = FakeEmbeddings(size=384)
    vectorstore = FAISS.from_texts(chunks, embeddings)

    st.success("🧠 AI ready!")

    # -------- PAGE: UPLOAD --------
    if page == "Upload":

        st.subheader("📌 Important Concepts")

        if st.button("Highlight Important Parts"):

            response = ollama.chat(
                model="llama3",
                messages=[
                    {
                        "role": "user",
                        "content": f"""
                        Extract the most important concepts.
                        Give bullet points.

                        {text[:3000]}
                        """
                    }
                ]
            )

            st.success(response["message"]["content"])

    # -------- CHATBOT --------
    elif page == "Chatbot":

        st.subheader("💬 Ask questions about your course")

        col1, col2 = st.columns([2, 1])

        with col1:
            question = st.text_input("Your question")

            if question:
                docs = vectorstore.similarity_search(question, k=3)
                context = " ".join([doc.page_content for doc in docs])

                response = ollama.chat(
                    model="llama3",
                    messages=[
                        {"role": "system", "content": "Answer only using context"},
                        {"role": "user", "content": f"Context: {context}\nQuestion: {question}"}
                    ]
                )

                st.success(response["message"]["content"])

        with col2:
            st.info("💡 Tips:")
            st.write("- Ask clear questions")
            st.write("- Example: Explain this concept")

    # -------- SUMMARY --------
    elif page == "Summary":

        st.subheader("📝 Course Summary")

        col1, col2 = st.columns([2, 1])

        with col1:
            if st.button("Generate Summary"):
                response = ollama.chat(
                    model="llama3",
                    messages=[{"role": "user", "content": f"Summarize:\n{text[:3000]}"}
                    ]
                )
                st.success(response["message"]["content"])

        with col2:
            st.info("📌 Simplified version of the course")

    # -------- QUESTIONS --------
    elif page == "Questions":

        st.subheader("❓ Exam Questions")

        col1, col2 = st.columns([2, 1])

        with col1:
            if st.button("Generate Questions"):
                response = ollama.chat(
                    model="llama3",
                    messages=[
                        {"role": "user", "content": f"Generate exam questions:\n{text[:3000]}"}
                    ]
                )
                st.success(response["message"]["content"])

        with col2:
            st.info("🎯 Practice like a real exam")

    # -------- TRANSLATE --------
    elif page == "Translate":

        st.subheader("🌍 Translate Course")

        col1, col2 = st.columns([2, 1])

        with col1:
            lang = st.selectbox("Choose language", ["French", "English", "Arabic"])

            if st.button("Translate"):
                response = ollama.chat(
                    model="llama3",
                    messages=[
                        {"role": "user", "content": f"Translate to {lang}:\n{text[:3000]}"}
                    ]
                )
                st.success(response["message"]["content"])

        with col2:
            st.info("🌎 Helps international students")
            
else:
    st.info("⬆️ Upload a PDF to start")