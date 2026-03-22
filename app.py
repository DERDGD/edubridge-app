import streamlit as st
from pypdf import PdfReader
from openai import OpenAI

from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

# -------- CONFIG --------
st.set_page_config(page_title="EduBridge", layout="wide")

# -------- API KEY --------
client = OpenAI(api_key=st.secrets["Osk-proj-LdQir3rUwCSzXxctPx0pH_bgUdpyR28cr3jw5sKyk8zs8Vzpjkm3funBWP1lObHOvEhx_F9DmJT3BlbkFJywKFgUbH7_TT8tBB5xHN0J1w3t8FLsjzkjswepMpoKZT_X8sd3fnnBhm6ePw6QJiexp_29BJAA"])

# -------- DESIGN --------
st.markdown("""
<style>
body {background-color: #f5f7fb;}
h1 {color: #1f4e79; text-align: center;}
.stButton>button {
    background-color: #1f4e79;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
.stButton>button:hover {background-color: #163a5f;}
</style>
""", unsafe_allow_html=True)

# -------- HEADER --------
st.markdown("<h1>🎓 EduBridge</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered academic assistant</p>", unsafe_allow_html=True)
st.markdown("---")

# -------- SIDEBAR --------
st.sidebar.markdown("## 🎓 EduBridge")
st.sidebar.markdown("### Navigation")
st.sidebar.info("Upload a PDF and explore AI features 🚀")

page = st.sidebar.selectbox(
    "",
    ["Upload", "Chatbot", "Summary", "Questions", "Translate"]
)

# -------- FUNCTION AI --------
def ask_ai(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

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

    # -------- VECTOR STORE --------
    embeddings = FakeEmbeddings(size=384)
    vectorstore = FAISS.from_texts(chunks, embeddings)

    st.success("🧠 AI ready!")

    # -------- PAGE: UPLOAD --------
    if page == "Upload":

        st.subheader("📌 Important Concepts")

        if st.button("Highlight Important Parts"):
            result = ask_ai(f"Extract key concepts:\n{text[:3000]}")
            st.success(result)

    # -------- CHATBOT --------
    elif page == "Chatbot":

        st.subheader("💬 Ask questions about your course")

        col1, col2 = st.columns([2, 1])

        with col1:
            question = st.text_input("Your question")

            if question:
                docs = vectorstore.similarity_search(question, k=3)
                context = " ".join([doc.page_content for doc in docs])

                answer = ask_ai(f"Context:\n{context}\n\nQuestion: {question}")
                st.success(answer)

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
                result = ask_ai(f"Summarize:\n{text[:3000]}")
                st.success(result)

        with col2:
            st.info("📌 Simplified version of the course")

    # -------- QUESTIONS --------
    elif page == "Questions":

        st.subheader("❓ Exam Questions")

        col1, col2 = st.columns([2, 1])

        with col1:
            if st.button("Generate Questions"):
                result = ask_ai(f"Generate exam questions:\n{text[:3000]}")
                st.success(result)

        with col2:
            st.info("🎯 Practice like a real exam")

    # -------- TRANSLATE --------
    elif page == "Translate":st.subheader("🌍 Translate Course")

    col1, col2 = st.columns([2, 1])

    with col1:
            lang = st.selectbox("Choose language", ["French", "English", "Arabic"])

            if st.button("Translate"):
                result = ask_ai(f"Translate to {lang}:\n{text[:3000]}")
                st.success(result)

    with col2:
            st.info("🌎 Helps international students")

else:
    st.info("⬆️ Upload a PDF to start")
