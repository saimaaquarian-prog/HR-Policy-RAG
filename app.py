import streamlit as st
import fitz
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from groq import Groq


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="📋",
    layout="wide"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666;
        margin-bottom: 25px;
    }

    .answer-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f5f5f5;
        border: 1px solid #ddd;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">📋 HR Policy Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions about your HR policy document using Retrieval-Augmented Generation (RAG).'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# LOAD EMBEDDING MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    return model


embedding_model = load_embedding_model()


# ---------------------------------------------------------
# PDF TEXT EXTRACTION
# ---------------------------------------------------------

def extract_pdf_text(pdf_file):

    pdf_document = fitz.open(
        stream=pdf_file.read(),
        filetype="pdf"
    )

    pages = []

    for page_number, page in enumerate(pdf_document):

        text = page.get_text("text")

        if text.strip():

            pages.append(
                {
                    "page": page_number + 1,
                    "text": text
                }
            )

    pdf_document.close()

    return pages


# ---------------------------------------------------------
# TEXT CHUNKING
# ---------------------------------------------------------

def create_chunks(pages, chunk_size=800, overlap=150):

    chunks = []

    for page_data in pages:

        text = page_data["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end]

            if chunk_text.strip():

                chunks.append(
                    {
                        "text": chunk_text.strip(),
                        "page": page_data["page"]
                    }
                )

            start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------
# CREATE FAISS VECTOR DATABASE
# ---------------------------------------------------------

def create_vector_database(chunks):

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embeddings = embeddings.astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ---------------------------------------------------------
# RETRIEVE RELEVANT CHUNKS
# ---------------------------------------------------------

def retrieve_documents(question, index, chunks, top_k=4):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = question_embedding.astype("float32")

    scores, indices = index.search(
        question_embedding,
        min(top_k, len(chunks))
    )

    retrieved_chunks = []

    for score, index_number in zip(scores[0], indices[0]):

        if index_number == -1:
            continue

        retrieved_chunks.append(
            {
                "text": chunks[index_number]["text"],
                "page": chunks[index_number]["page"],
                "score": float(score)
            }
        )

    return retrieved_chunks


# ---------------------------------------------------------
# GROQ RESPONSE GENERATION
# ---------------------------------------------------------

def generate_answer(question, retrieved_chunks, api_key):

    client = Groq(
        api_key=api_key
    )

    context_parts = []

    for item in retrieved_chunks:

        context_parts.append(
            f"[Page {item['page']}]\n{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are an HR Policy Assistant.

Your job is to answer questions ONLY using the provided HR policy context.

Rules:

1. Use only information from the provided context.
2. Do not invent HR policies.
3. Do not assume information that is not present.
4. If the answer is not available in the policy, clearly say:
   "I could not find this information in the uploaded HR policy."
5. Keep answers clear and beginner-friendly.
6. When possible, mention the relevant policy page.
7. Do not provide legal advice.
8. Do not change or reinterpret the company's policy.
9. If the policy is ambiguous, say that it is ambiguous and explain what the document actually states.
"""

    user_prompt = f"""
HR POLICY CONTEXT:

{context}

EMPLOYEE QUESTION:

{question}

Please answer the employee's question based only on the HR policy context.
"""

    response = client.chat.completions.create(

        model="openai/gpt-oss-20b",

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0.1,

        max_tokens=700
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("⚙️ HR Policy Setup")

    uploaded_file = st.file_uploader(
        "Upload HR Policy PDF",
        type=["pdf"]
    )

    st.markdown("---")

    st.info(
        "Upload a company HR policy PDF and then ask questions about "
        "leave, attendance, benefits, working hours, conduct, and other "
        "policies contained in the document."
    )


# ---------------------------------------------------------
# API KEY
# ---------------------------------------------------------

try:

    groq_api_key = st.secrets["GROQ_API_KEY"]

except Exception:

    groq_api_key = None


# ---------------------------------------------------------
# PROCESS PDF
# ---------------------------------------------------------

if uploaded_file:

    if not groq_api_key:

        st.error(
            "GROQ_API_KEY is not configured. "
            "Add it in Streamlit Cloud Secrets."
        )

        st.stop()

    file_identifier = (
        uploaded_file.name,
        uploaded_file.size
    )

    if (
        "processed_file" not in st.session_state
        or st.session_state.processed_file != file_identifier
    ):

        with st.spinner(
            "Processing HR policy document..."
        ):

            try:

                # Extract PDF
                pages = extract_pdf_text(
                    uploaded_file
                )

                if not pages:

                    st.error(
                        "No readable text was found in this PDF."
                    )

                    st.stop()

                # Create chunks
                chunks = create_chunks(
                    pages
                )

                # Create vector database
                vector_index = create_vector_database(
                    chunks
                )

                # Store everything in session
                st.session_state.pages = pages

                st.session_state.chunks = chunks

                st.session_state.vector_index = vector_index

                st.session_state.processed_file = file_identifier

                st.session_state.chat_history = []

            except Exception as error:

                st.error(
                    f"Error while processing PDF: {error}"
                )

                st.stop()

        st.success(
            f"HR policy processed successfully: "
            f"{len(pages)} pages and {len(chunks)} chunks."
        )

    else:

        st.success(
            f"Document ready: "
            f"{len(st.session_state.pages)} pages and "
            f"{len(st.session_state.chunks)} chunks."
        )


# ---------------------------------------------------------
# QUESTION AREA
# ---------------------------------------------------------

if (
    uploaded_file
    and "vector_index" in st.session_state
):

    st.markdown("---")

    st.subheader("💬 Ask the HR Policy")

    question = st.text_input(
        "Enter your question",
        placeholder=(
            "Example: How many annual leaves can an employee take?"
        )
    )

    ask_button = st.button(
        "🔎 Ask Policy",
        type="primary"
    )

    if ask_button:

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching the HR policy..."
            ):

                try:

                    retrieved_chunks = retrieve_documents(
                        question,
                        st.session_state.vector_index,
                        st.session_state.chunks,
                        top_k=4
                    )

                    answer = generate_answer(
                        question,
                        retrieved_chunks,
                        groq_api_key
                    )

                    # Save conversation
                    st.session_state.chat_history.append(
                        {
                            "question": question,
                            "answer": answer
                        }
                    )

                    st.markdown("### 🤖 Answer")

                    st.markdown(
                        f'<div class="answer-box">{answer}</div>',
                        unsafe_allow_html=True
                    )

                    st.markdown(
                        "### 📚 Retrieved Policy Sections"
                    )

                    for i, item in enumerate(
                        retrieved_chunks,
                        start=1
                    ):

                        with st.expander(
                            f"Source {i} — Page {item['page']}"
                        ):

                            st.write(
                                item["text"]
                            )

                            st.caption(
                                f"Similarity score: "
                                f"{item['score']:.3f}"
                            )

                except Exception as error:

                    st.error(
                        f"Something went wrong: {error}"
                    )


# ---------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------

if (
    "chat_history" in st.session_state
    and st.session_state.chat_history
):

    st.markdown("---")

    st.subheader("🕘 Previous Questions")

    for chat in reversed(
        st.session_state.chat_history
    ):

        with st.expander(
            chat["question"]
        ):

            st.write(
                chat["answer"]
            )


# ---------------------------------------------------------
# INITIAL INSTRUCTIONS
# ---------------------------------------------------------

if not uploaded_file:

    st.markdown("---")

    st.subheader("🚀 How to Use")

    st.markdown(
        """
        **Step 1:** Upload an HR Policy PDF from the sidebar.

        **Step 2:** The application extracts the PDF text.

        **Step 3:** The text is divided into smaller chunks.

        **Step 4:** Sentence Transformer creates embeddings.

        **Step 5:** FAISS stores the embeddings.

        **Step 6:** Ask a question about the policy.

        **Step 7:** The application retrieves relevant policy sections.

        **Step 8:** Groq `openai/gpt-oss-20b` generates the answer.

        **Important:** Answers are grounded in the uploaded HR policy.
        """
    )
