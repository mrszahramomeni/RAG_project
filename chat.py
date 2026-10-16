from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


# =========================
# 1. Load Embedding Model
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# 2. Load Vector Database
# =========================

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# =========================
# 3. Create Retriever
# =========================

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)


# =========================
# 4. Load LLM
# =========================

llm = ChatOllama(
    model="gemma3:1b"
)


# =========================
# 5. Get Question
# =========================

question = input("\nYour question: ")


# =========================
# 6. Retrieve Relevant Documents
# =========================

documents = retriever.invoke(question)


# =========================
# 7. Show Retrieved Chunks
# =========================

print("\n==============================")
print("Retrieved Documents")
print("==============================")

for i, doc in enumerate(documents):
    print(f"\n--- Document {i + 1} ---")
    print(doc.page_content)


# =========================
# 8. Create Context
# =========================

context = "\n\n".join(
    doc.page_content
    for doc in documents
)


# =========================
# 9. Create Prompt
# =========================

prompt = f"""
You are a helpful assistant answering questions about a PDF document.

Answer the question using ONLY the information provided in the context.

If the answer is not available in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""


# =========================
# 10. Generate Answer
# =========================

response = llm.invoke(prompt)


# =========================
# 11. Show Final Answer
# =========================

print("\n==============================")
print("Answer")
print("==============================")

print(response.content)