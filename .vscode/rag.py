from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama


# ==========================================
# 1. Load PDF
# ==========================================

pdf_path = "documents/AI.pdf"

loader = PyPDFLoader(pdf_path)
documents = loader.load()

print("PDF loaded successfully.")
print("Number of pages:", len(documents))


# ==========================================
# 2. Split documents into chunks
# ==========================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


# ==========================================
# 3. Create Embeddings
# ==========================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ==========================================
# 4. Create Vector Database
# ==========================================

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("Vector database created.")


# ==========================================
# 5. Load Ollama LLM
# ==========================================

llm = ChatOllama(
    model="gemma3:1b"
)

print("LLM loaded.")


# ==========================================
# 6. Question-Answer Loop
# ==========================================

while True:

    question = input(
        "\nYour question (type 'exit' to quit): "
    )

    # Exit program
    if question.lower() == "exit":
        print("Goodbye!")
        break


    # ======================================
    # 7. Retrieve relevant chunks
    # ======================================

    results = vectorstore.similarity_search(
        question,
        k=5
    )


    # ======================================
    # 8. Create context
    # ======================================

    context = "\n\n".join(
        result.page_content
        for result in results
    )


    # ======================================
    # 9. Create prompt
    # ======================================

    prompt = f"""
You are a helpful question-answering assistant.

Answer the question using ONLY the information
provided in the context.

If the answer cannot be found in the context,
say exactly:

"I don't know based on the provided document."

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""


    # ======================================
    # 10. Generate answer
    # ======================================

    response = llm.invoke(prompt)


    # ======================================
    # 11. Display answer
    # ======================================

    print("\n==============================")
    print("Answer:")
    print("==============================")

    print(response.content)


    # ======================================
    # 12. Display sources
    # ======================================

    print("\n==============================")
    print("Sources:")
    print("==============================")

    for i, result in enumerate(results, start=1):

        page = result.metadata.get(
            "page",
            "Unknown"
        )

        if page != "Unknown":
            page = page + 1

        print(f"\nSource {i} - Page {page}")

        print(
            result.page_content[:300]
        )