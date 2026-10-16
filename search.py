from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Load existing vector database
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)


# User question
question = input("Your question: ")


# Search
results = vectorstore.similarity_search(
    question,
    k=3
)


# Show results
print("\nRelevant information:\n")

for i, result in enumerate(results):
    print(f"--- Result {i+1} ---")
    print(result.page_content)
    print()