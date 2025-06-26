from sentence_transformers import SentenceTransformer, util
import os

# Load CodeBERT
model = SentenceTransformer("microsoft/codebert-base")

# Load code files
def load_code_snippets(folder):
    code_chunks = []
    for fname in os.listdir(folder):
        if fname.endswith((".py", ".java", ".c", ".cpp", ".js")):
            with open(os.path.join(folder, fname), "r", encoding="utf-8") as f:
                code = f.read()
                code_chunks.append({"filename": fname, "code": code})
    return code_chunks

# Embed the code
def embed_code(chunks):
    texts = [chunk["code"] for chunk in chunks]
    embeddings = model.encode(texts, convert_to_tensor=True)
    return embeddings

# Semantic search
def search_code(query, code_chunks, code_embeddings):
    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, code_embeddings, top_k=3)[0]
    return [code_chunks[hit["corpus_id"]] for hit in hits]

# Example usage
print(os.listdir("./Files"))
code_chunks = load_code_snippets("./Files")
embeddings = embed_code(code_chunks)

query = "timsort function"
results = search_code(query, code_chunks, embeddings)

for res in results:
    print(f"\n📄 File: {res['filename']}\n---\n{res['code']}")
