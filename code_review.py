import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ollama import Client as OllamaClient

# ---- Supported file extensions ----
SUPPORTED_EXTENSIONS = [".py", ".java", ".cs", ".sql", ".c"]

# ---- PocketFlow Core ----
class Node:
    def forward(self, inputs: dict) -> dict:
        raise NotImplementedError

class Flow:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add(self, name, node: Node, after=None):
        self.nodes[name] = node
        self.edges[name] = after

    def run(self, input_dict: dict):
        outputs = {}
        for name in self.nodes:
            deps = self.edges[name]
            node_inputs = outputs.get(deps, input_dict)
            outputs[name] = self.nodes[name].forward(node_inputs)
        return outputs[name]

# ---- Code Review Nodes ----
class EmbedCodebase(Node):
    def __init__(self, folder_path, embedding_model=None, db_dir="chroma_db"):
        self.folder_path = folder_path
        self.db_dir = db_dir
        self.embedding_model = embedding_model or HuggingFaceEmbeddings(model_name="microsoft/codebert-base")

    def forward(self, inputs):
        print("🔍 Embedding code files...")
        docs = []

        for root, _, files in os.walk(self.folder_path):
            for file in files:
                if any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    path = os.path.join(root, file)
                    try:
                        with open(path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            docs.append(Document(page_content=content, metadata={"source": path}))
                    except Exception as e:
                        print(f"⚠️ Skipped {path}: {e}")

        if not docs:
            raise ValueError("❌ No valid code documents found to embed.")

        print(f"📄 Loaded {len(docs)} documents.")
        print("🧩 Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
        split_docs = splitter.split_documents(docs)

        print(f"🧩 Created {len(split_docs)} chunks. Embedding and saving to {self.db_dir}...")
        self.db = Chroma.from_documents(split_docs, self.embedding_model, persist_directory=self.db_dir)
        return {"docs": docs, "db": self.db, "query": inputs["query"]}

class RetrieveCode(Node):
    def __init__(self, db_dir="chroma_db", model_name="microsoft/codebert-base"):
        self.db_dir = db_dir
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    def forward(self, inputs: dict) -> dict:
        query = inputs["query"]
        print("📥 Retrieving relevant code chunks...")

        db = Chroma(persist_directory=self.db_dir, embedding_function=self.embedding_model)
        results = db.similarity_search_with_score(query, k=10)

        context_chunks = []
        print(f"\n🔎 Top {len(results)} relevant chunks retrieved for query: '{query}'")
        for i, (doc, score) in enumerate(results):
            print(f"\n--- Chunk {i+1} (Score: {score:.4f}) ---")
            print(f"🔗 Source: {doc.metadata.get('source', 'N/A')}")
            print(doc.page_content[:400])  # Print first 400 chars
            context_chunks.append(doc.page_content)

        context = "\n\n".join(context_chunks)
        return {"query": query, "context": context}


class LLaMAAnswer(Node):
    def __init__(self):
        self.client = OllamaClient(host="http://localhost:11434")

    def forward(self, inputs: dict) -> dict:
        print("💬 Generating answer with LLaMA...")
        prompt = fprompt = f"""
            You are a senior Java developer and software auditor. Given the code context below, explain the user’s question as clearly and technically as possible.

            CODE CONTEXT:
            {inputs['context']}

            QUESTION:
            {inputs['query']}

            ANSWER:
            """

        response = self.client.chat(model="llama3", messages=[
            {"role": "user", "content": prompt}
        ])
        return {"answer": response["message"]["content"]}

# ---- Run the Pipeline ----
if __name__ == "__main__":
    code_folder = "./Files"
    user_question = input("🧠 Ask a question about the code: ")

    print("🔧 Building pipeline...")
    flow = Flow()
    flow.add("embedder", EmbedCodebase(folder_path=code_folder))
    flow.add("retriever", RetrieveCode(), after="embedder")
    flow.add("llm", LLaMAAnswer(), after="retriever")

    print("🧪 Running...")
    result = flow.run({"query": user_question})
    print(f"\n📘 Answer:\n{result['answer']}")
