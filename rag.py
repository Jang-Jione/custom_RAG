from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz
import os, json


class FileRAG:
    def __init__(self, model_name="all-MiniLM-L6-v2", store_dir="rag_store"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.metadata = []  # [{file_name, file_contents}]
        self.store_dir = store_dir
        os.makedirs(self.store_dir, exist_ok=True)

        self._load_if_exists()


    def pdf_to_text(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            return text.strip()
        except Exception as e:
            print(f"[WARN] PDF error: {pdf_path} ({e})")
            return ""


    def build_index(self, file_dict: dict):
        texts, file_names = [], []
        self.metadata = []

        for fname, content in file_dict.items():
            try:
                if fname.lower().endswith(".pdf"):
                    text = self.pdf_to_text(content)
                else:
                    text = str(content)
            except Exception as e:
                print(f"[WARN] {fname} fail: {e}")
                continue

            if text.strip():
                texts.append(text)
                file_names.append(fname)
                self.metadata.append({
                    "file_name": fname,
                    "file_contents": text
                })

        if not texts:
            raise RuntimeError("there is no valid text to build index.")

        embeddings = self.model.encode(texts)
        embeddings = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

        self._save_index()
        self._save_metadata()

        print(f"[BUILD DONE] {len(texts)} files indexed.")


    def search(self, query, top_k=6):
        if self.index is None:
            raise RuntimeError("FAISS must be initialized before searching.")

        q_emb = self.model.encode([query]).astype("float32")
        scores, idxs = self.index.search(q_emb, top_k)

        results = []
        for i in idxs[0]:
            if i < len(self.metadata):
                results.append({
                    "file_name": self.metadata[i]["file_name"],
                    "file_contents": self.metadata[i]["file_contents"]
                })
        return results

  
    def _save_index(self):
        faiss.write_index(self.index, os.path.join(self.store_dir, "rag_index.faiss"))

    def _save_metadata(self):
        with open(os.path.join(self.store_dir, "rag_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def _load_if_exists(self):
        idx_path = os.path.join(self.store_dir, "rag_index.faiss")
        meta_path = os.path.join(self.store_dir, "rag_metadata.json")

        if os.path.exists(idx_path) and os.path.exists(meta_path):
            try:
                self.index = faiss.read_index(idx_path)
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                print(f"[LOAD] {len(self.metadata)} are loaded.")
            except Exception as e:
                print(f"[WARN] RAG fail: {e}")
