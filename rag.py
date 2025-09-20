from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class FileRAG:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.files = []

    def build_index(self, file_dict):
        """파일 단위로 임베딩"""
        self.files = list(file_dict.keys())
        texts = list(file_dict.values())

        embeddings = self.model.encode(texts)
        embeddings = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query, top_k=3):
        """쿼리에 맞는 파일 반환"""
        q_emb = self.model.encode([query]).astype("float32")
        scores, idxs = self.index.search(q_emb, top_k)

        results = []
        for i in idxs[0]:
            fname = self.files[i]
            results.append(fname)
        return results
