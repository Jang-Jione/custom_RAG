from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz  # PyMuPDF


class FileRAG:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.files = []
        self.docs = []

    def pdf_to_text(self, pdf_path: str) -> str:
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text()
                text += page_text
            return text.strip()
        except Exception as e:
            raise

    def build_index(self, file_dict: dict):
        texts = []
        self.files = []

        total = len(file_dict)
        print(f"[BUILD INDEX] 총 {total}개 파일 인덱싱 시작")

        for count, (fname, content) in enumerate(file_dict.items(), start=1):
            text = ""

            try:
                if isinstance(content, str) and content.lower().endswith(".pdf"):
                    text = self.pdf_to_text(content)
                else:
                    text = str(content)
            except Exception as e:
                text = ""

            if text.strip():
                texts.append(text)
                self.files.append(fname)
                self.docs.append((fname, text))
                print(f"[{count}/{total}]")
            else:
                print(f"[{count}/{total}]")

        embeddings = self.model.encode(texts)
        embeddings = np.array(embeddings).astype("float32")
   
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query, top_k=3):
        if self.index is None:
            raise RuntimeError("FAISS 인덱스가 초기화되지 않았습니다. build_index() 실행 여부를 확인하세요.")

        q_emb = self.model.encode([query]).astype("float32")
        scores, idxs = self.index.search(q_emb, top_k)

        results = []
        for rank, (i, score) in enumerate(zip(idxs[0], scores[0]), start=1):
            fname = self.files[i]
            results.append(fname)

        return results
