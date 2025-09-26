from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz  # PyMuPDF


class FileRAG:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        # print(f"[INIT] SentenceTransformer 모델 로드: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.files = []
        self.docs = []

    def pdf_to_text(self, pdf_path: str) -> str:
        """PDF → 텍스트 추출"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text()
                text += page_text
            # print(f"    [PDF 추출 완료] {pdf_path} (총 {len(doc)}페이지, {len(text)}자)")
            return text.strip()
        except Exception as e:
            # print(f"    [PDF 추출 실패] {pdf_path} → {e}")
            raise

    def build_index(self, file_dict: dict):
        texts = []
        self.files = []

        total = len(file_dict)
        print(f"[BUILD INDEX] 총 {total}개 파일 인덱싱 시작")

        for count, (fname, content) in enumerate(file_dict.items(), start=1):
            # print(f"[{count}/{total}] 업로드 중: {fname}")
            text = ""

            try:
                if isinstance(content, str) and content.lower().endswith(".pdf"):
                    # PDF 파일 경로라면 텍스트 추출
                    text = self.pdf_to_text(content)
                else:
                    # 문자열 그대로 사용
                    text = str(content)
                    # print(f"    [TEXT 처리 완료] {fname} (길이 {len(text)}자)")
            except Exception as e:
                # print(f"    [경고] {fname} 처리 실패: {e}")
                text = ""

            if text.strip():
                texts.append(text)
                self.files.append(fname)
                self.docs.append((fname, text))
                print(f"[{count}/{total}]")
                # print(f"[{count}/{total}] 업로드 완료: {fname} (길이 {len(text)}자)")
            else:
                print(f"[{count}/{total}]")
                # print(f"[{count}/{total}] 업로드 실패: {fname} (내용 없음)")

        # if not texts:
        #     raise ValueError("인덱싱할 텍스트가 없습니다. file_dict를 확인하세요.")

        # print("[임베딩 생성 시작]")
        embeddings = self.model.encode(texts)
        embeddings = np.array(embeddings).astype("float32")
        # print(f"[임베딩 생성 완료] shape={embeddings.shape}")

        # if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        #     raise ValueError("임베딩 생성 실패: texts 비어있음")

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        # print(f"[FAISS 인덱스 구축 완료] 벡터 개수={self.index.ntotal}")

    def search(self, query, top_k=3):
        if self.index is None:
            raise RuntimeError("FAISS 인덱스가 초기화되지 않았습니다. build_index() 실행 여부를 확인하세요.")

        # print(f"[SEARCH] 쿼리=\"{query}\" (top_k={top_k})")
        q_emb = self.model.encode([query]).astype("float32")
        scores, idxs = self.index.search(q_emb, top_k)

        results = []
        for rank, (i, score) in enumerate(zip(idxs[0], scores[0]), start=1):
            fname = self.files[i]
            # print(f"   {rank}. {fname} (score={score:.4f})")
            results.append(fname)

        return results
