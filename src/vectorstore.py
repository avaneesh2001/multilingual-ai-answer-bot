from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorStore:
    def __init__(self):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = Chroma(embedding_function=embeddings)

    def _chunk_text(self,text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_text(text)
        return chunks
    
    def chunk_docs(self,docs):
        chunks = [self._chunk_text(text["snippet"]) for text in docs]
        return chunks

    def create_vector_store(self, docs):
        chunks = self.chunk_docs(docs)

        for chunks_list, doc in zip(chunks, docs):
            if not chunks_list:
                continue

            self.db.add_texts(
                texts=chunks_list,
                metadatas=[
                    {
                        "url": doc.get("url", ""),
                        "title": doc.get("title", "")
                    }
                    for _ in chunks_list
                ]
            )

    def retrieval(self, query, k=5):
        docs = self.db.similarity_search_with_score(query, k=20)
        texts = [(score, d.page_content, d.metadata) for d, score in docs]
        # Sort by score (lower = better)
        texts = sorted(texts, key=lambda x: x[0])
        texts_top_10 = [(text, metadata) for score, text, metadata in texts[:10]]
        return texts_top_10
    

