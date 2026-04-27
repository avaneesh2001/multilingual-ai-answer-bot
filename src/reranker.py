from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self,model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self,query,docs):
        pairs = [[query, doc] for doc in docs]
        scores = self.model.predict(pairs)
        scored_docs = list(zip(scores, docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        reranked_docs = [doc for score, doc in scored_docs]
        return reranked_docs[:5]