from src.llm import LLMCall
from src.prompt import summary_prompt, answer_prompt

class SummarizeAnswer:
    def __init__ (self):
        self.llm = LLMCall()

    def summarizer(self,top_5_reranked, query):
        context = "\n\n".join([f"Summary {i+1}: {summary}" for i, summary in enumerate(top_5_reranked) if summary])
        prompt = summary_prompt(context=context,query=query)
        context_summary = self.llm.call_llm(prompt)
        return context_summary
    
    def answer(self,summary,query):
        prompt = answer_prompt(summary,query)
        response = self.llm.call_llm(prompt)
        return response
    