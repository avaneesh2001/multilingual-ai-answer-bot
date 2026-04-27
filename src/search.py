from src.config import WEB_SEARCH_LIMIT, WIKI_SEARCH_TERM_COUNT
from src.llm import LLMCall
from src.prompt import wiki_key_word_prompt
import wikipedia
from ddgs import DDGS

class Search():
    def __init__(self):
        self.llm =  LLMCall()

    def get_wiki_search_terms(self,query):
        prompt = wiki_key_word_prompt(WIKI_SEARCH_TERM_COUNT,query)
        wiki_search_terms = self.llm.call_llm(prompt)
        
        terms = [term.strip() for term in wiki_search_terms.split(",")]
        return terms
    
    def _get_wiki_summary(self,query):
        wikipedia.set_lang("en")

        try:
            results = wikipedia.search(query)
            print("SEARCH:", query, results)

            if not results:
                return None

            for title in results[:3]:
                try:
                    print("TRYING:", title)

                    page = wikipedia.page(title, auto_suggest=False)
                    
                    return {
                    "title":page.title,
                    "url": page.url,
                    "snippet": page.content
                }
                
                except wikipedia.exceptions.DisambiguationError as e:
                    try:
                        page = wikipedia.page(e.options[0], auto_suggest=False)
                        return {
                    "title":page.title,
                    "url": page.url,
                    "snippet": page.content
                }
                    except:
                        continue

                except wikipedia.exceptions.PageError:
                    continue

            return {
                    "title":None,
                    "url": None,
                    "snippet": None
                }

        except Exception as e:
            print(f"Wiki error: {e}")
        return {
                    "title":None,
                    "url": None,
                    "snippet": None
                }
    
    def get_wiki_seach(self,query):
        wiki_summaries = []
        terms = self.get_wiki_search_terms(query)
        for term in terms:
            try:
                summary = self._get_wiki_summary(term)
                if summary["snippet"] is None:
                    continue

                wiki_summaries.append(summary)
            except Exception as e:
                print(f"Error fetching summary for {term}: {e}")
        return wiki_summaries

    def web_search(self,query, k=WEB_SEARCH_LIMIT):
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=k, region="us-en",):
                if r.get("body") is None:
                    continue
                results.append({
                    "title": r.get("title"),
                    "url": r.get("href"),
                    "snippet": r.get("body")
                })

        return results
    
    def search(self,query):
        wiki_results = self.get_wiki_seach(query)
        web_resutls = self.web_search(query)
        results = wiki_results + web_resutls
        return results

