from langdetect import detect as lang_detect
from src.config import GLOBAL_LANGS, INDIAN_LANGS
from src.llm import LLMCall
from src.prompt import tanslate_prompt, retanslate_prompt

class LanguageDetector:
    def __init__(self):
        self.langs = {**GLOBAL_LANGS, **INDIAN_LANGS}
        self.llm = LLMCall()

    def detect_language(self, text):
        try:
            if len(text) < 20:
                return None  
            else:
                lang = lang_detect(text)
                if lang == "ne":
                    lang = "hi"
            return self.langs.get(lang,None)
        except Exception as e:
            print(f"Error detecting language: {e}")
            return None
        
    def translate(self, text, language):
        if language is not None and language != "English":
            prompt = tanslate_prompt(language,text)
            translated_resp = self.llm.call_llm(prompt)
            return translated_resp
        return text
    
    def retranslate(self, text, language):
        if language is not None and language != "English":
            prompt = retanslate_prompt(language,text)
            translated_resp = self.llm.call_llm(prompt)
            return translated_resp
        return text
    
