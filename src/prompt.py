from langchain_core.prompts import PromptTemplate

TRANSLATE_PROMPT = """
The Following text is in {lang}.
Translate it to English: {query}.
Provide only the translated text without any additional commentary.
"""

def tanslate_prompt(lang,query):
    prompt = PromptTemplate(
        input_variables=["lang", "query"],
        template=TRANSLATE_PROMPT,
    )
    return prompt.format(lang=lang,query=query)


WIKI_KEY_WORD_PROMPT = """
Extract {count} search terms from the following text that can be used for a Wikipedia search: {query}. 
Provide only the keywords separated by commas.
"""

def wiki_key_word_prompt(count,query):
    prompt = PromptTemplate(
        input_variables=["count","query"],
        template=WIKI_KEY_WORD_PROMPT,
    )
    return prompt.format(count=count,query=query)

SUMMARY_PROMPT = """
Summarize the context below in about 80-100 words so it helps answer the question.
Focus only on relevant information.
Do not add anything not present in the text.

Question:
{query}

Context:
{context}
"""


def summary_prompt(context,query):
    prompt = PromptTemplate(
        input_variables=["context","query"],
        template=SUMMARY_PROMPT,
    )
    return prompt.format(context=context,query=query)

ANSWER_PROMPT = """
You are a helpful assistant for a question answering system.

Answer the question using ONLY the provided context.
If the answer is not in the context, respond with "Not found".

Guidelines:
- Be clear and concise
- Do not hallucinate
- Do not add external knowledge
- Prefer short, factual answers

Context:
{context}

Question:
{query}

Answer:
"""


def answer_prompt(context,query):
    prompt = PromptTemplate(
        input_variables=["context","query"],
        template=ANSWER_PROMPT,
    )
    return prompt.format(context=context,query=query)

RETRANSLATE_PROMPT = """
The following answer is based on a context that was extracted from Wikipedia summaries. The original question was in {language} and the answer is in English. Translate the answer back to {language} without adding any commentary. If the answer is "Not found", translate that as well.
Answer: {answer}
"""

def retanslate_prompt(language,answer):
    safe_answer = answer.replace("{", "{{").replace("}", "}}")
    return f"""
The following answer is based on a context that was extracted from Wikipedia summaries.
The original question was in {language} and the answer is in English.
Translate the answer back to {language} without adding any commentary.
If the answer is "Not found", translate that as well.

Answer:
{safe_answer}
"""