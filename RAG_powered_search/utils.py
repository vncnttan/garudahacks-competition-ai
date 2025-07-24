from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import langdetect

language_dict = {
    "jv": "Javanese",
    "id": "Indonesian",
}

def detect_language(word):
    try:
        return langdetect.detect(word) 
    except:
        return "unknown"

def create_prompt(word, language_src):
    template = ChatPromptTemplate.from_template("""
You are a linguist AI assistant.

The user has provided the word: "{word}"

The word is from the language: "{language_src}"

Please provide:
1. The meaning, definition, and example in {language_dst}.
2. A direct translation (if available) to {language_dst}.
""")
    return template.format_messages(word=word, language_src=language_src)

def handle_user_query(lang_hint, user_input):
    
    if lang_hint:
        language_src = lang_hint
    else:
        language_src = detect_language(user_input)

    prompt = create_prompt(user_input, language_dict.get(language_src, "Indonesian"))
    return prompt