from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import langdetect
from constants import translation_lang

def detect_language(word):
    try:
        return langdetect.detect(word) 
    except:
        return "unknown"

def create_prompt(word, language_src, language_dst):
    template = ChatPromptTemplate.from_template("""
You are a linguist AI assistant.

The user has provided the word: "{word}"

The word is from the language: "{language_src}"

Please provide:
1. The meaning, definition, and example in {language_dst}.
2. A direct translation (if available) to {language_dst}.
""")
    return template.format_messages(word=word, language_src=language_src, language_dst=language_dst)

def handle_user_query(user_input, language_src, language_dst):
    
    if language_src:
        language_src = language_src
    else:
        language_src = detect_language(user_input)


    language_src = translation_lang.get(language_src, {}).get('verbose', '')
    language_dst = translation_lang.get(language_dst, {}).get('verbose', 'Indonesian')
    prompt = create_prompt(user_input, 
                           language_src, 
                           language_dst)
    return prompt