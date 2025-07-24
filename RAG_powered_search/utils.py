from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
import langdetect

def detect_language(word):
    # Fallback to detection (you could use fastText or langdetect)
    try:
        return langdetect.detect(word)  # may return 'jv' for Javanese
    except:
        return "unknown"

def create_prompt(word, language):
    template = ChatPromptTemplate.from_template("""
You are a linguist AI assistant.

The user has provided the word: "{word}"

The word is from the language: "{language}"

Please provide:
1. The meaning, definition, and example in Bahasa Indonesia.
2. A direct translation (if available).
""")
    return template.format_messages(word=word, language=language)

def handle_user_query(lang_hint, user_input):
    word = user_input.split()[0]  
    
    if lang_hint:
        language = lang_hint
    else:
        language = detect_language(word)

    prompt = create_prompt(word, language)
    return prompt