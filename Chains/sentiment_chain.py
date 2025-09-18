from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
import re
from langchain_core.runnables import Runnable
load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="deepseek-r1-distill-llama-70b"
)

sentiment_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="""
You are an assistant that determines customer sentiment in transcripts.

Classify the sentiment of the following conversation as one of:
Positive, Neutral, or Negative. Respond with one word only.

Transcript:
{transcript}
"""
)

sentiment_chain: Runnable = sentiment_prompt | llm

def clean_sentiment_output(raw_output: str) -> str:
    cleaned = re.sub(r"<.*?>.*?</.*?>", "", raw_output, flags=re.DOTALL)
    match = re.search(r"\b(positive|neutral|negative)\b", cleaned, re.IGNORECASE)
    return match.group(1).capitalize() if match else "Unknown"


