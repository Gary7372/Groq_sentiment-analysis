from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
import re
load_dotenv()
from langchain_core.runnables import Runnable
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="deepseek-r1-distill-llama-70b"
)

summary_prompt = PromptTemplate(
    input_variables=["transcript"],
    template="""
You are a helpful assistant that summarizes customer support transcripts.

Summarize the following customer call in 2–3 sentences:

{transcript}
"""
)

summary_chain: Runnable = summary_prompt | llm
def clean_summary_output(raw_output: str) -> str:
    # Removes XML-like tags and extra whitespace
    cleaned = re.sub(r"<.*?>.*?</.*?>", "", raw_output, flags=re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned