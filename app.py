import streamlit as st
from Chains.summary_chain import summary_chain , clean_summary_output
from Chains.sentiment_chain import sentiment_chain ,clean_sentiment_output

from Utils.csv_utils import save_to_csv

st.set_page_config(page_title="Transcript Analyzer", page_icon="📞")

st.title("Transcript Analyzer")
st.markdown("Paste a customer call transcript below to get a summary and sentiment analysis.")

# Input area
transcript = st.text_area("Transcript", height=250)

# Button to trigger analysis
if st.button("🔍 Analyze"):
    if not transcript.strip():
        st.warning("Please paste a transcript first.")
    else:
        with st.spinner("Analyzing with Groq LLM..."):
            inputs = {"transcript": transcript}
            raw_summary = summary_chain.invoke(inputs).content
            raw_sentiment = sentiment_chain.invoke(inputs).content

            summary = clean_summary_output(raw_summary)
            sentiment = clean_sentiment_output(raw_sentiment)


            st.subheader("Summary")
            st.success(summary)

            st.subheader("Sentiment")
            st.info(sentiment)


            save_to_csv(transcript.strip(), summary, sentiment)
            st.toast("Saved to transcripts.csv")
