from Chains.summary_chain import summary_chain, clean_summary_output
from Chains.sentiment_chain import sentiment_chain, clean_sentiment_output
from Utils.csv_utils import save_to_csv

def analyze_transcript(transcript: str):
    inputs = {"transcript": transcript}


    raw_summary = summary_chain.invoke(inputs).content
    raw_sentiment = sentiment_chain.invoke(inputs).content


    summary = clean_summary_output(raw_summary)
    sentiment = clean_sentiment_output(raw_sentiment)

    return summary, sentiment

if __name__ == "__main__":
    transcript = """
    Customer: Hi, I recently ordered a laptop from your website, and I noticed it hasn’t shipped yet. Can you tell me the status?
    Agent: Hello! Thank you for reaching out. Let me check the status of your order. Could you please provide me with your order number?
    Customer: Sure, it’s 123456789.
    Agent: Thank you. I’m looking into it now. Your order was received on September 10th, and it’s currently being processed. The estimated shipping date is September 22nd.
    Customer: Okay, thanks for the update. Will I get a tracking number once it ships?
    Agent: Yes, once your order ships, you will receive an email with the tracking details.
    Customer: Alright, that’s clear. Thanks for the information.
    Agent: You’re welcome. If you have any other questions, feel free to ask.
    Customer: That’s all for now. Have a good day.
    Agent: You too. Thank you for contacting us.
    """

    print("\nAnalyzing transcript...\n")
    summary, sentiment = analyze_transcript(transcript)

    print("Transcript:\n", transcript.strip())
    print("\nSummary:\n", summary)
    print("\nSentiment:\n", sentiment)

    save_to_csv(transcript.strip(), summary, sentiment)
