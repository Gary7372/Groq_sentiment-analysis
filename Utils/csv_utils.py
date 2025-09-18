import os
import pandas as pd

def save_to_csv(transcript, summary, sentiment, filename="call_analysis.csv"):
    if os.path.exists(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=["Transcript", "Summary", "Sentiment"])

    new_row = pd.DataFrame([[transcript, summary, sentiment]], columns=["Transcript", "Summary", "Sentiment"])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(filename, index=False)
    print(f"\nSaved to csv")
