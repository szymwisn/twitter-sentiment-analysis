import pandas as pd
import numpy as np
import re

def parse_text_preprocessing():
    df = pd.read_csv('../data-analysis/data/tweets.csv')
    df.head()

    # Remove rows without sentiment
    df = df.dropna(subset=['sentiment'])
    df = df.reset_index(drop=True)

    # Remove rows if language is not english
    df = df[df['language'].isin(['en'])]
    df = df.reset_index(drop=True)



    EMAIL_PATTERN = re.compile(
        r'(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)'
    )

    MENTION_PATTERN = re.compile(r'(^|[^@\w])@(\w{1,15})\b')

    HASHTAG_PATTERN = re.compile(r'(^|[^@\w])#(\w{1,15})\b')

    URL_PATTERN = re.compile(r'((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)')

    NON_ALPHA_PATTERN = re.compile('[^a-zA-Z0-9]')

    SEQUENCE_PATTERN = re.compile(r'(.)\1\1+')

    SEQUENCE_REPLACE_PATTERN = r'\1\1'


    def replace_emails(text):
        return re.sub(EMAIL_PATTERN, ' email', text)


    def replace_mentions(text):
        return re.sub(MENTION_PATTERN, ' mention', text)


    def replace_hashtags(text):
        return re.sub(HASHTAG_PATTERN, ' hashtag', text)


    def replace_urls(text):
        return re.sub(URL_PATTERN, ' URL', text)


    def replace_non_alphas(text):
        return re.sub(NON_ALPHA_PATTERN, ' ', text)


    def replace_sequences(text):
        return re.sub(SEQUENCE_PATTERN, SEQUENCE_REPLACE_PATTERN, text)




            
    df['text'] = df['text'].map(lambda text: text.lower())
    df['text'] = df['text'].map(lambda text: replace_emails(text))
    df['text'] = df['text'].map(lambda text: replace_mentions(text))
    df['text'] = df['text'].map(lambda text: replace_hashtags(text))
    df['text'] = df['text'].map(lambda text: replace_urls(text))
    df['text'] = df['text'].map(lambda text: replace_non_alphas(text))
    df['text'] = df['text'].map(lambda text: replace_sequences(text))


    X = df['text'].to_numpy()
    y = df['sentiment'].to_numpy()

    # save preprocessed data to .npy file
    np.save('../data-analysis/data/X_rnn.npy', X)  
    np.save('../data-analysis/data/y_rnn.npy', y)  


parse_text_preprocessing()