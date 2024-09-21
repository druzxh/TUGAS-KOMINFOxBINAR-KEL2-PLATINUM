import re
import pandas as pd
import sqlite3

connection = sqlite3.connect(r'data.db', check_same_thread=False)

def clean_text(text):
    text = re.sub(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)
    text = re.sub(r'pic.twitter.com\.\w+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text.lower())
    text = text.replace('user', '')
    text = re.sub(' +', ' ', text)
    text = text.replace('\n', ' ')
    return text.strip()

def standardize_text(text):
    alay_df = pd.read_sql_query('SELECT * FROM kamus_alay', connection)
    alay_dict = dict(zip(alay_df['alay'], alay_df['fix']))
    words = text.split()
    standardized_words = [alay_dict.get(word, word) for word in words]
    return ' '.join(standardized_words).strip()

def remove_stopwords(text):
    stopword_df = pd.read_sql_query('SELECT * FROM stopword', connection)
    stopwords = set(stopword_df['stop'])
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

def preprocess_input(text):
    original_text = text
    text = clean_text(text)
    text = standardize_text(text)
    cleaned_text = text
    return original_text, cleaned_text
