from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
from flasgger import swag_from
import json
import re, pandas as pd, sqlite3, pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
from utils import preprocess_input
from config import create_app

app = create_app()
connection = sqlite3.connect(r'data.db', check_same_thread=False)

@app.route('/')
def redirect_to_docs():
    return redirect("/docs/", code=302)

@swag_from("docs/lstm_input_data.yml", methods=['POST'])
@app.route('/api/v1/lstm/text', methods=['POST'])
def lstm_text():
    try:
        original_text = str(request.form["lstm_text"])
        cleaned_text = preprocess_input(original_text)[1]

        loaded_model = load_model(r'LSTM_MODEL/lstm.h5')

        with open('tokenizer_lstm.json') as f:
            tokenizer_lstm = tokenizer_from_json(json.load(f))

        def pred_sentiment(text):
            sequences = tokenizer_lstm.texts_to_sequences([text])
            padded_sequences = pad_sequences(sequences, maxlen=100)
            predictions = loaded_model.predict(padded_sequences, batch_size=10)
            return predictions[0]

        def pred(predictions):
            labels = ['Negatif', 'Netral', 'Positif']
            return labels[predictions.argmax()]

        predictions = pred_sentiment(cleaned_text)
        sentiment = pred(predictions)

        json_response = {
            'message': "Success",
            'status_code': 200,
            'data': {
                'label': sentiment,
                'text_clean': cleaned_text,
                'text_original': original_text,
            },
        }

        cursor = connection.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS lstm_TEXTINPUT (original_text TEXT, text TEXT, sentiment TEXT)')
        cursor.execute('INSERT INTO lstm_TEXTINPUT (original_text, text, sentiment) VALUES (?, ?, ?)', (original_text, cleaned_text, sentiment))
        connection.commit()

        return jsonify(json_response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@swag_from("docs/lstm_upload_data.yml", methods=['POST'])
@app.route('/api/v1/lstm/upload', methods=['POST'])
def lstm_upload():
    try:
        file = request.files["lstm_upload"]
        df_csv = pd.read_csv(file, encoding="latin-1")
        df_csv['text_clean'] = df_csv['text'].apply(preprocess_input).apply(lambda x: x[1])

        loaded_model = load_model(r'LSTM_Model/lstm.h5')

        with open('tokenizer_lstm.json') as f:
            tokenizer = tokenizer_from_json(json.load(f))

        def pred_sentiment(text):
            sequences = tokenizer.texts_to_sequences([text])
            padded_sequences = pad_sequences(sequences, maxlen=100)
            predictions = loaded_model.predict(padded_sequences, batch_size=10)
            return predictions[0]

        def pred(predictions):
            labels = ['Negatif', 'Netral', 'Positif']
            return labels[predictions.argmax()]

        df_csv['label'] = df_csv['text_clean'].apply(lambda text: pred(pred_sentiment(text)))

        cursor = connection.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS UPLOAD_lstm (TweetOri TEXT, TweetClean TEXT, Sentimen TEXT)')
        for _, row in df_csv.iterrows():
            cursor.execute('INSERT INTO UPLOAD_lstm (TweetOri, TweetClean, Sentimen) VALUES (?, ?, ?)', 
                           (row['text'], row['text_clean'], row['label']))
        connection.commit()

        df_csv.to_csv('sentimen_analisis_lstm.csv', index=False)

        df_csv.to_json('sentimen_analisis_lstm.json', orient='records')

        with open('sentimen_analisis_lstm.json', 'r') as json_file:
            json_data = json.load(json_file)

        return jsonify({
            "status_code": 200,
            "message": "Success",
            "data": json_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@swag_from("docs/nn_input_data.yml", methods=['POST'])
@app.route('/api/v1/neural_network/text', methods=['POST'])
def nn_text():
    try:
        original_text = str(request.form.get('nn_text'))
        cleaned_text = preprocess_input(original_text)[1]

        with open('nn_feature.pkl', 'rb') as f:
            loaded_vectorizer = pickle.load(f)
        with open('nn_model.pkl', 'rb') as f:
            model_NN = pickle.load(f)

        text_vectorized = loaded_vectorizer.transform([cleaned_text])
        sentiment = model_NN.predict(text_vectorized)[0]

        json_response = {
            'message': "Success",
            'status_code': 200,
            'data': {
                'label': sentiment,
                'text_clean': cleaned_text,
                'text_original': original_text,
            },
        }

        cursor = connection.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS nn_TEXTINPUT (original_text TEXT, text TEXT, sentiment TEXT)')
        cursor.execute('INSERT INTO nn_TEXTINPUT (original_text, text, sentiment) VALUES (?, ?, ?)', (original_text, cleaned_text, sentiment))
        connection.commit()

        return jsonify(json_response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@swag_from("docs/nn_upload_data.yml", methods=['POST'])
@app.route('/api/v1/neural_network/upload', methods=['POST'])
def nn_upload():
    try:
        file = request.files["nn_upload"]
        df_csv = pd.read_csv(file, encoding="latin-1")
        df_csv['text_clean'] = df_csv['text'].apply(preprocess_input).apply(lambda x: x[1])

        with open('nn_feature.pkl', 'rb') as f:
            loaded_vectorizer = pickle.load(f)
        with open('nn_model.pkl', 'rb') as f:
            model_NN = pickle.load(f)

        df_csv['label'] = df_csv['text_clean'].apply(lambda text: model_NN.predict(loaded_vectorizer.transform([text]))[0])
        cursor = connection.cursor()
        cursor.execute('CREATE TABLE IF NOT EXISTS Upload_nn (original_text TEXT, text TEXT, sentiment TEXT)')
        for _, row in df_csv.iterrows():
            cursor.execute('INSERT INTO Upload_nn (original_text, text, sentiment) VALUES (?, ?, ?)', 
                           (row['text'], row['text_clean'], row['label']))
        connection.commit()

        df_csv.to_csv('sentimen_analisis_nn.csv', index=False)

        df_csv.to_json('sentimen_analisis_nn.json', orient='records')

        with open('sentimen_analisis_nn.json', 'r') as json_file:
            json_data = json.load(json_file)
            
        return jsonify({
            "status_code": 200,
            "message": "Success",
            "data": json_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)