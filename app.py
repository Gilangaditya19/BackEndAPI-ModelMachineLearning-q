import os
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib  # Tambahkan impor joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)  

# Muat model
model = tf.keras.models.load_model('learning_style_model.h5')

# Muat scaler menggunakan joblib
scaler = joblib.load('learning_style_scaler.joblib')

# Fungsi untuk preprocessing input
def preprocess_input(input_data):
    # Konversi input ke numpy array
    input_array = np.array(input_data).reshape(1, -1)
    
    # Lakukan scaling menggunakan scaler yang dimuat
    scaled_input = scaler.transform(input_array)
    
    return scaled_input

# Fungsi untuk konversi numpy float ke float standar Python
def convert_numpy_to_native(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# Endpoint untuk prediksi gaya belajar
@app.route('/predict', methods=['POST'])
def predict_learning_style():
    try:
        # Ambil data input dari request
        data = request.json
        
        # Validasi input
        if not data or 'answers' not in data:
            return jsonify({
                'error': 'Input tidak valid. Harap kirim jawaban untuk 30 pertanyaan.'
            }), 400
        
        input_answers = data['answers']
        
        # Validasi jumlah jawaban
        if len(input_answers) != 30:
            return jsonify({
                'error': 'Harus ada 30 jawaban. Setiap jawaban harus bernilai 1-4.'
            }), 400
        
        # Validasi rentang jawaban
        if any(answer < 1 or answer > 4 for answer in input_answers):
            return jsonify({
                'error': 'Setiap jawaban harus bernilai antara 1 sampai 4.'
            }), 400
        
        # Preprocessing input
        processed_input = preprocess_input(input_answers)
        
        # Prediksi
        predictions = model.predict(processed_input)
        predicted_class = int(np.argmax(predictions))
        
        # Label gaya belajar
        learning_styles = ['Auditory', 'Visual', 'Kinesthetic']
        
        # Konversi probabilitas ke float biasa
        prob_dict = {
            style: float(prob) for style, prob in zip(learning_styles, predictions[0])
        }
        
        # Siapkan response
        response = {
            'predicted_style': learning_styles[predicted_class],
            'probabilities': prob_dict
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': f'Terjadi kesalahan: {str(e)}'
        }), 500

# Endpoint untuk mengecek status API
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'API untuk Prediksi Gaya Belajar aktif',
        'version': '1.0'
    })

# Jalankan aplikasi
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)