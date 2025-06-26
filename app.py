from flask import Flask, render_template, request, redirect, url_for, session
import os
import json
import numpy as np
from PIL import Image
import tensorflow as tf
from datetime import datetime, timedelta
import requests
import webbrowser

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Add a secret key for session management
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Telegram Config
BOT_TOKEN = '7331406271:AAE-q2mlx8Jr1SvcNnaWKiifM6k7npbikv8'
CHAT_ID = '1773371283'

# Files for state saving
HISTORY_FILE = 'flood_history.json'
HISTORY_DETAIL_FILE = 'upload_log.json'
LATEST_FILE = 'latest_upload.json'
WEATHER_HISTORY_FILE = 'weather_history.json'

prediction_history = []
upload_log = []
latest_upload = {}
prediction_details = []

# ✅ Load History
def load_history():
    global prediction_history, upload_log, latest_upload, prediction_details
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            prediction_history[:] = json.load(f)
    if os.path.exists(HISTORY_DETAIL_FILE):
        with open(HISTORY_DETAIL_FILE, 'r') as f:
            data = json.load(f)
            upload_log[:] = data
            prediction_details[:] = data  # ✅ ensure popup data syncs
    if os.path.exists(LATEST_FILE):
        with open(LATEST_FILE, 'r') as f:
            content = json.load(f)
            if isinstance(content, dict):
                latest_upload.clear()
                latest_upload.update(content)
            else:
                latest_upload.clear()

# ✅ Save History
def save_history():
    with open(HISTORY_FILE, 'w') as f:
        json.dump(prediction_history, f)
    with open(HISTORY_DETAIL_FILE, 'w') as f:
        json.dump(upload_log, f)
    with open(LATEST_FILE, 'w') as f:
        json.dump(latest_upload, f)

def send_telegram_alert(filename, percent):
    message = f"""
    🌊 **Flood Alert!**
    🚨 **Flood Detected**: {percent}%
    📸 **Image**: {filename}

    ⚠️ **Impacted Area**: Significant flood detected in the region!
    🛑 **Urgent Action Required**: Please follow local evacuation protocols and stay safe.

    📞 **For Assistance**:
       - **Contact**: SAKTHYNII
       - **Phone**: 014-9325314
       - **Email**: support@floodpredictor.com
    """
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print("Telegram Error:", e)


# ✅ Load model
model = tf.keras.models.load_model('flood_segmentation_unet.h5')
load_history()

# ✅ Prediction Function
def predict_flood(image_path):
    original = Image.open(image_path).convert('RGB')
    original_size = original.size
    resized = original.resize((256, 256))
    img = np.array(resized) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0]
    mask = (prediction > 0.5).astype(np.uint8).squeeze()
    flooded_percent = round((np.sum(mask) / mask.size) * 100, 2)

    mask_img = Image.fromarray(mask * 255).resize(original_size)
    overlay = np.array(original)
    mask_arr = np.array(mask_img)
    overlay[mask_arr > 127] = [255, 0, 0]

    name = os.path.splitext(os.path.basename(image_path))[0]
    Image.fromarray(mask_arr).save(f"{UPLOAD_FOLDER}/{name}_mask.png")
    Image.fromarray(overlay).save(f"{UPLOAD_FOLDER}/{name}_overlay.png")
    original.save(f"{UPLOAD_FOLDER}/{name}_original.png")

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    prediction_history.append(flooded_percent)
    entry = {
        "filename": name,
        "flood_percent": flooded_percent,
        "timestamp": timestamp
    }
    upload_log.append(entry)
    prediction_details.append(entry)

    latest_upload.clear()
    latest_upload.update({
        "filename": name,
        "flood_percent": flooded_percent
    })
    save_history()

    if flooded_percent > 20:
        send_telegram_alert(name, flooded_percent)

    return name + "_original.png", name + "_mask.png", name + "_overlay.png", flooded_percent

# ✅ Weather handling
def load_weather_history():
    if os.path.exists(WEATHER_HISTORY_FILE):
        with open(WEATHER_HISTORY_FILE, 'r') as f:
            return json.load(f)
    return []

def save_weather_history(weather_data):
    weather_history = load_weather_history()
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    if not any(entry['date'] == yesterday for entry in weather_history):
        weather_history.append(weather_data)
    weather_history = weather_history[-7:]
    with open(WEATHER_HISTORY_FILE, 'w') as f:
        json.dump(weather_history, f)

def fetch_weather_data(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid=4257a2de4d308adc1bc73af584db4f51&units=metric"
    response = requests.get(url)
    data = response.json()
    return {
        'date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
        'city': data['name'],
        'description': data['weather'][0]['description'],
        'temperature': data['main']['temp'],
        'humidity': data['main']['humidity'],
        'wind_speed': data['wind']['speed']
    }

# ✅ Routes
@app.route('/')
def home():
    return redirect(url_for('upload'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', original=None, mask=None, overlay=None, flooded_percent="No file uploaded")

        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', original=None, mask=None, overlay=None, flooded_percent="No file selected")

        if file:
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            original, mask, overlay, flooded_percent = predict_flood(filepath)

            # Store in session for persistence across tabs
            session['original'] = original
            session['mask'] = mask
            session['overlay'] = overlay
            session['flooded_percent'] = flooded_percent

            return render_template('upload.html',
                                   original=original,
                                   mask=mask,
                                   overlay=overlay,
                                   flooded_percent=flooded_percent)

    # Retrieve from session if available
    original = session.get('original', None)
    mask = session.get('mask', None)
    overlay = session.get('overlay', None)
    flooded_percent = session.get('flooded_percent', None)

    return render_template('upload.html', original=original, mask=mask, overlay=overlay, flooded_percent=flooded_percent)

@app.route('/chart')
def chart():
    labels = [f"Prediction {i+1}" for i in range(len(prediction_history))]
    return render_template("chart.html", labels=labels, history=prediction_history, details=prediction_details)

@app.route('/weather')
def weather():
    weather_data = fetch_weather_data("Perlis")
    save_weather_history(weather_data)
    weather_history = load_weather_history()
    return render_template('weather.html', weather_data=weather_data, weather_history=weather_history)

@app.route('/info')
def info():
    return render_template('info.html')  # Render the new info page with contact and safety info

@app.route('/history')
def history():
    return render_template('history.html', uploads=upload_log)

@app.route('/delete/<filename>', methods=['POST'])
def delete_upload(filename):
    global upload_log
    upload_log = [entry for entry in upload_log if entry["filename"] != filename]

    if latest_upload.get("filename") == filename:
        latest_upload.clear()

    for suffix in ["_original.png", "_mask.png", "_overlay.png"]:
        try:
            os.remove(os.path.join(UPLOAD_FOLDER, filename + suffix))
        except:
            pass

    save_history()
    return redirect(url_for('history'))

# ✅ Run App
if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(debug=True)
