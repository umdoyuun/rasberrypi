import subprocess
import base64
import os
from flask import Flask, request, jsonify

app = Flask(__name__)
face_recognition_process = None

get_image = True

def train_model():
    train_process = subprocess.Popen(["python", "train_model.py"])
    train_process.wait()

@app.route('/start', methods=['POST'])
def start_recognition():
    global face_recognition_process
    if face_recognition_process is None or face_recognition_process.poll() is not None:
        face_recognition_process = subprocess.Popen(["python", "recog_sonic.py"])
        return "Face recognition started"
    else:
        return "Face recognition is already running"

@app.route('/stop', methods=['POST'])
def stop_recognition():
    global face_recognition_process
    #global get_image
    if face_recognition_process is not None and face_recognition_process.poll() is None:
        face_recognition_process.terminate()
        #train_model()
        return "Face recognition stopped"
    else:
        return "Face recognition is not running"

@app.route('/images', methods=['POST'])
def upload_images():
    #global get_image
    images = request.json['images']
    stdNum = request.json['studentNum']
    
    # 이미지 저장 경로
    upload_folder = "/home/pi/FinalProject/images"
    
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    
    for idx, image_base64 in enumerate(images):
        image_data = base64.b64decode(image_base64)
        filename = f'{stdNum[1:9]+"_"}{+idx+1}.jpg'
        filepath = os.path.join(upload_folder, filename)
    
        with open(filepath, 'wb') as f:
            f.write(image_data)
    get_image = False
    return "Image uploaded"

if __name__ == '__main__':
    #train_model()
    app.run(host='0.0.0.0', port=9999)
