import dlib
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import requests
import time
import RPi.GPIO as GPIO
import sys
import json
from gpiozero import LED


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")

TRIG_PIN = 18
ECHO_PIN = 24
led = LED(17)

GPIO.setwarnings(False)

# 얼굴 찾는 함수
def find_faces(img):
    dets = detector(img, 1)

    if len(dets) == 0:
        return []

    shapes = []

    for d in dets:
        shape = sp(img, d)
        shapes.append(shape)

    return shapes

# 얼굴을 인코딩하는 함수
def encode_faces(img, shapes):
    face_descriptors = []

    for shape in shapes:
        face_descriptor = np.array(facerec.compute_face_descriptor(img, shape))
        face_descriptors.append(face_descriptor)

    return face_descriptors

# 초음파 센서로 거리 구하는 함수
def measure_distance():
    GPIO.output(TRIG_PIN, GPIO.HIGH)
    time.sleep(0.00001)
    GPIO.output(TRIG_PIN, GPIO.LOW)

    while GPIO.input(ECHO_PIN) == 0:
        pulse_start = time.time()

    while GPIO.input(ECHO_PIN) == 1:
        pulse_end = time.time()

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150  # 소리의 속도인 34300 cm/s로 나누어서 거리 계산
    distance = round(distance, 2)      # 소수점 둘째 자리까지 반올림

    return distance


            

# 모델 로드
knn = joblib.load('face_recognition_model.joblib')
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder_classes.npy')

# 서버 URL
url = "http://192.168.137.100:8080/attendance"

def sonicSenor():
    # GPIO 핀 설정
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(TRIG_PIN, GPIO.OUT)
    GPIO.setup(ECHO_PIN, GPIO.IN)

    while True:
        distance = measure_distance()
        print("측정 거리:", distance, "cm")

        if distance < 50:
            return

def faceRecognition():
    cap = cv2.VideoCapture(0)
    ret = cap.set(3,320)
    ret = cap.set(4,240)

    if not cap.isOpened():
        exit()

    check_time = time.time()
    start_time = None
    recognized_id = None

    while True:
        ret, img_bgr = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        faces = find_faces(img_rgb)
        face_descriptors = encode_faces(img_rgb, faces)

        face_detected = False

        for face_descriptor, face in zip(face_descriptors, faces):
            face_detected = True
            check_time = time.time()
            label = knn.predict([face_descriptor])[0]
            dist = knn.kneighbors([face_descriptor])[0][0][0]

            threshold = 0.4

            if dist < threshold:
                name = label_encoder.inverse_transform([label])[0]

                if recognized_id == name:
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= 1.5:
                        datas = {
                            'stdId': recognized_id
                        }
                        headers = {
                            "Content-Type": "application/json"    
                        }
                        try:
                            response = requests.post(url, data=json.dumps(datas), headers=headers, verify=False)
                            if response.status_code == 200:
                                print("Student ID sent successfully.")
                                led.on()
                                time.sleep(1)
                                led.off()
                            else:
                                print("Failed to send student ID.")
                        except requests.exceptions.RequestException as e:
                            print("Failed to connect to the server:", str(e))
                            exit()
                        start_time = None 
                else:
                    start_time = time.time()
                    recognized_id = name

                left = face.rect.left()
                top = face.rect.top()
                right = face.rect.right()
                bottom = face.rect.bottom()

                cv2.rectangle(img_bgr, (left, top), (right, bottom), color=(255, 0, 0), thickness=2)
                cv2.putText(img_bgr, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=2)

        cv2.imshow('Face Recognition', img_bgr)
        if cv2.waitKey(1) == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()
            return
        if not face_detected and time.time() - check_time >= 15:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    while True:
        sonicSenor()
        faceRecognition()
