from flask import Flask, render_template, Response, request, send_from_directory
import cv2
import numpy as np
import io
from camera import VideoCamera

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('base.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame +
               b'\r\n\r\n')
        
        

@app.route('/video')
def video():
    return Response(gen(VideoCamera()),  # Update to VideoCamera
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/static/<path:path>')
def static_file(path):
    return send_from_directory('static', path)

@app.route('/process_image', methods=['POST'])
def process_image():
    image = request.files['image'].read()
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Applying face detection
    faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = faceDetect.detectMultiScale(img, 1.3, 5)
    for x, y, w, h in faces:
        x1, y1 = x + w, y + h
        cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 255), 1)
        cv2.line(img, (x, y), (x + 30, y), (255, 0, 255), 6) 
        cv2.line(img, (x, y), (x, y + 30), (255, 0, 255), 6)
        cv2.line(img, (x1, y), (x1 - 30, y), (255, 0, 255), 6) 
        cv2.line(img, (x1, y), (x1, y + 30), (255, 0, 255), 6)
        cv2.line(img, (x, y1), (x + 30, y1), (255, 0, 255), 6) 
        cv2.line(img, (x, y1), (x, y1 - 30), (255, 0, 255), 6)
        cv2.line(img, (x1, y1), (x1 - 30, y1), (255, 0, 255), 6) 
        cv2.line(img, (x1, y1), (x1, y1 - 30), (255, 0, 255), 6)
    
    # Encoding the processed image and send it back to the client
    _, processed_image = cv2.imencode('.jpg', img)
    return processed_image.tobytes(), {'Content-Type': 'image/jpeg'}


if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=5000)