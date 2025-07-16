from flask import Flask, render_template, Response, send_file, redirect, url_for
from ultralytics import YOLO
from datetime import datetime
import cv2
import pandas as pd
import os

app = Flask(__name__)
model = YOLO("model/best.pt")
history = []

# Khởi tạo camera
cap = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Resize và dự đoán
        input_frame = cv2.resize(frame, (640, 640))
        results = model(input_frame)[0]
        labels = []

        for box in results.boxes:
            if box.conf[0] < 0.6:
                continue
            cls = int(box.cls[0])
            label = model.names[cls]
            labels.append(label)

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Lưu lịch sử
        if labels:
            now = datetime.now().strftime("%H:%M:%S %d-%m-%Y")
            for label in labels:
                record = (now, label)
                history.append(record)

        # Encode frame để trả về stream
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', history=history)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/export')
def export():
    if not history:
        return "Không có dữ liệu", 400
    df = pd.DataFrame(history, columns=["Thời gian", "Loại rác"])
    filename = f"lich_su_phat_hien_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    filepath = os.path.join("static", filename)
    df.to_excel(filepath, index=False)
    return send_file(filepath, as_attachment=True)

@app.route('/clear')
def clear():
    history.clear()
    return redirect(url_for('index'))

@app.route('/shutdown')
def shutdown():
    cap.release()
    return "Camera released."

if __name__ == '__main__':
    app.run(debug=True)
