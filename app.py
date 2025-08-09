from flask import Flask, render_template, Response, request, jsonify, redirect, url_for
import cv2
import time
import os
import torch
import torchvision
from PIL import Image
import numpy as np
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch
app = Flask(__name__)

camera = None
backends = [cv2.CAP_V4L2, cv2.CAP_ANY, cv2.CAP_FFMPEG]
for backend in backends:
    for index in range(10):
        temp_camera = cv2.VideoCapture(index, backend)
        if temp_camera.isOpened():
            camera = temp_camera
            print(f"Camera opened successfully at index {index} with backend {backend}")
            break
        temp_camera.release()
    if camera is not None:
        break

if camera is None:
    for index in range(10):
        pipeline = f"v4l2src device=/dev/video{index} ! videoconvert ! appsink"
        temp_camera = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if temp_camera.isOpened():
            camera = temp_camera
            print(f"Camera opened successfully at index {index} with GStreamer")
            break
        temp_camera.release()

if camera is None:
    print("Warning: No camera could be opened. Endpoints /video_feed and /classify may not work.")

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load các model
models = {
    "yolov12": YOLO("model/best.pt"),
    "detr": None,
    "fasterrcnn": None
}


def load_detr_model():
    if models["detr"] is None:
        models["detr"] = torch.hub.load(
            'facebookresearch/detr:main',
            'detr_resnet50',
            pretrained=False,
            num_classes=6  # 6 lớp của bạn (bao gồm background)
        )

        state_dict = torch.load("model/Deformable-DETR-fine-tuned.pth", map_location="cpu")

        # Nếu train multi-GPU, key sẽ có prefix "module."
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Load state_dict, cho phép bỏ qua mismatch ở head nếu khác
        missing, unexpected = models["detr"].load_state_dict(state_dict, strict=False)
        print("DETR missing keys:", missing)
        print("DETR unexpected keys:", unexpected)

        models["detr"].eval()
    return models["detr"]


def load_fasterrcnn_model():
    if models["fasterrcnn"] is None:
        num_classes = 7  # Ví dụ: 6 lớp vật thể + 1 background
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)

        # Load state_dict đã train
        state_dict = torch.load("model/fasterrcnn_resnet50.pth", map_location="cpu")

        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Faster R-CNN missing keys:", missing)
        print("Faster R-CNN unexpected keys:", unexpected)

        model.eval()
        models["fasterrcnn"] = model
    return models["fasterrcnn"]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def classify_and_draw(frame):
    try:
        h, w, _ = frame.shape
        roi_w, roi_h = int(w * 0.5), int(h * 0.5)
        left, top = (w - roi_w)//2, (h - roi_h)//2
        right, bottom = left + roi_w, top + roi_h

        roi = frame[top:bottom, left:right]
        results = models["yolov12"](roi)
        annotated_roi = results[0].plot()

        frame_copy = frame.copy()
        frame_copy[top:bottom, left:right] = annotated_roi
        return frame_copy, results
    except Exception as e:
        print(f"Error in classify_and_draw: {e}")
        raise

def generate_frames():
    if camera is None:
        print("No camera available for video feed")
        return
    while True:
        success, frame = camera.read()
        if not success:
            print("Failed to read frame from camera")
            break
        try:
            annotated_frame, _ = classify_and_draw(frame)
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            if not ret:
                print("Failed to encode frame")
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.05)
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    if camera is None:
        return jsonify({"error": "No camera available"}), 500
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/classify', methods=['GET'])
def classify():
    try:
        if camera is None:
            return jsonify({"error": "No camera available"}), 500

        success, frame = camera.read()
        if not success:
            return jsonify({"error": "No frame"}), 500

        _, results = classify_and_draw(frame)
        print("Boxes detected:", len(results[0].boxes) if results[0].boxes else 0)
        trash_type = results[0].names[results[0].boxes.cls[0].item()] if results[0].boxes else None

        if trash_type:
            mode = "pickup_and_drop"
            bin_coords = {
                "can": [-0.06, 0.52],
                "cigarette": [0.08, 0.52],
                "glass": [0.08, 0.35],
                "paper waste": [0.08, 0.52],
                "plastic bag": [-0.06, 0.35],
                "plastic bottle": [-0.06, 0.35]
            }
            coords = {"drop": bin_coords.get(trash_type, [0, 0])}
        else:
            mode, coords = "idle", {}

        return jsonify({"trash_type": trash_type, "mode": mode, "coords": coords})
    except Exception as e:
        print(f"Error in classify: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        model_name = request.form.get('model', 'yolov12')
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            image = Image.open(filepath).convert("RGB")
            img_np = np.array(image)
            
            if model_name == "yolov12":
                results = models[model_name](img_np)
                annotated_img = results[0].plot()
                annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                detections = []
                if results[0].boxes:
                    for box in results[0].boxes:
                        cls_id = int(box.cls)
                        conf = float(box.conf)
                        label = results[0].names[cls_id]
                        detections.append({
                            "label": label,
                            "confidence": conf,
                            "bbox": box.xyxy[0].tolist()
                        })
            
            elif model_name == "detr":
                model = load_detr_model()
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(800),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                
                img_tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(img_tensor)
                
                probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
                keep = probas.max(-1).values > 0.7
                
                bboxes_scaled = outputs['pred_boxes'][0, keep].cpu().numpy()
                scores = probas[keep].max(-1).values.cpu().numpy()
                labels = probas[keep].argmax(-1).cpu().numpy()
                
                annotated_img = img_np.copy()
                h, w = annotated_img.shape[:2]
                
                detections = []
                for bbox, score, label in zip(bboxes_scaled, scores, labels):
                    cx, cy, bw, bh = bbox
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    
                    color = (0, 255, 0)
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_img, f"Class {label}: {score:.2f}", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    detections.append({
                        "label": f"Class {label}",
                        "confidence": float(score),
                        "bbox": [x1, y1, x2, y2]
                    })
            
            elif model_name == "fasterrcnn":
                model = load_fasterrcnn_model()
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.ToTensor(),
                ])
                
                img_tensor = transform(image)
                with torch.no_grad():
                    outputs = model([img_tensor])
                
                scores = outputs[0]['scores'].cpu().numpy()
                labels = outputs[0]['labels'].cpu().numpy()
                bboxes = outputs[0]['boxes'].cpu().numpy()
                
                keep = scores > 0.5
                
                annotated_img = img_np.copy()
                detections = []
                for bbox, score, label in zip(bboxes[keep], scores[keep], labels[keep]):
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    color = (0, 0, 255)
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_img, f"Class {label}: {score:.2f}", 
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    detections.append({
                        "label": f"Class {label}",
                        "confidence": float(score),
                        "bbox": [x1, y1, x2, y2]
                    })
            
            annotated_filename = f"annotated_{filename}"
            annotated_path = os.path.join(app.config['UPLOAD_FOLDER'], annotated_filename)
            Image.fromarray(annotated_img).save(annotated_path)
            
            return render_template('result.html', 
                                  original_image=filepath, 
                                  annotated_image=annotated_path,
                                  detections=detections,
                                  model_name=model_name)
        
        return redirect(url_for('upload'))
    except Exception as e:
        print(f"Error in detect: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/check_camera', methods=['GET'])
def check_camera():
    return jsonify({"camera_available": camera is not None})

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        if camera is not None:
            camera.release()
            print("Camera released")