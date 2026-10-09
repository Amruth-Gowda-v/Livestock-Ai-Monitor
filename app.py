import base64
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import IsolationForest
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse

load_dotenv()

app = Flask(__name__)
# Keep uploads small enough for a laptop hackathon demo.
app.config["MAX_CONTENT_LENGTH"] = 35 * 1024 * 1024

BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER", "")
FARMER_PHONE_NUMBER = os.getenv("FARMER_PHONE_NUMBER", "")

# -----------------------------
# DEMO AI ANOMALY MODEL
# -----------------------------
# Features:
# 1) temperature deviation from the demo baseline
# 2) movement percentage
# 3) feeding percentage
#
# IMPORTANT: These are DEMO values, not veterinary thresholds.
rng = np.random.default_rng(42)
normal_data = np.column_stack([
    rng.normal(0.0, 0.35, 500),   # temp deviation
    np.clip(rng.normal(80, 9, 500), 0, 100),
    np.clip(rng.normal(82, 8, 500), 0, 100),
])

model = IsolationForest(
    n_estimators=150,
    contamination=0.04,
    random_state=42
)
model.fit(normal_data)

animals = {}
for i in range(1, 13):
    animals[i] = {
        "id": i,
        "temperature": round(float(rng.normal(38.6, 0.25)), 1),
        "movement": int(np.clip(rng.normal(82, 7), 0, 100)),
        "feeding": int(np.clip(rng.normal(84, 6), 0, 100)),
        "status": "LOW",
        "risk": 0,
        "reason": "Pattern currently within the demo baseline."
    }


def evaluate(animal):
    baseline = 38.5
    temp_deviation = animal["temperature"] - baseline

    # Demo scoring only.
    temp_risk = np.clip((temp_deviation / 2.0) * 100, 0, 100)
    movement_risk = 100 - animal["movement"]
    feeding_risk = 100 - animal["feeding"]

    risk = (
        0.45 * temp_risk +
        0.30 * movement_risk +
        0.25 * feeding_risk
    )

    features = np.array([[
        temp_deviation,
        animal["movement"],
        animal["feeding"]
    ]])

    anomaly = model.predict(features)[0] == -1
    if anomaly:
        risk = max(risk, 72)

    if risk >= 70:
        status = "HIGH"
    elif risk >= 40:
        status = "MEDIUM"
    else:
        status = "LOW"

    reasons = []
    if temp_risk >= 40:
        reasons.append("temperature trend is abnormal")
    if animal["movement"] <= 55:
        reasons.append("movement is unusually low")
    if animal["feeding"] <= 55:
        reasons.append("feeding activity is unusually low")

    if not reasons:
        reasons.append("pattern currently within the demo baseline")

    animal["risk"] = int(round(min(risk, 100)))
    animal["status"] = status
    animal["reason"] = "; ".join(reasons) + "."

    return animal


for a in animals.values():
    evaluate(a)


# -----------------------------
# YOLO LIVESTOCK DETECTION
# -----------------------------
# The default COCO model knows cow, sheep, and horse. If YOLO_MODEL_NAME is
# changed to a compatible custom livestock .pt model, the class-name mapping
# below will automatically include supported species present in that model.
YOLO_MODEL_NAME = os.getenv("YOLO_MODEL_NAME", "yolo26s.pt")
yolo_model = None

LIVESTOCK_LABEL_ALIASES = {
    "cow": "Cow", "cattle": "Cattle", "bull": "Cattle", "bullock": "Cattle",
    "calf": "Cattle", "ox": "Cattle", "buffalo": "Buffalo", "water buffalo": "Buffalo",
    # COCO only labels generic birds; do not claim the model knows they are hens.
    "bird": "Bird / Poultry (species uncertain)",
    "goat": "Goat", "kid": "Goat", "sheep": "Sheep", "lamb": "Sheep",
    "ram": "Sheep", "ewe": "Sheep", "horse": "Horse", "pony": "Horse", "foal": "Horse",
    "donkey": "Donkey", "mule": "Mule", "pig": "Pig", "piglet": "Pig",
    "swine": "Pig", "hog": "Pig", "boar": "Pig", "chicken": "Chicken",
    "hen": "Chicken", "rooster": "Chicken", "cockerel": "Chicken", "duck": "Duck",
    "goose": "Goose", "turkey": "Turkey", "camel": "Camel", "yak": "Yak",
    "alpaca": "Alpaca", "llama": "Llama", "rabbit": "Rabbit"
}


def get_yolo_model():
    global yolo_model
    if yolo_model is None:
        from ultralytics import YOLO
        yolo_model = YOLO(YOLO_MODEL_NAME)
    return yolo_model


def get_livestock_class_map(model):
    """Map this model's known livestock classes to class IDs and display names."""
    names = model.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    class_map = {}
    for class_id, raw_name in items:
        normalized = " ".join(str(raw_name).lower().replace("_", " ").replace("-", " ").split())
        if normalized in LIVESTOCK_LABEL_ALIASES:
            class_map[int(class_id)] = LIVESTOCK_LABEL_ALIASES[normalized]
    return class_map


def draw_animal_label(cv2, image, box, label, confidence):
    x1, y1, x2, y2 = [int(value) for value in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), (30, 210, 90), 3)
    text = f"{label}  {confidence:.0%}"
    text_y = max(24, y1 - 10)
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    )
    cv2.rectangle(
        image,
        (x1, text_y - text_height - 8),
        (x1 + text_width + 8, text_y + 4),
        (30, 210, 90),
        -1
    )
    cv2.putText(
        image, text, (x1 + 4, text_y - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (15, 25, 20), 2,
        cv2.LINE_AA
    )


@app.post("/api/detect")
def detect_livestock_in_image():
    """Detect livestock in an uploaded photo. Live-camera scanning is disabled."""
    uploaded = request.files.get("image")
    if uploaded is None:
        return jsonify({"ok": False, "error": "Please attach an image in the 'image' field."}), 400

    # Reject frames sent by an old version of the live-camera scan button.
    # The live camera remains a preview; detection is upload-photo only.
    if Path(uploaded.filename or "").name.lower() == "livestock-camera-capture.jpg":
        return jsonify({
            "ok": False,
            "error": "Live-camera detection is disabled. Please choose a livestock photo in the Photo detection section."
        }), 400

    file_bytes = uploaded.read()
    if not file_bytes:
        return jsonify({"ok": False, "error": "The uploaded image is empty."}), 400
    if len(file_bytes) > 12 * 1024 * 1024:
        return jsonify({"ok": False, "error": "Please use an image smaller than 12 MB."}), 413

    try:
        import cv2
        image = cv2.imdecode(np.frombuffer(file_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"ok": False, "error": "That file could not be read as an image."}), 400

        # Keep the original pixels instead of shrinking the photo before inference.
        # A larger inference size helps when animals occupy a small part of a herd photo.

        detection_model = get_yolo_model()
        class_map = get_livestock_class_map(detection_model)
        if not class_map:
            return jsonify({
                "ok": False,
                "error": f"The model '{YOLO_MODEL_NAME}' has no recognised livestock classes. Use a livestock-trained YOLO model."
            }), 400

        result = detection_model.predict(
            source=image,
            classes=list(class_map.keys()),
            imgsz=1280,
            conf=0.25,
            iou=0.50,
            max_det=300,
            verbose=False
        )[0]

        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].cpu().item())
                species = class_map.get(class_id)
                if not species:
                    continue
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                confidence = float(box.conf[0].cpu().item())
                detections.append({"box": xyxy, "confidence": confidence, "species": species})

        detections.sort(key=lambda item: (item["box"][0], item["box"][1]))
        annotated = image.copy()
        response_detections = []
        for number, detection in enumerate(detections, start=1):
            label = f"Animal {number:02d} · {detection['species']}"
            draw_animal_label(cv2, annotated, detection["box"], label, detection["confidence"])
            response_detections.append({
                "id": number,
                "label": label,
                "species": detection["species"],
                "confidence": round(detection["confidence"], 4),
                "box": [round(value, 1) for value in detection["box"]]
            })

        success, encoded = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not success:
            return jsonify({"ok": False, "error": "Could not create the annotated image."}), 500

        image_data_url = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")
        supported_species = sorted(set(class_map.values()))
        return jsonify({
            "ok": True,
            "count": len(response_detections),
            "detections": response_detections,
            "supported_species": supported_species,
            "annotated_image": image_data_url,
            "note": "Animal IDs are assigned left-to-right for this image only. Counts depend on photo quality and the model. The default general-purpose model labels birds generically, so it cannot confirm that a detected bird is a hen. For dependable species-specific counts, use a model trained and evaluated on the target livestock and image conditions."
        })

    except ImportError:
        return jsonify({
            "ok": False,
            "error": "YOLO/OpenCV is not installed yet. Run: python -m pip install ultralytics opencv-python"
        }), 503
    except Exception:
        app.logger.exception("Livestock image detection failed")
        return jsonify({
            "ok": False,
            "error": "Detection failed. Check the VS Code terminal for the detailed error."
        }), 500



@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/state")
def state():
    for a in animals.values():
        evaluate(a)

    counts = {
        "total": len(animals),
        "low": sum(a["status"] == "LOW" for a in animals.values()),
        "medium": sum(a["status"] == "MEDIUM" for a in animals.values()),
        "high": sum(a["status"] == "HIGH" for a in animals.values()),
    }

    return jsonify({
        "animals": list(animals.values()),
        "counts": counts
    })


@app.post("/api/simulate")
def simulate():
    # This creates the demo event.
    a = animals[7]
    a["temperature"] = 40.1
    a["movement"] = 28
    a["feeding"] = 35
    evaluate(a)

    return jsonify({
        "message": "Abnormal pattern simulated for Animal #07.",
        "animal": a
    })


@app.post("/api/reset")
def reset():
    rng2 = np.random.default_rng(123)
    for i, a in animals.items():
        a["temperature"] = round(float(rng2.normal(38.6, 0.25)), 1)
        a["movement"] = int(np.clip(rng2.normal(82, 7), 0, 100))
        a["feeding"] = int(np.clip(rng2.normal(84, 6), 0, 100))
        evaluate(a)

    return jsonify({"message": "Demo reset."})


@app.post("/call")
def call_farmer():
    if not all([
        BASE_URL,
        TWILIO_ACCOUNT_SID,
        TWILIO_AUTH_TOKEN,
        TWILIO_FROM_NUMBER,
        FARMER_PHONE_NUMBER
    ]):
        return jsonify({
            "ok": False,
            "error": "Missing Twilio/PUBLIC_BASE_URL settings in .env"
        }), 400

    animal = evaluate(animals[7])
    if animal["status"] != "HIGH":
        return jsonify({
            "ok": False,
            "error": "Animal #07 is not currently HIGH risk."
        }), 400

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    call = client.calls.create(
        to=FARMER_PHONE_NUMBER,
        from_=TWILIO_FROM_NUMBER,
        url=f"{BASE_URL}/voice"
    )

    return jsonify({
        "ok": True,
        "message": "Call triggered.",
        "sid": call.sid
    })


@app.post("/voice")
def voice():
    response = VoiceResponse()

    # Put a self-recorded Kannada MP3 in static/alert_kn.mp3.
    audio_file = Path(app.static_folder) / "alert_kn.mp3"

    if audio_file.exists() and BASE_URL:
        response.play(f"{BASE_URL}/static/alert_kn.mp3")
    else:
        # Fallback so the call still works if the MP3 is not ready.
        response.say(
            "Alert. Animal number 7 has an abnormal health risk pattern. "
            "Please check the animal and contact a veterinarian.",
            language="en-IN"
        )

    response.hangup()
    return str(response), 200, {"Content-Type": "text/xml"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
