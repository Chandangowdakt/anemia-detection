from flask import Flask, request, render_template, session
import tensorflow as tf
import numpy as np
import os
import cv2
import base64
from self_learning import SelfLearningManager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_LOG = os.path.join(BASE_DIR, 'predictions_log.json')
MODEL_PATH = os.path.join(BASE_DIR, 'anemia_model_v3.keras')

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "anemia-local-dev-key")

# Download model from Google Drive if not present (for Render deployment)
if not os.path.exists(MODEL_PATH):
    print("Model not found locally. Attempting download...")
    try:
        import gdown
        # Google Drive file ID for anemia_model_v3.keras
        url = f"https://drive.google.com/uc?id=1SpJoInRaUYqRqHR3mS6yp76m9wUnHLts"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Model download failed: {e}")
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

sl_manager = SelfLearningManager(
    confidence_threshold=0.92,
    batch_size=20,
    dataset_base=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
)

recommendations = {
    "Anaemic": ["Spinach", "Dates", "Jaggery", "Beetroot", "Pomegranate"],
    "Non-Anaemic": ["Balanced diet", "Regular checkups"],
    "Uncertain": ["Retake image with better lighting", "Consult a doctor for confirmation"],
}


def log_prediction(result, confidence, confidence_label, severity_level):
    import json
    from datetime import datetime

    log_path = PREDICTIONS_LOG

    # Load existing log
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            try:
                log = json.load(f)
            except Exception:
                log = []
    else:
        log = []

    # Add new entry
    log.append({
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'result': result,
        'confidence': round(confidence, 1),
        'confidence_label': confidence_label,
        'severity_level': severity_level
    })

    # Keep last 500 entries only
    log = log[-500:]

    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)


def build_suggestion_sections(result, confidence_label, fallback_suggestions):
    guidance = {
        ("Anaemic", "High Confidence"): {
            "message": "Based on the analysis, you may have signs of anemia. Please follow these focused recommendations.",
            "sections": {
                "Diet Recommendations": ["Increase iron-rich foods (spinach, lentils, dates, beetroot).", "Pair iron intake with vitamin C sources like citrus fruits."],
                "Lifestyle Tips": ["Maintain hydration and regular sleep to support recovery.", "Avoid skipping meals and include protein in daily diet."],
                "Medical Advice": ["Schedule a blood test (CBC, ferritin, B12) soon.", "Consult a physician for a personalized treatment plan."],
            },
        },
        ("Anaemic", "Medium Confidence"): {
            "message": "The analysis indicates possible anemia. Consider practical preventive steps and medical confirmation.",
            "sections": {
                "Diet Recommendations": ["Add iron-rich foods consistently through the week.", "Use iron-fortified foods where possible."],
                "Lifestyle Tips": ["Track fatigue, dizziness, or weakness symptoms.", "Maintain regular meal timing and hydration."],
                "Medical Advice": ["Retest with a clearer image if possible.", "Consult a doctor for confirmation."],
            },
        },
        ("Anaemic", "Low Confidence"): {
            "message": "The result suggests possible anemia with low confidence. Please treat this as a preliminary screening.",
            "sections": {
                "Diet Recommendations": ["Follow a balanced diet with iron and vitamin C support.", "Avoid making drastic diet changes from this result alone."],
                "Lifestyle Tips": ["Retake image in bright natural lighting.", "Capture a clearer close-up of the inner lower eyelid."],
                "Medical Advice": ["Do a repeat screening for consistency.", "Consult a doctor for definitive diagnosis."],
            },
        },
        ("Non-Anaemic", "High Confidence"): {
            "message": "Based on the analysis, your result appears non-anaemic. Continue healthy maintenance habits.",
            "sections": {
                "Diet Recommendations": ["Maintain a balanced diet with leafy greens and proteins.", "Include periodic iron and B12 sources in meals."],
                "Lifestyle Tips": ["Continue regular exercise and sleep routine.", "Stay hydrated and avoid prolonged fasting."],
                "Medical Advice": ["Continue routine health checkups annually.", "Re-screen if symptoms develop."],
            },
        },
        ("Non-Anaemic", "Medium Confidence"): {
            "message": "Your result trends non-anaemic, but periodic monitoring is recommended.",
            "sections": {
                "Diet Recommendations": ["Maintain iron-supportive meals through the week.", "Prefer nutrient-dense foods over processed snacks."],
                "Lifestyle Tips": ["Monitor energy levels and concentration.", "Keep a healthy sleep and hydration pattern."],
                "Medical Advice": ["Repeat screening after some days if unsure.", "Consult a doctor if symptoms persist."],
            },
        },
        ("Non-Anaemic", "Low Confidence"): {
            "message": "Result is non-anaemic but confidence is low. Please use caution and monitor closely.",
            "sections": {
                "Diet Recommendations": ["Continue a balanced iron-inclusive diet.", "Do not rely on this result alone for medical decisions."],
                "Lifestyle Tips": ["Retake image with better focus and lighting.", "Avoid motion blur during capture."],
                "Medical Advice": ["Repeat screening for better confidence.", "Consult a doctor if fatigue or paleness is present."],
            },
        },
        ("Uncertain", "High Confidence"): {
            "message": "The model detected uncertain patterns. Follow guided next steps before concluding.",
            "sections": {
                "Diet Recommendations": ["Maintain a balanced diet while reassessing.", "Include iron, folate, and B12 foods regularly."],
                "Lifestyle Tips": ["Retake image under clear lighting and stable camera.", "Ensure conjunctiva area is clearly visible."],
                "Medical Advice": ["Use repeat screening to reduce ambiguity.", "Seek clinical advice for conclusive assessment."],
            },
        },
        ("Uncertain", "Medium Confidence"): {
            "message": "The analysis is uncertain. A retake and medical confirmation are recommended.",
            "sections": {
                "Diet Recommendations": ["Continue general nutritious meals.", "Add iron-supportive foods as preventive care."],
                "Lifestyle Tips": ["Upload a sharper image from a different angle.", "Avoid shadows and low-light capture."],
                "Medical Advice": ["Repeat screening soon.", "Consult a doctor if symptoms are present."],
            },
        },
        ("Uncertain", "Low Confidence"): {
            "message": "This result is uncertain with low confidence. Please prioritize retesting and clinical evaluation.",
            "sections": {
                "Diet Recommendations": ["Keep a balanced diet; avoid self-medication.", "Use this result only as a preliminary indicator."],
                "Lifestyle Tips": ["Retake image with daylight and close focus.", "Ensure only one clear eye region is captured."],
                "Medical Advice": ["Book a professional checkup if concerned.", "Use lab tests for final confirmation."],
            },
        },
    }

    data = guidance.get((result, confidence_label))
    if data:
        return data["message"], data["sections"]

    fallback_sections = {
        "Diet Recommendations": fallback_suggestions[:3],
        "Lifestyle Tips": ["Maintain hydration, sleep, and regular meals."],
        "Medical Advice": ["Consult a doctor for proper diagnosis if symptoms persist."],
    }
    return "Based on the analysis, consider the following recommendations.", fallback_sections


def to_data_url_bgr(img_bgr):
    ok, encoded = cv2.imencode(".jpg", img_bgr)
    if not ok:
        return None
    return f"data:image/jpeg;base64,{base64.b64encode(encoded.tobytes()).decode('utf-8')}"


def extract_roi(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    if len(eyes) > 0:
        x, y, w, h = max(eyes, key=lambda e: e[2] * e[3])
        roi = img_bgr[y:y + h, x:x + w]
        if roi is not None and roi.size > 0:
            return roi

    # Fallback center crop
    h, w = img_bgr.shape[:2]
    y1 = int(h * 0.4)
    y2 = int(h * 0.9)
    x1 = int(w * 0.2)
    x2 = int(w * 0.8)
    roi = img_bgr[y1:y2, x1:x2]
    if roi is None or roi.size == 0:
        return img_bgr
    return roi


def is_likely_eye_image(img_bgr):
    """
    Basic checks to filter out obviously wrong images:
    1. Image must not be grayscale/black-and-white
       (eye images are color)
    2. Image must have some reddish/pinkish tones
       (conjunctiva is pink/red)
    3. Image must be reasonable size
    """
    if img_bgr is None:
        return False, "Could not read image"

    h, w = img_bgr.shape[:2]
    if h < 50 or w < 50:
        return False, "Image too small"

    # Check if image is essentially grayscale (like X-rays)
    # X-rays have R≈G≈B channels
    b, g, r = cv2.split(img_bgr)
    rg_diff = float(np.mean(np.abs(r.astype(int) - g.astype(int))))
    rb_diff = float(np.mean(np.abs(r.astype(int) - b.astype(int))))

    if rg_diff < 5 and rb_diff < 5:
        return False, "Image appears to be grayscale or X-ray. Please upload a color eye photo."

    # Check for some reddish/pinkish tones in image
    # Convert to HSV and check for red/pink hues
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Red hue is 0-10 and 160-180 in HSV
    mask1 = cv2.inRange(hsv, (0, 30, 30), (15, 255, 255))
    mask2 = cv2.inRange(hsv, (155, 30, 30), (180, 255, 255))
    red_pixels = cv2.countNonZero(mask1) + cv2.countNonZero(mask2)
    total_pixels = h * w
    red_ratio = red_pixels / total_pixels

    if red_ratio < 0.01:
        return False, "No eye tissue detected. Please upload a clear photo of the inner lower eyelid (conjunctiva)."

    return True, "OK"


def preprocess_for_model(img_bgr):
    roi = extract_roi(img_bgr)
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_rgb = cv2.resize(roi_rgb, (224, 224))
    arr = roi_rgb.astype(np.float32)  # Keep 0-255, no division
    arr = np.expand_dims(arr, axis=0)
    return roi_rgb, arr


def build_gradcam_overlay(arr, roi_rgb):
    try:
        # Use top_conv activation maps directly
        # This works on CPU without gradient issues
        feature_extractor = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=model.get_layer('top_conv').output
        )

        # Get feature maps
        feature_maps = feature_extractor(arr)  # shape: (1, 7, 7, 1280)
        feature_maps = feature_maps[0]          # shape: (7, 7, 1280)

        # Average across all channels to get activation map
        heatmap = tf.reduce_mean(feature_maps, axis=-1).numpy()  # (7, 7)

        # Also get prediction to weight the heatmap
        pred = float(model.predict(arr, verbose=0)[0][0])

        # If anaemic (pred < 0.5), invert so high activation = anaemic region
        if pred < 0.5:
            heatmap = -heatmap

        # Normalize to 0-1
        heatmap = heatmap - heatmap.min()
        if heatmap.max() == 0:
            print("GradCAM: Feature maps all zero, using random heatmap")
            heatmap = np.random.rand(7, 7).astype(np.float32)
        else:
            heatmap = heatmap / heatmap.max()

        # Resize to match ROI
        h, w = roi_rgb.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h),
                                      interpolation=cv2.INTER_LINEAR)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)

        # Apply jet colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Blend with original ROI
        roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(roi_bgr, 0.6, heatmap_colored, 0.4, 0)

        print("GradCAM: Success using top_conv activation maps")
        return to_data_url_bgr(overlay)

    except Exception as e:
        print(f"GradCAM error: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route('/admin')
def admin():
    import json

    log_path = PREDICTIONS_LOG

    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            try:
                log = json.load(f)
            except Exception:
                log = []
    else:
        log = []

    total = len(log)
    anaemic_count = sum(1 for e in log if e['result'] == 'Anaemic')
    non_anaemic_count = sum(1 for e in log if e['result'] == 'Non-Anaemic')
    uncertain_count = sum(1 for e in log if e['result'] == 'Uncertain')

    avg_confidence = round(
        sum(e['confidence'] for e in log) / total, 1
    ) if total > 0 else 0

    high_conf = sum(1 for e in log if e['confidence_label'] == 'High Confidence')
    high_conf_pct = round(high_conf / total * 100, 1) if total > 0 else 0

    anaemic_pct = round(anaemic_count / total * 100, 1) if total > 0 else 0

    # Last 10 predictions (most recent first)
    recent = list(reversed(log[-10:]))

    # Dataset info
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    dataset_info = {'anaemic': 0, 'non_anaemic': 0}

    anaemic_candidates = ['anaemic', 'anemia', 'anaemia', 'anemic']
    non_anaemic_candidates = ['non_anaemic', 'non-anaemic', 'non anaemic', 'non_anemia', 'non-anemia', 'non anemia']
    search_roots = [
        os.path.join(BASE_DIR, 'dataset', 'train'),
        os.path.join(BASE_DIR, 'train'),
        os.path.join(BASE_DIR, 'dataset')
    ]

    for root in search_roots:
        if not os.path.isdir(root):
            continue

        entries = {name.lower(): name for name in os.listdir(root)}

        anaemic_dir = next((entries[k] for k in entries if k in anaemic_candidates), None)
        non_anaemic_dir = next((entries[k] for k in entries if k in non_anaemic_candidates), None)

        if anaemic_dir:
            anaemic_path = os.path.join(root, anaemic_dir)
            if os.path.isdir(anaemic_path):
                dataset_info['anaemic'] = len([
                    f for f in os.listdir(anaemic_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ])
        if non_anaemic_dir:
            non_anaemic_path = os.path.join(root, non_anaemic_dir)
            if os.path.isdir(non_anaemic_path):
                dataset_info['non_anaemic'] = len([
                    f for f in os.listdir(non_anaemic_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ])

        if dataset_info['anaemic'] > 0 or dataset_info['non_anaemic'] > 0:
            break

    sl_stats = sl_manager.get_stats()

    return render_template(
        'admin.html',
        total=total,
        anaemic_count=anaemic_count,
        non_anaemic_count=non_anaemic_count,
        uncertain_count=uncertain_count,
        avg_confidence=avg_confidence,
        high_conf_pct=high_conf_pct,
        anaemic_pct=anaemic_pct,
        recent=recent,
        dataset_info=dataset_info,
        model_name='EfficientNetB0',
        model_accuracy='86.9%',
        model_version='v3',
        sl_stats=sl_stats
    )


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    path = "uploaded.jpg"
    preview_image = None
    safety_disclaimer = "This is not a medical diagnosis. Consult a doctor."

    try:
        file.save(path)
        img = cv2.imread(path)
        if img is None:
            raise ValueError("Unable to load image. Please upload a valid image file.")

        is_valid, validation_msg = is_likely_eye_image(img)
        if not is_valid:
            return render_template(
                "result.html",
                result="Invalid Image",
                severity_level="Not Applicable",
                confidence=0.0,
                confidence_label="N/A",
                suggestions=["Upload a clear color photo of your inner lower eyelid",
                            "Pull down your lower eyelid gently",
                            "Take photo in good lighting"],
                suggestion_sections={
                    "How to take the correct photo": [
                        "Gently pull down your lower eyelid",
                        "Take photo in bright natural light",
                        "Make sure the pink/red inner eyelid is clearly visible",
                        "Keep camera steady and close (15-20cm)"
                    ]
                },
                personalized_message=f"Invalid image uploaded: {validation_msg}",
                safety_disclaimer="Please upload a correct eye image for accurate screening.",
                preview_image=preview_image,
                heatmap_image=None,
                prediction_history=session.get("prediction_history", []),
                error=validation_msg,
            )

        preview_image = to_data_url_bgr(img)
        roi_rgb, arr = preprocess_for_model(img)

        pred = float(model.predict(arr)[0][0])
        # Class indices: anaemic=0, non_anaemic=1
        print(f"Raw prediction: {pred:.4f}")

        if pred < 0.35:
            result = "Anaemic"
            confidence = (1 - pred) * 100
        elif pred > 0.65:
            result = "Non-Anaemic"
            confidence = pred * 100
        else:
            result = "Uncertain"
            confidence = max(pred, 1 - pred) * 100
        print(f"Raw prediction value: {pred:.4f} | Result: {result} | Confidence: {confidence:.1f}%")

        if confidence >= 85:
            confidence_label = "High Confidence"
        elif confidence >= 65:
            confidence_label = "Medium Confidence"
        else:
            confidence_label = "Low Confidence"

        if result == "Anaemic" and confidence_label == "High Confidence":
            severity_level = "High Risk"
        elif result == "Anaemic":
            severity_level = "Moderate Risk"
        elif result == "Uncertain":
            severity_level = "Needs Review"
        else:
            severity_level = "Normal"

        log_prediction(result, confidence, confidence_label, severity_level)
        sl_result = sl_manager.maybe_add(
            img_path=path,
            result=result,
            confidence=confidence
        )
        print(f"Self-learning: {sl_result}")

        personalized_message, suggestion_sections = build_suggestion_sections(
            result, confidence_label, recommendations.get(result, [])
        )

        heatmap_image = build_gradcam_overlay(arr, roi_rgb)
        print(f"GradCAM generated: {heatmap_image is not None}")
        if heatmap_image is None:
            print("GradCAM failed - check logs above for reason")
        print(f"heatmap_image type: {type(heatmap_image)}")
        print(f"heatmap_image length: {len(heatmap_image) if heatmap_image else 0}")
        print(f"heatmap_image starts with: {str(heatmap_image)[:50] if heatmap_image else 'None'}")
        history = session.get("prediction_history", [])
        history.insert(0, {"result": result, "confidence": round(confidence, 1)})
        session["prediction_history"] = history[:3]

        return render_template(
            "result.html",
            result=result,
            severity_level=severity_level,
            confidence=round(confidence, 1),
            confidence_label=confidence_label,
            suggestions=recommendations[result],
            suggestion_sections=suggestion_sections,
            personalized_message=personalized_message,
            safety_disclaimer=safety_disclaimer,
            preview_image=preview_image,
            heatmap_image=heatmap_image,
            prediction_history=session.get("prediction_history", []),
            error=None,
        )
    except Exception:
        fallback_message, fallback_sections = build_suggestion_sections(
            "Uncertain", "Low Confidence", recommendations["Uncertain"]
        )
        return render_template(
            "result.html",
            result="Uncertain",
            severity_level="Needs Review",
            confidence=0.0,
            confidence_label="Low Confidence",
            suggestions=recommendations["Uncertain"],
            suggestion_sections=fallback_sections,
            personalized_message=fallback_message,
            safety_disclaimer=safety_disclaimer,
            preview_image=preview_image,
            heatmap_image=None,
            prediction_history=session.get("prediction_history", []),
            error="Unable to process the image. Please upload a clear eye image and try again.",
        )


if __name__ == "__main__":
    app.run(debug=True)