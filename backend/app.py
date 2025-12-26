import base64
import time
import cv2
import numpy as np
import face_recognition
from flask import Flask, request, jsonify
from flask_cors import CORS
from io import BytesIO
from PIL import Image

from faces_recognition.hnsw_manager import FaceSearchEngine

# --- Server setup ---
app = Flask(__name__)
# Allow frontend from any origin (local or deployed)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- Initialize HNSW ---
print("[INFO] Dang khoi dong Server va nap du lieu...")
search_engine = FaceSearchEngine()

try:
    # Load Mongo data and build the HNSW index during startup
    search_engine.load_data_and_build_index()
    print("[SUCCESS] Server da san sang phuc vu tai http://localhost:8000")
except Exception as e:
    print(f"[ERROR] LOI NGHIEM TRONG: Khong the khoi dong HNSW. Chi tiet: {e}")

# ----------------- Brute-force helper -----------------
def brute_force_search(query_vector, threshold=0.5):
    """
    Brute-force search on MongoDB:
    - Iterate every feature_vector
    - Compute L2 distance
    - Return the doc with the smallest distance
    """
    collection = search_engine.collection  # configured inside FaceSearchEngine
    cursor = collection.find({"feature_vector": {"$exists": True}})

    best_doc = None
    best_dist = None

    q = np.array(query_vector, dtype=np.float32)

    for doc in cursor:
        vec = doc.get("feature_vector")
        if not isinstance(vec, list):
            continue

        v = np.array(vec, dtype=np.float32)
        if v.shape[0] != q.shape[0]:
            continue

        d = np.linalg.norm(q - v)

        if best_dist is None or d < best_dist:
            best_dist = d
            best_doc = doc

    if best_doc is None or best_dist is None:
        return {"status": "unknown", "distance": None, "info": {}}

    status = "found" if best_dist <= threshold else "unknown"

    info = {
        "MSSV": best_doc.get("MSSV", "Unknown"),
        "Ten": best_doc.get("Ten", "Unknown"),
    }

    return {
        "status": status,
        "distance": float(best_dist),
        "info": info,
    }

# --- Helper: decode Base64 image ---
def decode_base64_image(base64_string):
    """Convert a webcam Base64 string into an RGB OpenCV image."""
    try:
        # Strip off any data URI header (e.g., "data:image/jpeg;base64,...")
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        img_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # OpenCV uses BGR; face_recognition expects RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_rgb
    except Exception as e:
        print(f"[ERROR] Loi decode anh: {e}")
        return None

# --- Helper: crop face to Base64 ---
def crop_face_to_base64(image_rgb, top, right, bottom, left):
    try:
        # Crop by coordinates [y:y+h, x:x+w]
        face_image = image_rgb[top:bottom, left:right]

        # Convert to PIL Image
        pil_img = Image.fromarray(face_image)

        # Save into buffer
        buffered = BytesIO()
        pil_img.save(buffered, format="JPEG")

        # Encode base64
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"[ERROR] Loi crop anh: {e}")
        return ""
# --- API 1: Upload image file ---
# Called from frontend: POST /recognize_image?mode=hnsw|bruteforce
@app.route("/recognize_image", methods=["POST"])
def search_by_file():
    start_time = time.time()

    mode = request.args.get("mode", "hnsw").lower()
    if mode not in {"hnsw", "bruteforce"}:
        mode = "hnsw"

    if "file" not in request.files:
        return jsonify({"error": "Vui long gui kem file anh (key='file')"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Chua chon file"}), 400

    try:
        # Read the uploaded image file directly
        image = face_recognition.load_image_file(file)

        # 1. Find face locations (HOG for speed)
        face_locations = face_recognition.face_locations(image, model="hog")

        if len(face_locations) == 0:
            elapsed_ms = (time.time() - start_time) * 1000
            return jsonify({"faces": [], "mode": mode, "elapsed_ms": elapsed_ms}), 200

        # 2. Encode faces
        face_encodings = face_recognition.face_encodings(image, face_locations)

        results = []

        # 3. Loop through detected faces
        for i, query_vector in enumerate(face_encodings):
            # Choose the search algorithm
            if mode == "bruteforce":
                search_result = brute_force_search(query_vector)
            else:
                search_result = search_engine.search_face(query_vector)

            if not search_result:
                continue

            top, right, bottom, left = face_locations[i]

            # Crop the face image
            crop_b64 = crop_face_to_base64(image, top, right, bottom, left)

            face_data = {
                "student_id": search_result.get("info", {}).get(
                    "MSSV", "Unknown"
                )
                if search_result.get("status") == "found"
                else "Unknown",
                "name": search_result.get("info", {}).get(
                    "Ten", "Unknown"
                )
                if search_result.get("status") == "found"
                else "Unknown",
                "distance": search_result.get("distance", 0),
                "box": [top, right, bottom, left],
                "crop_image": crop_b64,
                "mode": mode,
            }
            results.append(face_data)

        elapsed_ms = (time.time() - start_time) * 1000
        return jsonify({"faces": results, "mode": mode, "elapsed_ms": elapsed_ms}), 200

    except Exception as e:
        print(f"[ERROR] Loi xu ly file: {e}")
        return jsonify({"error": str(e)}), 500

# --- API 2: Realtime recognition (webcam) ---
# Called from frontend: POST /recognize_frame?mode=hnsw|bruteforce
@app.route("/recognize_frame", methods=["POST"])
def search_by_base64():
    start_time = time.time()

    mode = request.args.get("mode", "hnsw").lower()
    if mode not in {"hnsw", "bruteforce"}:
        mode = "hnsw"

    # 1. Parse JSON body
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "Thieu du lieu 'image'"}), 400

    base64_str = data["image"]

    # 2. Decode Base64 into RGB image
    image_rgb = decode_base64_image(base64_str)
    if image_rgb is None:
        return jsonify({"error": "Anh base64 loi"}), 400

    try:
        # 3. Find face locations (HOG)
        face_locations = face_recognition.face_locations(image_rgb, model="hog")

        if len(face_locations) == 0:
            elapsed_ms = (time.time() - start_time) * 1000
            return jsonify({"faces": [], "mode": mode, "elapsed_ms": elapsed_ms}), 200

        # 4. Encode faces (128D embeddings)
        face_encodings = face_recognition.face_encodings(image_rgb, face_locations)

        results = []

        # 5. Loop and search
        for i, face_encoding in enumerate(face_encodings):
            # Choose the search algorithm
            if mode == "bruteforce":
                search_result = brute_force_search(face_encoding)
            else:
                search_result = search_engine.search_face(face_encoding)

            if not search_result:
                continue

            top, right, bottom, left = face_locations[i]
            crop_b64 = crop_face_to_base64(image_rgb, top, right, bottom, left)

            face_data = {
                "student_id": search_result.get("info", {}).get(
                    "MSSV", "Unknown"
                )
                if search_result.get("status") == "found"
                else "Unknown",
                "name": search_result.get("info", {}).get(
                    "Ten", "Unknown"
                )
                if search_result.get("status") == "found"
                else "Unknown",
                "distance": search_result.get("distance", 0),
                "box": [top, right, bottom, left],
                "crop_image": crop_b64,
                "mode": mode,
            }
            results.append(face_data)

        elapsed_ms = (time.time() - start_time) * 1000
        return jsonify({"faces": results, "mode": mode, "elapsed_ms": elapsed_ms}), 200

    except Exception as e:
        print(f"[ERROR] Loi realtime: {e}")
        return jsonify({"error": str(e)}), 500
# --- Run app ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
