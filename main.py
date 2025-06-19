import cv2, numpy as np, mediapipe as mp
from insightface.app import FaceAnalysis
from pathlib import Path
from collections import deque
from datetime import datetime

# ──────────────── CONFIG ───────────────────────────────────────────
DB_DIR         = Path("known_faces")  
SIM_THRESHOLD  = 0.55                
MARGIN         = 0.10                 
CAM_WIDTH      = 640           
MAX_RECENT     = 30                  
DEBUG          = False               

# ──────────────── INITIALISE MODELS ────────────────────────────────
print("[BOOT] Loading MediaPipe BlazeFace …")
mp_fd = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)

print("[BOOT] Loading ArcFace backbone …")
arc = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
arc.prepare(ctx_id=0, det_size=(640, 640))

# ──────────────── GALLERY BUILDER ─────────────────────────────────
def build_gallery(folder: Path) -> dict[str, np.ndarray]:
    temp: dict[str, list[np.ndarray]] = {}
    for img_path in folder.glob("*.*"):
        img = cv2.imread(str(img_path))
        if img is None: continue
        faces = arc.get(img)
        if not faces: continue
        emb = faces[0].normed_embedding
        name = img_path.stem.split("_")[0].lower()
        temp.setdefault(name, []).append(emb)
        print(f"[LOAD] {img_path.name} ➜ {name}")

    if not temp:
        raise RuntimeError("Gallery empty. Add images to 'known_faces/'.")

    gallery: dict[str, np.ndarray] = {}
    for name, vecs in temp.items():
        centroid = np.mean(np.stack(vecs), axis=0)
        centroid /= np.linalg.norm(centroid)  # renormalise
        gallery[name] = centroid
        print(f"[CENTROID] {name}: {len(vecs)} imgs → vector ready")
    return gallery

gallery = build_gallery(DB_DIR)
recent  = deque(maxlen=MAX_RECENT)

# ──────────────── HELPERS ─────────────────────────────────────────
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def match_identity(emb: np.ndarray) -> tuple[str, float]:
    best_name, best_sim = "Unknown", 0.0
    for name, vec in gallery.items():
        sim = cosine(emb, vec)
        if sim > best_sim:
            best_name, best_sim = name, sim
    if best_sim < SIM_THRESHOLD:
        best_name = "Unknown"
    return best_name, best_sim

def annotate(frame, box, name, sim):
    x1, y1, x2, y2 = map(int, box)
    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, f"{name} {sim:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ──────────────── MAIN LOOP ───────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Webcam not accessible.")

print("[RUN] Vion Phase-1 online — Q to quit • S to save live frame")
while True:
    ok, frame = cap.read()
    if not ok: break

    # optional downscale
    if frame.shape[1] > CAM_WIDTH:
        scale = CAM_WIDTH / frame.shape[1]
        frame = cv2.resize(frame, None, fx=scale, fy=scale)

    results = mp_fd.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.detections:
        for det in results.detections:
            bb = det.location_data.relative_bounding_box
            x1 = int((bb.xmin - MARGIN) * frame.shape[1])
            y1 = int((bb.ymin - MARGIN) * frame.shape[0])
            x2 = int((bb.xmin + bb.width + MARGIN)  * frame.shape[1])
            y2 = int((bb.ymin + bb.height+ MARGIN) * frame.shape[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 <= x1 or y2 <= y1: continue

            crop_rgb = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            faces = arc.get(crop_rgb)
            if not faces: continue

            emb = faces[0].normed_embedding
            name, sim = match_identity(emb)

            if DEBUG: print(f"[DBG] best={name} sim={sim:.3f}")

            sig = f"{name}:{sim:.2f}"
            if sig not in recent:
                recent.append(sig)
                print(f"{datetime.now():%H:%M:%S} | {sig}")

            annotate(frame, (x1, y1, x2, y2), name, sim)

    cv2.imshow("Vion • Hybrid FR", frame)
    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), ord('Q')):
        break
    if key in (ord('s'), ord('S')):              # save live shot -> gallery
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn = DB_DIR / f"live_{ts}.jpg"
        cv2.imwrite(str(fn), frame)
        print(f"[CAP] {fn.name} saved — restart Vion to ingest")

cap.release()
cv2.destroyAllWindows()
print("[STOP] Vion session ended.")
