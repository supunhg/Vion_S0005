from pathlib import Path
from collections import deque
from datetime import datetime
import cv2, numpy as np, mediapipe as mp
from insightface.app import FaceAnalysis

from config           import *        # tweak constants here
from gallery_manager  import GalleryManager
from context_engine   import ContextEngine


# ───────────── Initialise models ─────────────
mp_fd = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5)

arc = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
arc.prepare(ctx_id=0, det_size=(640, 640))

gallery_mgr = GalleryManager(arc, Path("known_faces"))
ctx_engine  = ContextEngine()


# ───────────── helpers ─────────────
def cosine(a, b) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def best_match(vec):
    best_name, best_sim = "Unknown", 0.0
    for name, gvec in gallery_mgr.all_vectors():
        sim = cosine(vec, gvec)
        if sim > best_sim:
            best_name, best_sim = name, sim
    return best_name, best_sim

def annotate(img, box, label, colour):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
    cv2.putText(img, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)


# ───────────── runtime state ─────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH , CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

recent_console = deque(maxlen=40)   # prevents spam
latest_bbox    = None               # for 'A' hot-key
latest_frame   = None
frame_idx      = 0

print("[RUN] Vion Phase-2 • Q quit • A add current face")
while True:
    ok, frame = cap.read()
    if not ok:
        print("[WARN] camera frame grab failed"); break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mp_fd.process(rgb)
    detections = res.detections or []

    unknown_present = False

    for det in detections:
        # ------- bounding box with margin -------
        bb = det.location_data.relative_bounding_box
        x1 = int((bb.xmin - MARGIN)          * CAM_WIDTH)
        y1 = int((bb.ymin - MARGIN)          * CAM_H)
        x2 = int((bb.xmin + bb.width + MARGIN)  * CAM_WIDTH)
        y2 = int((bb.ymin + bb.height + MARGIN) * CAM_H)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(CAM_WIDTH, x2), min(CAM_H, y2)

        # save for hot-key
        latest_bbox  = (x1, y1, x2, y2)
        latest_frame = frame.copy()

        # ------- per-face embedding EVERY frame -------
        crop = cv2.resize(frame[y1:y2, x1:x2], (160, 160))
        faces = arc.get(crop[..., ::-1])            # BGR→RGB
        name, sim = ("Unknown", 0.0)
        if faces:
            name, sim = best_match(faces[0].normed_embedding)

        unknown_present |= (name == "Unknown")

        colour = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        annotate(frame, (x1, y1, x2, y2), f"{name} {sim:.2f}", colour)

        sig = f"{name}:{sim:.2f}"
        if sig not in recent_console:
            recent_console.append(sig)
            print(f"{datetime.now():%H:%M:%S} | {sig}")

    # -------- context HUD --------
    ctx = ctx_engine.evaluate(len(detections), unknown_present)
    cv2.putText(frame, f"Risk:{ctx['risk']}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Vion Phase-2", frame)
    key = cv2.waitKey(1) & 0xFF

    # -------- hot-keys --------
    if key in (ord('q'), ord('Q')):
        break

    if key in (ord('a'), ord('A')):
        if latest_bbox is None or latest_frame is None:
            print("[ADD] No face captured yet — wait for detection.")
        else:
            label = input("Label for this face (Enter = cancel): ").strip().lower()
            if label:
                x1, y1, x2, y2 = latest_bbox
                crop_bgr = latest_frame[y1:y2, x1:x2]
                ok = gallery_mgr.add_image(crop_bgr, label)
                print(f"[GALLERY] Added '{label}' ✅" if ok
                      else "[GALLERY] Add failed (no clear face)")
            latest_bbox = latest_frame = None

    frame_idx += 1

# -------- teardown --------
cap.release()
cv2.destroyAllWindows()
print("[STOP] Vion Phase-2 session ended.")
