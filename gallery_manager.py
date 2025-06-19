import json, os, numpy as np, cv2
from pathlib import Path
from insightface.app import FaceAnalysis
from config import GALLERY_JSON, DATA_DIR

class GalleryManager:
    def __init__(self, arc: FaceAnalysis, seed_dir: Path):
        self.arc = arc
        self.gallery = {}               # {name: vector}
        os.makedirs(DATA_DIR, exist_ok=True)
        if Path(GALLERY_JSON).exists():
            self._load_json()
        else:
            self._bootstrap(seed_dir)
            self._save_json()

    # ---------- public API ----------
    def get_vector(self, name): 
        return self.gallery.get(name)
    def all_vectors(self):       
        return self.gallery.items()

    def add_image(self, img_bgr, name):
        faces = self.arc.get(img_bgr)
        if not faces: return False
        emb = faces[0].normed_embedding
        self.gallery[name] = emb / np.linalg.norm(emb)
        self._save_json()
        return True

    def remove(self, name):
        if name in self.gallery:
            del self.gallery[name]
            self._save_json()
            return True
        return False

    # ---------- internals -----------
    def _bootstrap(self, seed_dir):
        for p in seed_dir.glob("*.*"):
            img = cv2.imread(str(p))
            if img is None: continue
            faces = self.arc.get(img)
            if faces:
                emb = faces[0].normed_embedding
                key = p.stem.split("_")[0].lower()
                self.gallery.setdefault(key, []).append(emb)

        # centroid per person
        for k, vecs in self.gallery.items():
            v = np.mean(np.stack(vecs), axis=0)
            self.gallery[k] = v / np.linalg.norm(v)

    def _save_json(self):
        j = {k: v.tolist() for k, v in self.gallery.items()}
        with open(GALLERY_JSON, "w") as f: json.dump(j, f)

    def _load_json(self):
        with open(GALLERY_JSON) as f:
            j = json.load(f)
        self.gallery = {k: np.asarray(v, dtype=float) for k, v in j.items()}
