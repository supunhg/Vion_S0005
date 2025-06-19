# Central place for tweakables
BASE_THRESHOLD     = 0.55                # default cosine cutoff
DAY_NIGHT_SWITCH   = 18                  # 18:00 regarded as "night"
NIGHT_DELTA        = +0.07               # stricter at night
MULTIFACE_DELTA    = +0.05               # stricter if >1 face
UNKNOWN_DELTA      = +0.07               # stricter if an unknown seen
EMBED_INTERVAL     = 2                   # run ArcFace every N frames
MARGIN             = 0.10                # crop expansion
CAM_WIDTH, CAM_H   = 640, 480
DATA_DIR           = "data"
GALLERY_JSON       = f"{DATA_DIR}/gallery.json"
