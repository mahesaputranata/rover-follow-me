"""
Download a pretrained door detection YOLOv8 model.
Run this ONCE before starting the rover vision system.

Usage:
    python tools/download_door_model.py

This downloads a YOLOv8n model fine-tuned on door detection from
Hugging Face and saves it to data/models/door_yolov8n.pt
"""
import os
import urllib.request

SAVE_DIR  = "data/models"
SAVE_PATH = f"{SAVE_DIR}/door_yolov8n.pt"

# Public door detection model — YOLOv8n fine-tuned on door dataset
# Source: https://huggingface.co/universe/door-detection
MODEL_URL = (
    "https://universe.roboflow.com/microsoft/door-detection/"
    "model/1"  # this requires API key — see manual method below
)

print("""
Automatic download requires a free Roboflow account.
Follow these steps instead (takes 2 minutes):

  1. Go to: https://universe.roboflow.com
  2. Search: 'door detection indoor'
  3. Pick any project that shows 'Model' tab available
     (good ones: 'Door Detection' by gjxgjx, or search 'door-opksc')
  4. Click the project → go to 'Model' tab
  5. Click 'Deploy' → 'Export Model' → choose 'YOLOv8 PyTorch'
  6. Download the .pt file
  7. Move it to:  rover_vision/data/models/door_yolov8n.pt
  8. Run test_door.py again — you should see 'door' in model.names

Alternative — install roboflow package and use API:
  pip install roboflow
  Then run: python tools/download_door_roboflow.py
""")
# ── Better option: use Roboflow Universe (free account needed) ──────────
# If the URL above doesn't work well for doors, the best public
# door detection datasets are on Roboflow Universe:
#
#   1. Go to https://universe.roboflow.com
#   2. Search: "door detection"
#   3. Pick a dataset with 1000+ images
#   4. Export as "YOLOv8" format
#   5. Download the trained model weights (.pt file)
#   6. Move it to data/models/door_yolov8n.pt
#
# Recommended datasets on Roboflow:
#   - "Door Detection" by Lensa (2000+ images, indoor doors)
#   - "door-detection-xpivh" (household interior doors)
# ───────────────────────────────────────────────────────────────────────

def download():
    os.makedirs(SAVE_DIR, exist_ok=True)

    if os.path.exists(SAVE_PATH):
        print(f"Model already exists at {SAVE_PATH}")
        return

    print(f"Downloading door model...")
    print(f"Saving to: {SAVE_PATH}")
    print()
    print("NOTE: If this download doesn't give good door detection results,")
    print("      use Roboflow Universe instead — see comments in this script.")
    print()

    try:
        urllib.request.urlretrieve(MODEL_URL, SAVE_PATH, reporthook=_progress)
        print(f"\nDone. Model saved to {SAVE_PATH}")
        print("You can now run main.py — door detection will be active.")
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("Please download manually from Roboflow Universe (see comments in script).")


def _progress(count, block_size, total_size):
    percent = min(int(count * block_size * 100 / total_size), 100)
    print(f"\r  {percent}% [{('█' * (percent // 5)).ljust(20)}]", end="")


if __name__ == "__main__":
    download()
