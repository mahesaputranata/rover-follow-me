"""
Door Image Collector
Captures 10 photos per run with a countdown between each.
Run this script multiple times, changing angle/lighting/distance each time.

Usage:
    python collect_door_images.py

Controls:
    SPACE  → capture photo manually (skips countdown)
    Q      → quit early
"""

import cv2
import os
import time
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
SAVE_DIR       = "door_dataset/images"
PHOTOS_PER_RUN = 10
COUNTDOWN_SECS = 3       # seconds between auto-captures
CAMERA_INDEX   = 0
# ──────────────────────────────────────────────────────────────────────────────

def get_run_number(save_dir):
    """Count existing photos to determine run number."""
    if not os.path.exists(save_dir):
        return 1
    existing = [f for f in os.listdir(save_dir) if f.endswith(".jpg")]
    return (len(existing) // PHOTOS_PER_RUN) + 1

def draw_overlay(frame, countdown, captured, total_existing, run_num):
    h, w = frame.shape[:2]

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # Run info
    cv2.putText(frame, f"Run #{run_num}  |  Session: {captured}/{PHOTOS_PER_RUN}  |  Total saved: {total_existing + captured}",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Countdown or capturing
    if countdown > 0:
        msg   = f"Auto-capture in  {countdown}s"
        color = (0, 200, 255)
    else:
        msg   = "CAPTURING..."
        color = (0, 255, 100)

    cv2.putText(frame, msg, (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Bottom hint
    cv2.putText(frame, "SPACE = capture now   Q = quit",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # Progress bar
    progress = int((captured / PHOTOS_PER_RUN) * w)
    cv2.rectangle(frame, (0, h - 6), (progress, h), (0, 255, 100), -1)

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        return

    # Count already existing photos
    existing     = [f for f in os.listdir(SAVE_DIR) if f.endswith(".jpg")]
    total_existing = len(existing)
    run_num      = get_run_number(SAVE_DIR)

    print(f"\n{'='*50}")
    print(f"  Door Image Collector")
    print(f"  Run #{run_num} — capturing {PHOTOS_PER_RUN} photos")
    print(f"  Total already saved: {total_existing}")
    print(f"  Saving to: {SAVE_DIR}/")
    print(f"{'='*50}")
    print(f"\n  TIP for this run — try one of these setups:")

    tips = [
        "Front-facing, normal room light",
        "Slight left angle",
        "Slight right angle",
        "From further away (full door visible)",
        "From close up (partial door)",
        "Low angle (rover height ~50cm)",
        "With lights off / dim lighting",
        "Door partially open",
        "Door fully open",
        "Different door in the house",
    ]
    tip_index = (run_num - 1) % len(tips)
    print(f"  → Suggested: {tips[tip_index]}\n")
    print("  Press SPACE to start...\n")

    # Wait for user to press space before starting
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        cv2.putText(display, "Point camera at a door",
                    (20, frame.shape[0]//2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        cv2.putText(display, "Press SPACE to start capturing",
                    (20, frame.shape[0]//2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        cv2.imshow("Door Collector", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            break
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    # Capture loop
    captured   = 0
    last_time  = time.time()

    while captured < PHOTOS_PER_RUN:
        ret, frame = cap.read()
        if not ret:
            break

        now       = time.time()
        elapsed   = now - last_time
        countdown = max(0, COUNTDOWN_SECS - elapsed)

        display = frame.copy()
        draw_overlay(display, int(countdown) + 1 if countdown > 0 else 0,
                     captured, total_existing, run_num)
        cv2.imshow("Door Collector", display)

        key = cv2.waitKey(1) & 0xFF

        # Manual capture on SPACE or auto-capture when countdown hits 0
        should_capture = (key == ord(' ')) or (elapsed >= COUNTDOWN_SECS)

        if should_capture:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            filename  = f"door_run{run_num:02d}_{captured+1:02d}_{timestamp}.jpg"
            filepath  = os.path.join(SAVE_DIR, filename)
            cv2.imwrite(filepath, frame)
            captured  += 1
            last_time  = time.time()
            print(f"  [{captured}/{PHOTOS_PER_RUN}] Saved: {filename}")

            # Flash green feedback
            flash = frame.copy()
            cv2.rectangle(flash, (0, 0), (frame.shape[1], frame.shape[0]),
                          (0, 255, 100), 8)
            cv2.imshow("Door Collector", flash)
            cv2.waitKey(150)

        if key == ord('q'):
            print("\n  Quit early.")
            break

    cap.release()
    cv2.destroyAllWindows()

    total_now = total_existing + captured
    print(f"\n{'='*50}")
    print(f"  Run #{run_num} complete — {captured} photos saved")
    print(f"  Total dataset size: {total_now} photos")
    print(f"  {'▓' * (total_now // 5)}{'░' * max(0, 20 - total_now // 5)}  {total_now}/100")
    if total_now < 100:
        print(f"  {100 - total_now} more photos to go — run the script again!")
        print(f"  Next tip → {tips[run_num % len(tips)]}")
    else:
        print(f"  100+ photos collected! Ready to label and train.")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()