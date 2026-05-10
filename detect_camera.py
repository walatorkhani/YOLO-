import cv2
import os
import time
import csv
import winsound
from datetime import datetime
from ultralytics import YOLO
from collections import Counter

# YOLO model
model = YOLO("yolov8n.pt")

# webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Impossible d'ouvrir la webcam.")

print("System started... q = quitter")

# folders
os.makedirs("captures", exist_ok=True)

# alert classes
ALERT_CLASSES = {67, 15}

# timers
last_capture = 0
last_alert = 0
COOLDOWN_CAPTURE = 5
ALERT_COOLDOWN = 2

# CSV log
csv_file = open("log.csv", "a", newline="", buffering=1)
writer = csv.writer(csv_file)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model(frame, verbose=False)
    boxes = results[0].boxes

    annotated = frame.copy()

    detected_classes = set()
    now = time.time()

    if boxes is not None and len(boxes) > 0:

        for b in boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            cls_id = int(b.cls)
            conf = float(b.conf)

            label = model.names[cls_id]

            detected_classes.add(cls_id)

            is_alert = cls_id in ALERT_CLASSES
            color = (0, 0, 255) if is_alert else (0, 255, 0)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            cv2.putText(
                annotated,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

            writer.writerow([time.time(), label, conf])

    counts = Counter(int(b.cls) for b in boxes) if boxes is not None else Counter()

    y = 40
    for cls_id, n in counts.items():
        cv2.putText(
            annotated,
            f"{model.names[cls_id]} : {n}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )
        y += 25

    # ALERT SYSTEM
    if ALERT_CLASSES & detected_classes:
        if now - last_alert > ALERT_COOLDOWN:

            cv2.putText(
                annotated,
                "ALERTE !",
                (200, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                4
            )

            cv2.putText(
                annotated,
                "ohreb ya fatima",
                (200, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )

            try:
                winsound.Beep(1200, 150)
            except:
                pass

            last_alert = now

    # AUTO capture
    if len(detected_classes) > 0 and (now - last_capture > COOLDOWN_CAPTURE):

        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        path = f"captures/{ts}.jpg"

        cv2.imwrite(path, annotated)

        print(f"Capture saved: {path}")

        last_capture = now

    cv2.imshow("SMART AI RED BOX SYSTEM", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
csv_file.close()
cv2.destroyAllWindows()

print("System stopped")