import numpy as np
import cv2

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

# --------------------------------------------------------------------------
# -- Motion Tracker
# --------------------------------------------------------------------------

class MotionTracker:
    def __init__(self):
        self.previous_frame = None

    def detect_motion(self, current_frame, threshold=8):
        """
        Verwendet Frame-Differenzen zur Bewegungserkennung.
        """
        if self.previous_frame is None:
            self.previous_frame = current_frame
            return None

        # Frames in Graustufen konvertieren
        gray_prev = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Frame-Differenz berechnen
        frame_diff = cv2.absdiff(gray_prev, gray_curr)

        # Schwellenwert anwenden
        _, binary_mask = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)

        # Noise entfernen
        kernel = np.ones((2, 2), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        # Speichere das aktuelle Frame für den nächsten Schritt
        self.previous_frame = current_frame
        return binary_mask

    def track_object(self, binary_mask):
        """
        Findet die größte Bewegung und gibt eine Bounding Box zurück.
        """
        if binary_mask is None:
            return None

        # Konturen finden
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None

        # Größte Kontur auswählen
        largest_contour = max(contours, key=cv2.contourArea)

        # Mindestgröße überprüfen, um kleine Bewegungen zu ignorieren
        if cv2.contourArea(largest_contour) < 2000:  # Mindestfläche für Körper
            return None

        # Bounding Box berechnen
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x, y, x + w, y + h

# --------------------------------------------------------------------------
# -- Helper Functions
# --------------------------------------------------------------------------

def calculate_mce(box1, box2):
    """
    Berechnet die Matching Cost Error (MCE) zwischen zwei Bounding Boxen.
    """
    center1 = ((box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2)
    center2 = ((box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2)
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def process_video(video_path, tracker):
    """
    Führt die Verarbeitung eines Videos durch und berechnet MCE für Bounding Boxen.
    """
    cap = cv2.VideoCapture(video_path)
    previous_bbox = None
    mce_values = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Bewegungserkennung und Tracking
        motion_mask = tracker.detect_motion(frame)
        bbox = tracker.track_object(motion_mask) if motion_mask is not None else None

        # Glättung der Bounding Box
        if bbox and previous_bbox:
            min_x = int(0.7 * previous_bbox[0] + 0.3 * bbox[0])
            min_y = int(0.7 * previous_bbox[1] + 0.3 * bbox[1])
            max_x = int(0.7 * previous_bbox[2] + 0.3 * bbox[2])
            max_y = int(0.7 * previous_bbox[3] + 0.3 * bbox[3])
            bbox = (min_x, min_y, max_x, max_y)

            # Berechne die MCE zwischen der aktuellen und der geglätteten Bounding Box
            mce = calculate_mce(previous_bbox, bbox)
            mce_values.append(mce)

        if bbox:
            previous_bbox = bbox

    cap.release()
    return mce_values

# --------------------------------------------------------------------------
# -- Main Program
# --------------------------------------------------------------------------

# Tracker erstellen
trackers = [MotionTracker() for _ in range(6)]

# Video-Pfade definieren
video_paths = [
    "Videos/ArmeSchleudern_links_rechts.mp4",
    "Videos/link_rechts_hinten_vorne.mp4",
    "Videos/links_rechts_dynamisch.mp4",
    "Videos/rein_stehen_arme.mp4",
    "Videos/schnell_links_rechts.mp4",
    "Videos/StartimVideo_links_rechts.mp4"
]

# MCE für jedes Video berechnen
results = []
for i, video_path in enumerate(video_paths):
    mce_values = process_video(video_path, trackers[i])
    avg_mce = np.mean(mce_values)
    results.append((video_path, avg_mce))

# Ergebnisse ausgeben
for video_path, avg_mce in results:
    print(f"Durchschnittliche MCE für {video_path}: {avg_mce}")

# Vergleich der Ergebnisse
best_video = min(results, key=lambda x: x[1])
print(f"Das Video mit dem stabilsten Tracking ist: {best_video[0]} mit einer MCE von {best_video[1]}")
