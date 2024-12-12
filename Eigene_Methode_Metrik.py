import numpy as np
import cv2
import pygame
import random

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN = [SCREEN_WIDTH, SCREEN_HEIGHT]

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

        if bbox and previous_bbox:
            # Berechne die MCE zwischen der aktuellen und der vorherigen Bounding Box
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
tracker1 = MotionTracker()
tracker2 = MotionTracker()

# Video 1 verarbeiten
video1_path = "Neutral_jacke_langsam.mp4"  # Ersetze mit deinem ersten Video
mce_video1 = process_video(video1_path, tracker1)
avg_mce_video1 = np.mean(mce_video1)

# Video 2 verarbeiten
video2_path = "Neutral_jacke_schnell.mp4"  # Ersetze mit deinem zweiten Video
mce_video2 = process_video(video2_path, tracker2)
avg_mce_video2 = np.mean(mce_video2)

# Ergebnisse ausgeben
print(f"Durchschnittliche MCE für Video 1 ({video1_path}): {avg_mce_video1}")
print(f"Durchschnittliche MCE für Video 2 ({video2_path}): {avg_mce_video2}")

# Vergleich der Ergebnisse
if avg_mce_video1 < avg_mce_video2:
    print("Video 1 hat stabileres Tracking.")
elif avg_mce_video1 > avg_mce_video2:
    print("Video 2 hat stabileres Tracking.")
else:
    print("Beide Videos haben gleichwertiges Tracking.")
