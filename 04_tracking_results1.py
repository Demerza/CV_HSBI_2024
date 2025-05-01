import cv2
import numpy as np
import json
import os

class LucasKanadeTracker:
    def __init__(self):
        # Initialisierung des Trackers mit Basisparametern
        self.previous_gray = None  # Speicher für das vorherige Graustufenbild
        self.distance_threshold = 280  # Distanzschwelle für das Clustering
        self.clusters = {}  # Speicher für aktive Cluster
        self.next_id = 1  # Startwert für Cluster-IDs
        self.max_missing_frames = 90  # Maximale Anzahl von Frames ohne Erkennung
        self.still_threshold = 0.6  # Schwelle für Bewegungsdetektion
        self.alpha = 0.2  # Glättungsfaktor für die Bounding Box
        self.min_points_per_cluster = 9  # Minimale Anzahl von Punkten pro Cluster

    def update_velocity_and_predict(self):
        # Aktualisiert Geschwindigkeit und prognostiziert zukünftige Positionen der Cluster
        for cluster_id, cluster_data in self.clusters.items():
            if len(cluster_data.get('previous_centers', [])) >= 2:
                # Berechnet neue Geschwindigkeit und prognostizierte Position
                prev_centers = cluster_data['previous_centers']
                new_velocity = (prev_centers[-1][0] - prev_centers[-2][0], prev_centers[-1][1] - prev_centers[-2][1])
                cluster_data['velocity'] = new_velocity
                predicted_center = (prev_centers[-1][0] + new_velocity[0], prev_centers[-1][1] + new_velocity[1])
                cluster_data['predicted_center'] = predicted_center
            else:
                # Keine ausreichende Daten für eine Prognose
                cluster_data['velocity'] = (0, 0)
                cluster_data['predicted_center'] = cluster_data['center']

    def detect_and_track(self, frame):
        # Konvertiert das aktuelle Frame in Graustufen und weichzeichnet es
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)
        self.update_velocity_and_predict()

        if self.previous_gray is None:
            self.previous_gray = current_gray
            return [], {}

        points_prev = self.detect_features(current_gray)
        if points_prev is None or points_prev.size == 0:
            self.previous_gray = current_gray
            return [], {}

        return self.process_detection(points_prev, current_gray)

    def detect_features(self, gray_image):
        # Detektiert Ecken im Bild als Startpunkte für das Tracking
        corners = cv2.goodFeaturesToTrack(gray_image, maxCorners=100, qualityLevel=0.01, minDistance=10)
        return np.array([]) if corners is None else corners.reshape(-1, 2)

    def process_detection(self, points_prev, current_gray):
        # Verarbeitet die Detektion und berechnet den optischen Fluss
        lk_params = dict(winSize=(80, 80), maxLevel=1, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        points_prev = np.float32(points_prev).reshape(-1, 1, 2)
        points_next, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, current_gray, points_prev, None, **lk_params)
        valid_points_next = points_next[status == 1]
        motion_threshold = self.still_threshold
        valid_motion = np.linalg.norm(valid_points_next - points_prev[status == 1], axis=1) > motion_threshold
        valid_points_next = valid_points_next[valid_motion]
        return valid_points_next, self.update_clusters(valid_points_next)

    def update_clusters(self, valid_points_next):
        # Aktualisiert oder erstellt neue Cluster für jede detektierte Bewegung
        for point in valid_points_next:
            self.clusters[self.next_id] = {'center': point, 'bbox': (point[0]-25, point[1]-25, point[0]+25, point[1]+25)}
            self.next_id += 1
        return self.clusters

def process_video(video_path, output_json, tracker):
    # Verarbeitet ein Video und speichert die Tracking-Ergebnisse in einer JSON-Datei
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Fehler: Video {video_path} konnte nicht geöffnet werden.")
        return

    frame_idx = 0
    tracking_results = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        _, clusters = tracker.detect_and_track(frame)
        tracking_results[frame_idx] = [
            {"x": int(data['center'][0]), "y": int(data['center'][1]), "width": 50, "height": 50}
            for cluster_id, data in clusters.items()
        ]
        frame_idx += 1

    cap.release()

    with open(output_json, "w") as f:
        json.dump(tracking_results, f, indent=4)
    print(f"Tracking-Daten gespeichert in {output_json}")

# Initialisiere Tracker und starte die Verarbeitung
tracker = LucasKanadeTracker()
video_path = os.path.join(os.getcwd(), "Videos", "links_rechts_abstand.mp4")
output_json = "tracking_results.json"
process_video(video_path, output_json, tracker)
