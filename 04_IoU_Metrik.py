import cv2
import numpy as np
import json
from scipy.optimize import linear_sum_assignment

# --------------------------------------------------------------------------
# Lucas-Kanade-Tracker mit eigener Eckendetektion und stabiler ID-Verwaltung
# --------------------------------------------------------------------------

class LucasKanadeTracker:
    def __init__(self):
        self.previous_gray = None
        self.distance_threshold = 150  # Distanzschwelle für Clustering
        self.clusters = {}  # Cluster mit IDs
        self.next_id = 1  # Nächste verfügbare ID
        self.max_missing_frames = 50  # Anzahl Frames, die ein Cluster "unsichtbar" bleiben kann
        self.still_threshold = 0.5  # Schwelle für minimale Bewegung
        self.alpha = 0.3  # Glättungsfaktor für Bounding Box
        self.min_points_per_cluster = 10  # Mindestanzahl an Punkten pro Cluster

    def detect_and_track(self, frame):
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)  # Bild glätten

        if self.previous_gray is None:
            self.previous_gray = current_gray
            return [], {}  # Keine Punkte und keine Cluster

        points_prev = self.detect_features(current_gray)

        if points_prev is not None:
            # Lucas-Kanade Optical Flow
            lk_params = dict(winSize=(80, 80), maxLevel=1,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            points_prev = np.float32(points_prev).reshape(-1, 1, 2)  # Format für Lucas-Kanade
            points_next, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, current_gray, points_prev, None, **lk_params)

            # Bewegung und Gültigkeit prüfen
            valid_points_next = points_next[status == 1]
            motion_threshold = self.still_threshold
            valid_motion = np.linalg.norm(valid_points_next - points_prev[status == 1], axis=1) > motion_threshold
            valid_points_next = valid_points_next[valid_motion]

            # Clustering und IDs aktualisieren
            new_clusters = self.cluster_points(valid_points_next)
            self.update_clusters(new_clusters)
            self.previous_gray = current_gray

            return valid_points_next, self.clusters
        else:
            self.previous_gray = current_gray
            return [], self.clusters

    def detect_features(self, gray_image):
        resized_gray = cv2.resize(gray_image, (gray_image.shape[1] // 2, gray_image.shape[0] // 2))
        grad_x = cv2.Sobel(resized_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(resized_gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        _, binary_mask = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        points = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"]) * 2
                cY = int(M["m01"] / M["m00"]) * 2
                points.append((cX, cY))
        return points

    def cluster_points(self, points):
        clusters = []
        for point in points:
            added_to_cluster = False
            for cluster in clusters:
                if np.linalg.norm(np.array(cluster) - point, axis=1).min() < self.distance_threshold:
                    cluster.append(point)
                    added_to_cluster = True
                    break
            if not added_to_cluster:
                clusters.append([point])
        return clusters

    def update_clusters(self, new_clusters):
        new_clusters = [cluster for cluster in new_clusters if len(cluster) >= self.min_points_per_cluster]
        new_centers = [np.mean(cluster, axis=0) for cluster in new_clusters]

        existing_centers = [data["center"] for data in self.clusters.values()]
        cost_matrix = np.zeros((len(existing_centers), len(new_centers)))
        for i, center1 in enumerate(existing_centers):
            for j, center2 in enumerate(new_centers):
                cost_matrix[i, j] = np.linalg.norm(center1 - center2)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_new = set()
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.distance_threshold:
                cluster_id = list(self.clusters.keys())[i]
                min_x, min_y = np.min(new_clusters[j], axis=0)
                max_x, max_y = np.max(new_clusters[j], axis=0)
                self.clusters[cluster_id] = {
                    "points": new_clusters[j],
                    "center": new_centers[j],
                    "bbox": [min_x, min_y, max_x, max_y]
                }
                assigned_new.add(j)

        for j, cluster in enumerate(new_clusters):
            if j not in assigned_new:
                min_x, min_y = np.min(cluster, axis=0)
                max_x, max_y = np.max(cluster, axis=0)
                self.clusters[self.next_id] = {
                    "points": cluster,
                    "center": new_centers[j],
                    "bbox": [min_x, min_y, max_x, max_y]
                }
                self.next_id += 1

# --------------------------------------------------------------------------
# Tracking-Daten speichern
# --------------------------------------------------------------------------

def save_tracking_data(video_path, output_json):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Fehler: Video {video_path} konnte nicht geöffnet werden.")
        return

    tracker = LucasKanadeTracker()
    tracking_results = {}
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        _, clusters = tracker.detect_and_track(frame)

        tracking_results[frame_idx] = [
            {
                "id": cluster_id,
                "bbox": {
                    "x_min": int(data["bbox"][0]),
                    "y_min": int(data["bbox"][1]),
                    "x_max": int(data["bbox"][2]),
                    "y_max": int(data["bbox"][3])
                }
            }
            for cluster_id, data in clusters.items()
        ]
        frame_idx += 1

    cap.release()

    with open(output_json, "w") as f:
        json.dump(tracking_results, f, indent=4)
    print(f"Tracking-Daten gespeichert in {output_json}")

# --------------------------------------------------------------------------
# IoU berechnen und Tracking auswerten
# --------------------------------------------------------------------------

def calculate_iou(box1, box2):
    x1 = max(box1["x_min"], box2["x_min"])
    y1 = max(box1["y_min"], box2["y_min"])
    x2 = min(box1["x_max"], box2["x_max"])
    y2 = min(box1["y_max"], box2["y_max"])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1["x_max"] - box1["x_min"]) * (box1["y_max"] - box1["y_min"])
    box2_area = (box2["x_max"] - box2["x_min"]) * (box2["y_max"] - box2["y_min"])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

def evaluate_tracking(tracking_json, groundtruth_json):
    with open(tracking_json, "r") as f:
        tracking_data = json.load(f)
    with open(groundtruth_json, "r") as f:
        groundtruth_data = json.load(f)

    frame_results = {}
    for frame_idx, gt_boxes in groundtruth_data.items():
        if frame_idx not in tracking_data:
            continue

        tracked_boxes = tracking_data[frame_idx]
        frame_iou = []

        for gt_box in gt_boxes:
            gt_bbox = gt_box["bbox"]
            best_iou = 0
            for tracked_box in tracked_boxes:
                iou = calculate_iou(gt_bbox, tracked_box["bbox"])
                best_iou = max(best_iou, iou)

            frame_iou.append(best_iou)

        frame_results[frame_idx] = sum(frame_iou) / len(frame_iou) if frame_iou else 0

    overall_iou = sum(frame_results.values()) / len(frame_results) if frame_results else 0
    print(f"Durchschnittliche IoU: {overall_iou:.2f}")

# --------------------------------------------------------------------------
# Hauptprogramm
# --------------------------------------------------------------------------

video_path = "Videos/links_rechts_abstand.mp4"
tracking_output = "tracking_results.json"
groundtruth_path = "ground_truth.json"

# Tracking-Daten speichern
save_tracking_data(video_path, tracking_output)

# Tracking auswerten
evaluate_tracking(tracking_output, groundtruth_path)
