import cv2
import pygame
import numpy as np
import random
from scipy.optimize import linear_sum_assignment

# Pygame Fenstergröße
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN = [SCREEN_WIDTH, SCREEN_HEIGHT]

class LucasKanadeTracker:
    def __init__(self):
        # Initialisiert die Variablen für das Tracking-System
        self.previous_gray = None  # Letztes Graustufenbild für den optischen Fluss
        self.distance_threshold = 170  # Maximale Distanz zwischen Punkten eines Clusters
        self.clusters = {}  # Aktive Tracking-Cluster
        self.next_id = 1  # Start-ID für die nächste Cluster-Zuweisung
        self.max_missing_frames = 150  # Anzahl Frames, bevor ein inaktiver Cluster entfernt wird
        self.still_threshold = 0.8  # Bewegungsschwelle für das Tracking
        self.alpha = 0.2  # Glättungsfaktor für die Bounding Box
        self.min_points_per_cluster = 10  # Mindestanzahl von Punkten, die einen Cluster bilden


    def update_velocity_and_predict(self):
        # Aktualisiere die Geschwindigkeit und schätze die nächste Position der Cluster
        for cluster_id, cluster_data in self.clusters.items():
            if len(cluster_data.get('previous_centers', [])) >= 2:
                # Berechne neue Geschwindigkeit basierend auf den letzten zwei Zentren
                prev_centers = cluster_data['previous_centers']
                new_velocity = (prev_centers[-1][0] - prev_centers[-2][0], prev_centers[-1][1] - prev_centers[-2][1])
                cluster_data['velocity'] = new_velocity
                # Schätze die nächste Position
                predicted_center = (prev_centers[-1][0] + new_velocity[0], prev_centers[-1][1] + new_velocity[1])
                cluster_data['predicted_center'] = predicted_center
            else:
                cluster_data['velocity'] = (0, 0)
                cluster_data['predicted_center'] = cluster_data['center']

    def detect_and_track(self, frame):
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_gray = cv2.GaussianBlur(current_gray, (5, 5), 0)
        self.update_velocity_and_predict()

        if self.previous_gray is None:
            self.previous_gray = current_gray
            return [], {}

        points_prev = self.detect_features(current_gray)
        if points_prev is None or not points_prev:
            self.previous_gray = current_gray
            return [], {}

        # Lucas-Kanade-Parameter
        lk_params = dict(winSize=(90, 90), maxLevel=1, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        points_prev = np.float32(points_prev).reshape(-1, 1, 2)
        points_next, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, current_gray, points_prev, None, **lk_params)

        # Punkte filtern
        valid_points_next = points_next[status == 1]
        motion_threshold = self.still_threshold
        valid_motion = np.linalg.norm(valid_points_next - points_prev[status == 1], axis=1) > motion_threshold
        valid_points_next = valid_points_next[valid_motion]

        # Punkte gruppieren (Clustering)
        new_clusters = self.cluster_points(valid_points_next)

        # Cluster mit IDs verbinden
        self.update_clusters(new_clusters)

        self.previous_gray = current_gray
        return valid_points_next, self.clusters

    def detect_features(self, gray_image):
        # Reduzierte Bildgröße für schnellere Berechnung
        resized_gray = cv2.resize(gray_image, (gray_image.shape[1] // 2, gray_image.shape[0] // 2))

        # Sobel-Filter zur Gradientenberechnung
        grad_x = cv2.Sobel(resized_gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(resized_gray, cv2.CV_64F, 0, 1, ksize=3)

        # Gradientenmagnitude und -richtung
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Schwellenwert, um starke Gradienten zu finden
        _, binary_mask = cv2.threshold(magnitude, 75, 255, cv2.THRESH_BINARY)

        # Konturen finden, um potenzielle Eckpunkte zu identifizieren
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        points = []
        for contour in contours:
            # Berechne den Schwerpunkt der Kontur
            M = cv2.moments(contour)
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"]) * 2  # Skalierung zurücksetzen
                cY = int(M["m01"] / M["m00"]) * 2
                points.append((cX, cY))

        return points

    def cluster_points(self, points):
        clusters = []
        for point in points:
            added_to_cluster = False
            for cluster in clusters:
                # Prüfe, ob Punkt in die Nähe eines bestehenden Clusters fällt
                if np.linalg.norm(np.array(cluster) - point, axis=1).min() < self.distance_threshold:
                    cluster.append(point)
                    added_to_cluster = True
                    break
            if not added_to_cluster:
                # Erstelle neuen Cluster
                clusters.append([point])
        return clusters

    def update_clusters(self, new_clusters):
        # Filtere Cluster mit zu wenigen Punkten
        new_clusters = [cluster for cluster in new_clusters if len(cluster) >= self.min_points_per_cluster]

        # Berechne die Mittelpunkte der neuen Cluster
        new_centers = [np.mean(cluster, axis=0) for cluster in new_clusters]

        # Aktuelle Cluster-Mittelpunkte
        existing_centers = [data["center"] for data in self.clusters.values()]

        # Kostenmatrix basierend auf Distanzen
        cost_matrix = np.zeros((len(existing_centers), len(new_centers)))
        for i, center1 in enumerate(existing_centers):
            for j, center2 in enumerate(new_centers):
                cost_matrix[i, j] = np.linalg.norm(center1 - center2)

        # Zuordnung der neuen Cluster zu bestehenden IDs
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_new = set()
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < self.distance_threshold:
                cluster_id = list(self.clusters.keys())[i]
                velocity = (new_centers[j] - self.clusters[cluster_id]["center"])

                # Berechnung der Bounding Box
                min_x, min_y = np.min(new_clusters[j], axis=0)
                max_x, max_y = np.max(new_clusters[j], axis=0)
                previous_bbox = self.clusters[cluster_id].get("bbox", (min_x, min_y, max_x, max_y))
                smoothed_bbox = [
                    self.alpha * current + (1 - self.alpha) * previous
                    for current, previous in zip([min_x, min_y, max_x, max_y], previous_bbox)
                ]

                self.clusters[cluster_id] = {
                    "points": new_clusters[j],
                    "center": new_centers[j],
                    "age": 0,
                    "velocity": velocity,
                    "bbox": smoothed_bbox
                }
                assigned_new.add(j)

        # Neue Cluster, die keine bestehende ID haben
        for j, cluster in enumerate(new_clusters):
            if j not in assigned_new:
                min_x, min_y = np.min(cluster, axis=0)
                max_x, max_y = np.max(cluster, axis=0)
                self.clusters[self.next_id] = {
                    "points": cluster,
                    "center": new_centers[j],
                    "age": 0,
                    "velocity": (0, 0),
                    "bbox": [min_x, min_y, max_x, max_y]
                }
                self.next_id += 1

        # Alter von nicht aktualisierten Clustern erhöhen und Position extrapolieren
        to_remove = []
        for cluster_id, data in self.clusters.items():
            if cluster_id not in [list(self.clusters.keys())[i] for i in row_ind]:
                data["age"] += 1
                if data["age"] > self.max_missing_frames:
                    to_remove.append(cluster_id)

        for cluster_id in to_remove:
            del self.clusters[cluster_id]

# Main Program
pygame.init()
screen = pygame.display.set_mode(SCREEN)
pygame.display.set_caption("Spiel mit Lucas-Kanade-Tracker und eigener Eckendetektion")

# Hinzufügen dieser Zeile für die Schriftart
font = pygame.font.SysFont("arial", 24)

fps = 30  # Framerate
clock = pygame.time.Clock()

# Ball- und Platte-Parameter
ball_radius = 15
ball_x = SCREEN_WIDTH // 2
ball_y = SCREEN_HEIGHT // 2
ball_speed_x = random.choice([-4, 4])
ball_speed_y = -4

paddle_width = 150
paddle_height = 20
paddle_y = SCREEN_HEIGHT - 50
paddle_x = SCREEN_WIDTH // 2 - paddle_width // 2
smoothed_paddle_x = paddle_x  # Für geglättete Bewegung

# Für Spieler 2:
paddle2_y = SCREEN_HEIGHT - 100
paddle2_x = SCREEN_WIDTH // 2 - paddle_width // 2
smoothed_paddle2_x = paddle2_x  # Für geglättete Bewegung

# Kamera- oder Videoquelle
source = "webca"  # Wechsel zwischen Webcam und Video
if source == "webcam":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture("Videos/kleinesObjekt.mp4")

if not cap.isOpened():
    print("Fehler: Videoquelle konnte nicht geöffnet werden.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

# Tracker-Objekt
motion_tracker = LucasKanadeTracker()

running = True
alpha = 0.3  # Glättungsfaktor für Exponential Moving Average (EMA)

# Hauptspiel-Schleife
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    ret, frame = cap.read()
    if not ret:
        break

    points, clusters = motion_tracker.detect_and_track(frame)

    # Bild umwandeln und auf Bildschirm zeichnen
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgRGB = np.rot90(imgRGB)
    game_frame = pygame.surfarray.make_surface(imgRGB).convert()
    screen.blit(game_frame, (0, 0))

    # Cluster anzeigen
    for cluster_id, data in clusters.items():
        cluster_points = [(int(p[0]), int(p[1])) for p in data["points"]]
        bbox = data["bbox"]
        min_x, min_y, max_x, max_y = map(int, bbox)
        pygame.draw.rect(screen, (255, 0, 0), (SCREEN_WIDTH - max_x, min_y, max_x - min_x, max_y - min_y), 2)
        text = font.render(f"ID: {cluster_id}", True, (255, 255, 255))
        screen.blit(text, (SCREEN_WIDTH - max_x, min_y - 20))

    pygame.display.update()
    clock.tick(fps)

pygame.quit()
cap.release()
cv2.destroyAllWindows()
