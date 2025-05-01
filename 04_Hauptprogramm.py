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
        # Initialisierung der notwendigen Variablen für das Tracking
        self.previous_gray = None  # Zur Speicherung des letzten Graustufenbildes
        self.distance_threshold = 280  # Maximale Distanz für das Clustering
        self.clusters = {}  # Aktive Cluster
        self.next_id = 1  # ID für den nächsten neuen Cluster
        self.max_missing_frames = 120  # Frames ohne Aktualisierung bevor ein Cluster entfernt wird
        self.still_threshold = 0.9  # Bewegungsschwelle für das Tracking
        self.alpha = 0.2  # Glättungsfaktor für die Bounding Box
        self.min_points_per_cluster = 10  # Mindestanzahl von Punkten pro Cluster


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
        lk_params = dict(winSize=(80, 80), maxLevel=1, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
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

# Punkte, Leben und Level
score = 0
level = 1
lives = 8

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
paddle2_y = SCREEN_HEIGHT - 50
paddle2_x = SCREEN_WIDTH // 2 - paddle_width // 2
smoothed_paddle2_x = paddle2_x  # Für geglättete Bewegung

# Kamera- oder Videoquelle
source = "webcam"  # Wechsel zwischen Webcam und Video
if source == "webcam":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture("Videos/links_rechts_abstand.mp4")

if not cap.isOpened():
    print("Fehler: Videoquelle konnte nicht geöffnet werden.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

# Tracker-Objekt
motion_tracker = LucasKanadeTracker()

running = True
alpha = 0.3  # Glättungsfaktor für Exponential Moving Average (EMA)

# Variable zur Steuerung der Sichtbarkeit des zweiten Paddles
second_paddle_visible = False

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

    # Spieler 1 Steuerung (Paddle 1) - Cluster für den ersten Spieler finden
    if len(clusters) > 0:
        first_player_points = np.array(clusters[list(clusters.keys())[0]]["points"])
        if first_player_points.size > 0:
            avg_x = int(np.mean(first_player_points[:, 0]))
            mirrored_x = SCREEN_WIDTH - avg_x
            smoothed_paddle_x = alpha * mirrored_x + (1 - alpha) * smoothed_paddle_x
            paddle_x = max(0, min(SCREEN_WIDTH - paddle_width, smoothed_paddle_x - paddle_width // 2))

    # Spieler 2 Steuerung (Paddle 2) - Überprüfen, ob ein zweiter Cluster vorhanden ist
    second_paddle_visible = len(clusters) > 1 and np.array(clusters[list(clusters.keys())[1]]["points"]).size > 0
    if second_paddle_visible:
        second_player_points = np.array(clusters[list(clusters.keys())[1]]["points"])
        avg_x2 = int(np.mean(second_player_points[:, 0]))
        mirrored_x2 = SCREEN_WIDTH - avg_x2
        smoothed_paddle2_x = alpha * mirrored_x2 + (1 - alpha) * smoothed_paddle2_x
        paddle2_x = max(0, min(SCREEN_WIDTH - paddle_width, smoothed_paddle2_x - paddle_width // 2))

    # Ballbewegung aktualisieren
    ball_x += ball_speed_x
    ball_y += ball_speed_y

    # Kollision mit den Wänden
    if ball_x - ball_radius <= 0 or ball_x + ball_radius >= SCREEN_WIDTH:
        ball_speed_x *= -1
    if ball_y - ball_radius <= 0:
        ball_speed_y *= -1

    # Kollision mit Paddle 1
    if (paddle_y <= ball_y + ball_radius <= paddle_y + paddle_height and
        paddle_x <= ball_x <= paddle_x + paddle_width):
        ball_speed_y *= -1
        ball_speed_x += random.choice([-1, 1]) * (random.random() > 0.5)
        score += 1
        # Level-Up nach 5 Punkten
        if score % 5 == 0:
            level += 1
            ball_speed_x *= 1.2
            ball_speed_y *= 1.2

    # Kollision mit Paddle 2
    if second_paddle_visible and (paddle2_y <= ball_y + ball_radius <= paddle2_y + paddle_height and
                                  paddle2_x <= ball_x <= paddle2_x + paddle_width):
        ball_speed_y *= -1
        ball_speed_x += random.choice([-1, 1]) * (random.random() > 0.5)
        score += 1
        # Level-Up nach 5 Punkten
        if score % 5 == 0:
            level += 1
            ball_speed_x *= 1.2
            ball_speed_y *= 1.2

    # Ball fällt unten raus
    if ball_y - ball_radius > SCREEN_HEIGHT:
        lives -= 1
        ball_x = SCREEN_WIDTH // 2
        ball_y = SCREEN_HEIGHT // 2
        ball_speed_x = random.choice([-4, 4])
        ball_speed_y = -4
        if lives <= 0:
            print("Game Over!")
            running = False

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

    # Zeichnen des ersten Paddles (immer sichtbar)
    pygame.draw.rect(screen, (0, 255, 0), (paddle_x, paddle_y, paddle_width, paddle_height))

    # Zeichnen des zweiten Paddles (nur wenn sichtbar)
    if second_paddle_visible:
        pygame.draw.rect(screen, (0, 0, 255), (paddle2_x, paddle2_y, paddle_width, paddle_height))

    # Ball zeichnen
    pygame.draw.circle(screen, (255, 0, 0), (ball_x, ball_y), ball_radius)

    # Anzeige von Score, Level und Leben
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    level_text = font.render(f"Level: {level}", True, (255, 255, 255))
    lives_text = font.render(f"Lives: {lives}", True, (255, 255, 255))
    screen.blit(score_text, (10, 40))
    screen.blit(level_text, (10, 10))
    screen.blit(lives_text, (10, 70))

    pygame.display.update()
    clock.tick(fps)

pygame.quit()
cap.release()
cv2.destroyAllWindows()
