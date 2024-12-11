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
        if self.previous_frame is None:
            self.previous_frame = current_frame
            return None

        gray_prev = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        frame_diff = cv2.absdiff(gray_prev, gray_curr)
        _, binary_mask = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)

        kernel = np.ones((2, 2), np.uint8)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        self.previous_frame = current_frame
        return binary_mask

    def track_object(self, binary_mask):
        if binary_mask is None:
            return None

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 2000:
            return None

        x, y, w, h = cv2.boundingRect(largest_contour)
        return x, y, x + w, y + h

# --------------------------------------------------------------------------
# -- Main Program
# --------------------------------------------------------------------------

pygame.init()
screen = pygame.display.set_mode(SCREEN)
pygame.display.set_caption("Tracking with Stability Metric")

fps = 30
clock = pygame.time.Clock()

ball_radius = 15
ball_x = SCREEN_WIDTH // 2
ball_y = SCREEN_HEIGHT // 2
ball_speed_x = random.choice([-4, 4])
ball_speed_y = -4

paddle_width = 150
paddle_height = 20
paddle_x = SCREEN_WIDTH // 2 - paddle_width // 2
paddle_y = SCREEN_HEIGHT - 50

score = 0
lives = 3

font = pygame.font.SysFont("arial", 24)

# Frame-wise stability metric
stability_errors = []

source = "webca"
if source == "webcam":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture("Neutral_jacke_langsam.mp4")

if not cap.isOpened():
    print("Fehler: Videoquelle konnte nicht geöffnet werden.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

tracker = MotionTracker()
previous_bbox = None

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    ret, frame = cap.read()
    if not ret:
        break

    motion_mask = tracker.detect_motion(frame)
    bbox = tracker.track_object(motion_mask) if motion_mask is not None else None

    if bbox:
        if previous_bbox is not None:
            min_x = int(0.7 * previous_bbox[0] + 0.3 * bbox[0])
            min_y = int(0.7 * previous_bbox[1] + 0.3 * bbox[1])
            max_x = int(0.7 * previous_bbox[2] + 0.3 * bbox[2])
            max_y = int(0.7 * previous_bbox[3] + 0.3 * bbox[3])
            bbox = (min_x, min_y, max_x, max_y)

            # Berechnung der Bounding Box Stabilität
            stability = abs(min_x - previous_bbox[0]) + abs(min_y - previous_bbox[1])
            stability_errors.append(stability)

        previous_bbox = bbox

        min_x, min_y, max_x, max_y = bbox

        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        box_width = 200
        box_height = 300

        min_x = max(0, center_x - box_width // 2)
        max_x = min(SCREEN_WIDTH, center_x + box_width // 2)
        min_y = max(0, center_y - box_height // 2)
        max_y = min(SCREEN_HEIGHT, center_y + box_height // 2)

        bbox = (min_x, min_y, max_x, max_y)

        screen_width = screen.get_width()
        mirrored_min_x = screen_width - max_x
        mirrored_max_x = screen_width - min_x
        paddle_x = (mirrored_min_x + mirrored_max_x) // 2 - paddle_width // 2
        paddle_x = max(0, min(SCREEN_WIDTH - paddle_width, paddle_x))

    mean_stability = sum(stability_errors) / len(stability_errors) if stability_errors else 0

    ball_x += ball_speed_x
    ball_y += ball_speed_y

    if ball_x - ball_radius <= 0 or ball_x + ball_radius >= SCREEN_WIDTH:
        ball_speed_x *= -1

    if ball_y - ball_radius <= 0:
        ball_speed_y *= -1

    if (
        paddle_y <= ball_y + ball_radius <= paddle_y + paddle_height and
        paddle_x <= ball_x <= paddle_x + paddle_width
    ):
        ball_speed_y *= -1
        ball_y = paddle_y - ball_radius
        score += 1

    if ball_y - ball_radius > SCREEN_HEIGHT:
        lives -= 1
        ball_x = SCREEN_WIDTH // 2
        ball_y = SCREEN_HEIGHT // 2
        ball_speed_x = random.choice([-4, 4])
        ball_speed_y = -4

        if lives <= 0:
            print(f"Game Over! Mean Stability: {mean_stability:.2f}")
            running = False

    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgRGB = np.rot90(imgRGB)
    game_frame = pygame.surfarray.make_surface(imgRGB).convert()
    screen.blit(game_frame, (0, 0))

    if bbox:
        min_x, min_y, max_x, max_y = bbox
        screen_width = screen.get_width()
        mirrored_min_x = screen_width - max_x
        mirrored_max_x = screen_width - min_x
        pygame.draw.rect(screen, (255, 0, 0), (mirrored_min_x, min_y, mirrored_max_x - mirrored_min_x, max_y - min_y), 2)
        text = font.render("ID: 1", True, (255, 0, 0))
        screen.blit(text, (mirrored_min_x, min_y - 20))

    pygame.draw.rect(screen, (0, 255, 0), (paddle_x, paddle_y, paddle_width, paddle_height))
    pygame.draw.circle(screen, (255, 0, 0), (ball_x, ball_y), ball_radius)

    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    lives_text = font.render(f"Lives: {lives}", True, (255, 255, 255))
    stability_text = font.render(f"Stability: {mean_stability:.2f}", True, (255, 255, 255))
    screen.blit(score_text, (10, 10))
    screen.blit(lives_text, (10, 40))
    screen.blit(stability_text, (10, 70))

    if motion_mask is not None:
        fg_mask_resized = cv2.resize(motion_mask, (320, 180))
        fg_mask_surface = pygame.surfarray.make_surface(np.rot90(fg_mask_resized))
        screen.blit(fg_mask_surface, (SCREEN_WIDTH - 330, 10))

    pygame.display.update()
    clock.tick(fps)

pygame.quit()
cap.release()
cv2.destroyAllWindows()
