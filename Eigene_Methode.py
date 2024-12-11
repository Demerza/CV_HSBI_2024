import numpy as np
import cv2
import pygame

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN = [SCREEN_WIDTH, SCREEN_HEIGHT]

# --------------------------------------------------------------------------
# -- Motion Tracker
# --------------------------------------------------------------------------

class MotionTracker:
    def __init__(self):
        self.previous_frame = None

    def detect_motion(self, current_frame, threshold=10):
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
        kernel = np.ones((5, 5), np.uint8)
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

        # Mindestgröße überprüfen, um Rauschen zu ignorieren
        if cv2.contourArea(largest_contour) < 1500:
            return None

        # Bounding Box berechnen
        x, y, w, h = cv2.boundingRect(largest_contour)
        return x, y, x + w, y + h


# --------------------------------------------------------------------------
# -- Main Program
# --------------------------------------------------------------------------

# Pygame initialisieren
pygame.init()
screen = pygame.display.set_mode(SCREEN)
pygame.display.set_caption("Single Object Tracking")

# Init game clock
fps = 30
clock = pygame.time.Clock()

# Kamera- oder Videoquelle öffnen
source = "webcam"  # Ändere auf "video" für eine Videodatei
if source == "webcam":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture("Neutral_jacke_langsam.mp4")  # Passe den Dateinamen an

if not cap.isOpened():
    print("Fehler: Videoquelle konnte nicht geöffnet werden.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

# Tracker-Objekt erstellen
tracker = MotionTracker()

# Variable für die vorherige Bounding Box
previous_bbox = None

# Haupt-Loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    # Aktuelles Frame lesen
    ret, frame = cap.read()
    if not ret:
        break

    # Bewegungserkennung
    motion_mask = tracker.detect_motion(frame)

    # Objektverfolgung
    bbox = tracker.track_object(motion_mask) if motion_mask is not None else None

    # Glättung der Bounding Box (Pufferung)
    if bbox:
        if previous_bbox is not None:
            # Glätte die Bounding Box (Mittelwert zwischen vorheriger und aktueller Position)
            min_x = int(0.7 * previous_bbox[0] + 0.3 * bbox[0])
            min_y = int(0.7 * previous_bbox[1] + 0.3 * bbox[1])
            max_x = int(0.7 * previous_bbox[2] + 0.3 * bbox[2])
            max_y = int(0.7 * previous_bbox[3] + 0.3 * bbox[3])
            bbox = (min_x, min_y, max_x, max_y)

        # Speichere die aktuelle Bounding Box
        previous_bbox = bbox
    elif previous_bbox is not None:
        # Verwende die vorherige Bounding Box, wenn keine neue erkannt wurde
        bbox = previous_bbox

    # Zentriere die Bounding Box
    if bbox:
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2

        box_width = 150  # Feste Breite der Bounding Box
        box_height = 300  # Feste Höhe der Bounding Box

        min_x = max(0, center_x - box_width // 2)
        max_x = min(SCREEN_WIDTH, center_x + box_width // 2)
        min_y = max(0, center_y - box_height // 2)
        max_y = min(SCREEN_HEIGHT, center_y + box_height // 2)

        bbox = (min_x, min_y, max_x, max_y)

    # Frame für Pygame vorbereiten
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgRGB = np.rot90(imgRGB)
    game_frame = pygame.surfarray.make_surface(imgRGB).convert()
    screen.blit(game_frame, (0, 0))

    # Bounding Box zeichnen
    if bbox:
        min_x, min_y, max_x, max_y = bbox

        # Bildschirmbreite abrufen
        screen_width = screen.get_width()

        # Berechne die gespiegelten Bounding Box-Koordinaten
        mirrored_min_x = screen_width - max_x
        mirrored_max_x = screen_width - min_x

        # Zeichne die gespiegelte Bounding Box
        pygame.draw.rect(screen, (255, 0, 0), (mirrored_min_x, min_y, mirrored_max_x - mirrored_min_x, max_y - min_y), 2)

        # Text über der gespiegelten Bounding Box anzeigen
        font = pygame.font.SysFont("arial", 20)
        text = font.render("ID: 1", True, (255, 0, 0))
        screen.blit(text, (mirrored_min_x, min_y - 20))

    # Foreground-Maske (Debugging)
    if motion_mask is not None:
        fg_mask_resized = cv2.resize(motion_mask, (320, 180))  # Maske verkleinern
        fg_mask_surface = pygame.surfarray.make_surface(np.rot90(fg_mask_resized))
        screen.blit(fg_mask_surface, (10, 10))  # Debugging: Maske in der Ecke anzeigen

    # Update des Pygame-Displays
    pygame.display.update()
    clock.tick(fps)

# Ressourcen freigeben
pygame.quit()
cap.release()
cv2.destroyAllWindows()
