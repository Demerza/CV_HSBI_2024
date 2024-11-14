import numpy as np
import cv2
import pygame

SCREEN_WIDTH  = 1280
SCREEN_HEIGHT = 720
SCREEN       = [SCREEN_WIDTH, SCREEN_HEIGHT]

# --------------------------------------------------------------------------
# -- BackgroundSubtraction K-Nearest Neighbors (KNN) Methode
# --------------------------------------------------------------------------

class BackgroundSubtraction:
    def __init__(self):
        # Initialisiere den KNN-Hintergrundsubtraktor mit angepassten Parametern
        self.bg_subtractor = cv2.createBackgroundSubtractorKNN(history=50, dist2Threshold=1000.0, detectShadows=False)


    def apply(self, frame):
        # Wendet die Hintergrundsubtraktion an
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Konturen finden
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Bounding Box bestimmen
        min_x, min_y = frame.shape[1], frame.shape[0]
        max_x, max_y = 0, 0
        found_contour = False
        
        for contour in contours:
            if cv2.contourArea(contour) > 800:  
                x, y, w, h = cv2.boundingRect(contour)
                min_x, min_y = min(min_x, x), min(min_y, y)
                max_x, max_y = max(max_x, x + w), max(max_y, y + h)
                found_contour = True

        # Wenn keine Kontur gefunden wurde, gib None zurück
        if not found_contour:
            return None, fg_mask
        
        return (min_x, min_y, max_x, max_y), fg_mask


# init pygame
pygame.init()

# set display size and caption
screen = pygame.display.set_mode(SCREEN)
pygame.display.set_caption("KNN-Methode")

# init game clock
fps = 30
clock = pygame.time.Clock()

# Kamera- oder Videoquelle auswählen
source = "webca"  # Ändere auf "video" für eine Videodatei
if source == "webcam":
    cap = cv2.VideoCapture(0)  # Webcam
else:
    cap = cv2.VideoCapture("Neutral_Jacke_langsam.mp4")  # Videodatei (Dateiname anpassen)

if not cap.isOpened():
    print("Fehler: Videoquelle konnte nicht geöffnet werden.")
    exit()

# Bildschirmgröße der Videoquelle anpassen
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen.get_width())
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen.get_height())

# BackgroundSubtraction-Objekt erstellen
bg_subtraction = BackgroundSubtraction()

# example variable for game score
# gameScore = 0

# -------------
# -- main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        # press 'esc' to quit
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

    # -- opencv & viz image
    ret, cameraFrame = cap.read()
    if not ret or cameraFrame is None:
        print("Fehler: Frame konnte nicht gelesen werden.")
        break
    
    # print("Frame erfolgreich gelesen.")

    # Hintergrundsubtraktion und Bounding Box anwenden
    bbox, fg_mask = bg_subtraction.apply(cameraFrame)

    # Bild für Pygame anzeigen und rotieren
    imgRGB = cv2.cvtColor(cameraFrame, cv2.COLOR_BGR2RGB)
    imgRGB = np.rot90(imgRGB)
    gameFrame = pygame.surfarray.make_surface(imgRGB).convert()    
    screen.blit(gameFrame, (0, 0))

    # Zeige die Foreground Mask an, um die Erkennung zu überprüfen
    # cv2.imshow("Foreground Mask", fg_mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
     break

    # Zeichne die gespiegelte Bounding Box, wenn eine erkannt wurde
    if bbox:
        min_x, min_y, max_x, max_y = bbox
        # print("Bounding Box:", bbox)  # Debugging-Ausgabe

        # Berechne gespiegelte Bounding Box-Koordinaten
        screen_width = screen.get_width()
        mirrored_min_x = screen_width - max_x
        mirrored_max_x = screen_width - min_x

        # Zeichne die gespiegelte Bounding Box
        pygame.draw.rect(screen, (255, 0, 0), (mirrored_min_x, min_y, mirrored_max_x - mirrored_min_x, max_y - min_y), 2)
        
        # ID über der gespiegelten Bounding Box anzeigen
        font = pygame.font.SysFont("arial", 20)
        text = font.render("ID: 1", True, (255, 0, 0))
        screen.blit(text, (mirrored_min_x, min_y - 20))

    # -- add Text on screen (e.g. score)
    # textFont = pygame.font.SysFont("arial", 26)
    # textExample = textFont.render(f'Score: {gameScore}', True, (255, 0, 0))
    # screen.blit(textExample, (20, 20))

    # update entire screen
    pygame.display.update()
    # set clock
    clock.tick(fps)

# quit game
pygame.quit()
cv2.destroyAllWindows()

# release capture
cap.release()
