# 🎮 Computer Vision – Pong Game mit OpenCV & Pygame

### 📌 Projektbeschreibung

Dieses Projekt kombiniert Computer Vision mit einem klassischen Pong-Spiel, um Bewegungserkennung interaktiv sichtbar zu machen.
Das Ziel ist es, ein Videobild (z. B. aus einer Datei oder Webcam) mit OpenCV zu verarbeiten, bewegliche Objekte zu erkennen und deren Positionen in einem Pygame-Fenster darzustellen.

Dabei kommen verschiedene Methoden der Hintergrundsubtraktion zum Einsatz, wie MOG2 und KNN, um bewegte Objekte von einem statischen Hintergrund zu unterscheiden.

Die Bounding Boxes der erkannten Objekte werden live angezeigt und können für weitere Anwendungen (z. B. Tracking, Interaktionen im Spiel, Objektanalyse) genutzt werden.

### 🚀 Features

🎥 Videoquellen: Nutzung von Videodateien oder der Webcam

🔍 Bewegungserkennung mit OpenCV (MOG2 & KNN Hintergrundsubtraktion)

📦 Bounding Boxes zur Markierung bewegter Objekte

🖼 Visualisierung im Pygame-Fenster (statt nur in OpenCV)

⚡ Echtzeit-fähig für einfache Bewegungsanalyse und Spielelogik

### 📂 Projektstruktur

📦 project-root
 ┣ 📜 main.py          # Hauptskript: Bewegungserkennung & Visualisierung
 ┣ 📜 README.md        # Projektdokumentation
 ┗ 📂 videos/          # Beispielvideos (z. B. Wand_Jacke_dunkel.mp4)

### 🛠 Voraussetzungen

Python 3.8+

Benötigte Bibliotheken
pip install numpy opencv-python pygame

### ▶️ Ausführung

Projekt klonen oder herunterladen

Im Terminal ins Projektverzeichnis wechseln

Skript starten mit:
python main.py

Standardmäßig wird das Video Wand_Jacke_dunkel.mp4 genutzt.

Um die Quelle zu ändern, den Dateinamen im Code anpassen

Für Webcam-Nutzung: source = "webcam" setzen

### 🎯 Funktionsweise im Detail

Videoframes laden (entweder Datei oder Webcam)

Hintergrundsubtraktion anwenden (MOG2 oder KNN)

Konturenanalyse → Erkennen von bewegten Objekten

Bounding Boxes zeichnen zur Markierung dieser Objekte

Darstellung im Pygame-Fenster mit live aktualisiertem Bild

https://github.com/user-attachments/assets/e6034030-5987-4645-ac26-dc711a36dda5



### 💡 Anwendungsmöglichkeiten

Dieses Projekt dient als Grundlage für verschiedene Szenarien:

Objekterkennung & -verfolgung in Videos

Spielersteuerung in einfachen Games über Bewegung

Experimente mit verschiedenen Computer-Vision-Algorithmen

Einstieg in KI-gestützte Bildverarbeitung

### 📸 Beispiel

Video wird geladen → Objekte bewegen sich → Bounding Boxes erscheinen → Ausgabe live im Pygame-Fenster
