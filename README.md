# CV_2024
Computer Vision - Pong Game 

Beschreibung

    Bewegungserkennung OpenCV
    Darstellung in einem Pygame-Fenster
    Verwendung der MOG2-Methode und KNN zur Hintergrundsubtraktion

Voraussetzungen

    Python

    Bibliotheken:
        numpy
        opencv-python
        pygame

Installation der Bibliotheken

    pip install numpy opencv-python pygame

Dateistruktur

    main.py: Hauptscript mit der Logik für Bewegungserkennung und Darstellung

Navigiere im Terminal zum Projektverzeichnis und führe aus:

    python main.py

    Videoquelle: Standardmäßig Wand_Jacke_dunkel.mp4 (es gibt mehrere Video quellen)

    Ändere den Dateinamen im Script für andere Videos oder setze source auf "webcam" für Webcam-Nutzung

Funktionen

    Erkennt bewegliche Objekte und zeichnet Bounding Boxes
    Visualisiert das Video und Erkennungen in Pygame