# DOKU / Anleitungen

# Optimierung der Hintergrundsubtraktion mit MOG2

Der MOG2-Algorithmus (Mixture of Gaussian) ist eine effektive Methode zur Hintergrundsubtraktion, ideal für Videoüberwachung und ähnliche Anwendungen. Hier sind die Schlüsselparameter und einige spezifische Anwendungsdetails, die du für die Optimierung anpassen kannst:

## 1. **History**
- **Beschreibung**: Definiert die Anzahl der letzten Frames, die für die Modellierung des Hintergrunds verwendet werden.
- **Effekt**: Ein höherer Wert führt zu einer stabileren Hintergrundschätzung, kann aber die Anpassungsfähigkeit des Modells an schnelle Änderungen beeinträchtigen.
- **Typische Werte**: 100 - 1000
- **Anwendung**: Reduziere diesen Wert in dynamischen Umgebungen für eine schnellere Anpassung.

## 2. **VarThreshold**
- **Beschreibung**: Der Schwellenwert für die Einteilung eines Pixels in den Vorder- oder Hintergrund.
- **Effekt**: Ein niedriger Wert macht den Detektor empfindlicher für kleine Änderungen.
- **Typische Werte**: 10 - 200
- **Anwendung**: Erhöhe diesen Wert, um Falschmeldungen durch Rauschen und andere geringfügige Änderungen zu vermeiden.

## 3. **DetectShadows**
- **Beschreibung**: Bestimmt, ob Schatten im Bild erkannt und als solche gekennzeichnet werden sollen.
- **Effekt**: Kann die Genauigkeit verbessern, indem zwischen Schatten und tatsächlichen Objektbewegungen unterschieden wird.
- **Typische Werte**: `True` oder `False`
- **Anwendung**: Deaktiviere diese Option, wenn Schatten als Bewegungen fehlinterpretiert werden und dies zu Problemen führt.

## 4. **Konturerkennung und Bounding Box-Berechnung**
- **Beschreibung**: Nach Anwendung der Hintergrundsubtraktion werden Konturen aus der Vordergrundmaske extrahiert, um bewegliche Objekte zu identifizieren.
- **Schwellenwert für Konturflächen (`cv2.contourArea`)**: Bestimmt, welche Konturen als signifikante Objekte betrachtet werden.
- **Effekt**: Verhindert, dass kleine, irrelevante Bewegungen als Objekte erfasst werden.
- **Typische Werte**: 100 - 1000 Pixel²
- **Anwendung**: Anpassen des Schwellenwerts, um kleine Konturen zu ignorieren, die durch Rauschen entstehen könnten, oder um sicherzustellen, dass alle relevanten Bewegungen erkannt werden.


# Optimierung der Hintergrundsubtraktion mit KNN

Der KNN-Algorithmus (K-Nearest Neighbors) ist eine effektive Methode zur Hintergrundsubtraktion, ideal für Anwendungen, die eine adaptive Reaktion auf sich schnell ändernde Hintergründe benötigen. Hier sind die Schlüsselparameter und einige spezifische Anwendungsdetails, die du für die Optimierung anpassen kannst:

## 1. **History**
- **Beschreibung**: Definiert die Anzahl der letzten Frames, die zur Modellierung des Hintergrunds verwendet werden.
- **Effekt**: Ein höherer Wert führt zu einer stabileren Hintergrundschätzung, kann aber die Anpassungsfähigkeit des Modells an schnelle Änderungen beeinträchtigen.
- **Typische Werte**: 100 - 500
- **Anwendung**: Reduziere diesen Wert in dynamischen Umgebungen für eine schnellere Anpassung.

## 2. **Dist2Threshold**
- **Beschreibung**: Der Schwellenwert für die Entscheidung, ob ein Pixel zum Hintergrund gehört, basierend auf der Distanz im Farbraum.
- **Effekt**: Ein niedriger Wert macht den Detektor empfindlicher für kleine Änderungen.
- **Typische Werte**: 400.0 - 800.0
- **Anwendung**: Reduziere diesen Wert, um den Algorithmus empfindlicher gegenüber kleinen Änderungen zu machen, oder erhöhe ihn, um weniger empfindlich gegenüber Rauschen und leichten Bewegungen zu sein.

## 3. **DetectShadows**
- **Beschreibung**: Bestimmt, ob der Algorithmus Schatten von Objekten erkennen und behandeln soll.
- **Effekt**: Kann die Genauigkeit verbessern, indem zwischen Schatten und tatsächlichen Objektbewegungen unterschieden wird.
- **Typische Werte**: `True` oder `False`
- **Anwendung**: Deaktiviere diese Option, wenn Schatten als störend empfunden werden oder die Performance beeinträchtigen.

## 4. **Konturerkennung und Bounding Box-Berechnung**
- **Beschreibung**: Nach Anwendung der Hintergrundsubtraktion werden Konturen aus der Vordergrundmaske extrahiert, um bewegliche Objekte zu identifizieren.
- **Schwellenwert für Konturflächen (`cv2.contourArea`)**: Bestimmt, welche Konturen als signifikante Objekte betrachtet werden.
- **Effekt**: Verhindert, dass kleine, irrelevante Bewegungen als Objekte erfasst werden.
- **Typische Werte**: 100 - 1000 Pixel²
- **Anwendung**: Anpassen des Schwellenwerts, um kleine Konturen zu ignorieren, die durch Rauschen entstehen könnten, oder um sicherzustellen, dass alle relevanten Bewegungen erkannt werden.

## Fazit
Die Anpassung dieser Parameter erfordert eine sorgfältige Überlegung der spezifischen Anforderungen deiner Anwendung und der Umgebungsbedingungen. Durch Experimentieren und Anpassen dieser Einstellungen kannst du eine optimale Balance zwischen Empfindlichkeit und Stabilität des Hintergrundmodells erreichen, was entscheidend für die Leistungsfähigkeit deiner Hintergrundsubtraktionstechnik ist. Nutze Testszenarien, um die Auswirkungen von Änderungen in realen Anwendungsfällen zu beurteilen und die Parameter entsprechend anzupassen.
"""