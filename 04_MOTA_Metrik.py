import json
import numpy as np
from scipy.optimize import linear_sum_assignment
import os

def calculate_mota(ground_truth_path, tracking_results_path):
    """
    Berechnet die MOTA-Metrik basierend auf Ground Truth und Tracking-Ergebnissen.
    
    :param ground_truth_path: Pfad zur Ground-Truth-JSON-Datei
    :param tracking_results_path: Pfad zur Tracking-Ergebnisse-JSON-Datei
    """
    # Ground Truth und Tracking-Daten laden
    with open(ground_truth_path, "r") as gt_file:
        ground_truth = json.load(gt_file)

    with open(tracking_results_path, "r") as track_file:
        tracking_results = json.load(track_file)

    # Gemeinsame Frames identifizieren
    ground_truth_frames = set(ground_truth.keys())
    tracking_frames = set(tracking_results.keys())
    common_frames = sorted(ground_truth_frames.intersection(tracking_frames))

    total_gt_objects = 0
    false_negatives = 0
    false_positives = 0
    id_switches = 0

    previous_matches = {}

    # Iteration über gemeinsame Frames
    for frame_id in common_frames:
        gt_bboxes = ground_truth.get(frame_id, [])
        tracked_bboxes = tracking_results.get(frame_id, [])

        total_gt_objects += len(gt_bboxes)

        # Kostenmatrix für das Matching (basierend auf IoU)
        cost_matrix = np.zeros((len(gt_bboxes), len(tracked_bboxes)))

        for i, gt in enumerate(gt_bboxes):
            for j, tr in enumerate(tracked_bboxes):
                iou = calculate_iou(gt["bbox"], tr["bbox"])
                cost_matrix[i, j] = 1 - iou  # Kosten = 1 - IoU

        # Hungarian Algorithm für Matching
        if cost_matrix.size > 0:
            gt_indices, tr_indices = linear_sum_assignment(cost_matrix)
        else:
            gt_indices, tr_indices = np.array([]), np.array([])

        matched_gt = set()
        matched_tr = set()
        current_matches = {}

        # IoU-Schwelle für gültiges Matching
        iou_threshold = 0.5

        for gt_idx, tr_idx in zip(gt_indices, tr_indices):
            if cost_matrix[gt_idx, tr_idx] < 1 - iou_threshold:
                matched_gt.add(gt_idx)
                matched_tr.add(tr_idx)
                current_matches[gt_idx] = tracked_bboxes[tr_idx]["id"]

                # ID-Wechsel prüfen
                if gt_idx in previous_matches:
                    if previous_matches[gt_idx] != tracked_bboxes[tr_idx]["id"]:
                        id_switches += 1

        previous_matches = current_matches

        # Berechnung von FN und FP
        false_negatives += len(gt_bboxes) - len(matched_gt)
        false_positives += len(tracked_bboxes) - len(matched_tr)

    # MOTA berechnen
    if total_gt_objects > 0:
        mota = 1 - (false_negatives + false_positives + id_switches) / total_gt_objects
    else:
        mota = -1  # Kein gültiger MOTA-Wert, wenn keine Ground Truth Objekte existieren

    # Ergebnisse ausgeben
    print(f"Total Ground Truth Objects: {total_gt_objects}")
    print(f"False Negatives (FN): {false_negatives}")
    print(f"False Positives (FP): {false_positives}")
    print(f"ID Switches: {id_switches}")
    print(f"MOTA: {mota:.2f}")

def calculate_iou(box1, box2):
    """
    Berechnet die Intersection over Union (IoU) zwischen zwei Bounding Boxes.
    
    :param box1: Bounding Box 1 im Format {"x_min": int, "y_min": int, "x_max": int, "y_max": int}
    :param box2: Bounding Box 2 im gleichen Format
    :return: IoU-Wert
    """
    x_min1, y_min1, x_max1, y_max1 = box1["x_min"], box1["y_min"], box1["x_max"], box1["y_max"]
    x_min2, y_min2, x_max2, y_max2 = box2["x_min"], box2["y_min"], box2["x_max"], box2["y_max"]

    # Schnittmenge berechnen
    x_min_inter = max(x_min1, x_min2)
    y_min_inter = max(y_min1, y_min2)
    x_max_inter = min(x_max1, x_max2)
    y_max_inter = min(y_max1, y_max2)

    inter_area = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)

    # Vereinigungsmenge berechnen
    area_box1 = (x_max1 - x_min1) * (y_max1 - y_min1)
    area_box2 = (x_max2 - x_min2) * (y_max2 - y_min2)
    union_area = area_box1 + area_box2 - inter_area

    if union_area == 0:
        return 0

    return inter_area / union_area

# Aufruf
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ground_truth_path = os.path.join(base_dir, "ground_truth.json")
    tracking_results_path = os.path.join(base_dir, "tracking_results.json")
    calculate_mota(ground_truth_path, tracking_results_path)
