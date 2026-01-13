import numpy as np
import pandas as pd
import os
import time

class GazeEvaluator:
    def __init__(self, mode="geometry"):
        self.mode = mode
        # 4 Corner targets (Corners of the screen)
        self.targets = [
            (0.1, 0.1), (0.9, 0.1),
            (0.1, 0.9), (0.9, 0.9)
        ]
        self.current_idx = 0
        self.results = []
        self.point_start_time = None
        self.eval_duration = 5.0 # Increased for stability

    def update(self, predicted_gaze_norm):
        """
        Processes evaluation steps. 
        Returns (is_finished, current_target_coords)
        """
        if self.current_idx >= len(self.targets):
            return True, None

        if self.point_start_time is None:
            self.point_start_time = time.time()

        elapsed = time.time() - self.point_start_time
        
        # Collect data during the 3-second window
        self.results.append({
            'mode': self.mode,
            'target_idx': self.current_idx,
            'target_x': self.targets[self.current_idx][0],
            'target_y': self.targets[self.current_idx][1],
            'pred_x': predicted_gaze_norm[0],
            'pred_y': predicted_gaze_norm[1],
            'timestamp': time.time()
        })

        if elapsed >= self.eval_duration:
            self.current_idx += 1
            self.point_start_time = None
            print(f"Captured Target {self.current_idx}/{len(self.targets)}")

        return False, self.targets[self.current_idx] if self.current_idx < len(self.targets) else None

    def save_results(self, folder="results"):
        import csv
        path = os.path.join(folder, f"evaluation_{self.mode}.csv")
        os.makedirs(folder, exist_ok=True)
        
        keys = self.results[0].keys() if self.results else []
        with open(path, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(self.results)
        return self.results

    @staticmethod
    def generate_comparison_report(naive_results, geo_results):
        def calc_metrics(res):
            errors = [np.sqrt((r['target_x']-r['pred_x'])**2 + (r['target_y']-r['pred_y'])**2) for r in res]
            return np.mean(errors) if errors else 0, np.std(errors) if errors else 0

        n_mean, n_std = calc_metrics(naive_results)
        g_mean, g_std = calc_metrics(geo_results)
        
        report = f"""
======= RESEARCH COMPARISON REPORT =======
1. NAIVE MODE:
   - Mean Euclidean Error: {n_mean:.4f}
   - Stability (Std Dev): {n_std:.4f}

2. GEOMETRY-AWARE MODE:
   - Mean Euclidean Error: {g_mean:.4f}
   - Stability (Std Dev): {g_std:.4f}

RESULT: Geometry-aware normalization reduced error by {((n_mean - g_mean)/(n_mean+1e-6))*100:.2f}%
==========================================
"""
        return report
