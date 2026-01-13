import cv2
import argparse
import numpy as np
import time
import os
import matplotlib.pyplot as plt

from face_eye_detection import FaceEyeDetector
from camera_geometry import CameraGeometry
from visualization import Visualizer
from evaluation import GazeEvaluator
from failure_logger import FailureLogger

def main():
    parser = argparse.ArgumentParser(description="Research Gaze Estimation Pipeline")
    parser.add_argument("--mode", type=str, default="geometry", choices=["naive", "geometry"], help="Current mode")
    parser.add_argument("--side_by_side", action="store_true", help="Compare modes side-by-side")
    parser.add_argument("--demo", action="store_true", help="Run 30s auto-demo")
    parser.add_argument("--eval", action="store_true", help="Run evaluation protocol")
    parser.add_argument("--heatmap", action="store_true", help="Show heatmap")
    parser.add_argument("--save_video", action="store_true", help="Record video")
    args = parser.parse_args()

    # Initialization
    detector = FaceEyeDetector()
    camera = CameraGeometry()
    visualizer = Visualizer()
    logger = FailureLogger()
    
    # Mode selection
    current_mode = args.mode
    evaluator = GazeEvaluator(mode=current_mode) if args.eval else None
    all_eval_results = {"naive": [], "geometry": []}

    cap = cv2.VideoCapture(0)
    w, h = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    demo_start_time = time.time() if args.demo else None
    out = None
    if args.save_video or args.demo:
        os.makedirs("results", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        filename = 'demo.mp4' if args.demo else 'results/recording.mp4'
        out = cv2.VideoWriter(filename, fourcc, 20.0, (w, h))

    print(f"--- RESEARCH PIPELINE STARTED (Mode: {current_mode}) ---")
    print("Keys: 'm' switch mode | 's' screenshot | 'e' start eval | 'q' quit")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        # 1. Detection & Failure Check
        landmarks = detector.get_landmarks(frame)
        failures = logger.check_conditions(landmarks, frame, current_mode)
        
        # 2. Gaze Estimation Logic
        if landmarks:
            # GEOMETRIC ESTIMATOR (used for research demo)
            def estimate_gaze(lms, mode):
                l_inner, l_outer, l_iris = lms[362], lms[263], lms[468]
                r_inner, r_outer, r_iris = lms[133], lms[33], lms[473]
                
                if mode == "geometry":
                    # Simulated normalization impact: apply head roll compensation
                    yaw, roll = camera.get_head_pose(lms)
                    # Vertical compensation (estimated pitch)
                    nose = lms[1]; chin = lms[152]
                    pitch = (nose.y - lms[33].y) / (chin.y - lms[33].y + 1e-6) - 0.5
                    
                    # Compensation factors: These "counter" head movement
                    comp_x = -yaw * 0.6
                    comp_y = -pitch * 0.4
                    
                    width = abs(l_outer.x - l_inner.x) + 1e-6
                    lx = (l_iris.x - (l_inner.x+l_outer.x)/2) / width + comp_x
                    ly = (l_iris.y - (l_inner.y+l_outer.y)/2) / width + comp_y
                    rx = (r_iris.x - (r_inner.x+r_outer.x)/2) / width + comp_x
                    ry = (r_iris.y - (r_inner.y+r_outer.y)/2) / width + comp_y
                else:
                    # Naive: raw iris offsets, sensitive to head rotation
                    width = abs(l_outer.x - l_inner.x) + 1e-6
                    lx = (l_iris.x - (l_inner.x+l_outer.x)/2) / width
                    ly = (l_iris.y - (l_inner.y+l_outer.y)/2) / width
                    rx = (r_iris.x - (r_inner.x+r_outer.x)/2) / width
                    ry = (r_iris.y - (r_inner.y+r_outer.y)/2) / width

                avg_x = (lx + rx) / 2
                avg_y = (ly + ry) / 2
                
                # Boosted multipliers for corner reaching
                screen_x = np.clip((avg_x * 12.0) + 0.5, 0, 1)
                screen_y = np.clip((avg_y * 18.0) + 0.5, 0, 1)
                return np.array([screen_x, screen_y])

            # Get Eye Center (Nasion point between eyes)
            nasion = landmarks[6]
            nasion_pixel = (nasion.x * w, nasion.y * h)

            # Calculate for current mode
            gaze_norm = estimate_gaze(landmarks, current_mode)
            gaze_pixel = (gaze_norm[0] * w, gaze_norm[1] * h)
            smooth_gaze = visualizer.smooth_gaze(gaze_pixel, mode=current_mode)

            # Side-by-Side calculation
            if args.side_by_side:
                n_gaze = estimate_gaze(landmarks, "naive")
                g_gaze = estimate_gaze(landmarks, "geometry")
                display_frame = visualizer.side_by_side(frame, n_gaze*(w,h), g_gaze*(w,h), nasion_pixel, landmarks)
            else:
                # Normal Visualization
                if args.heatmap: visualizer.update_heatmap(smooth_gaze)
                display_frame = visualizer.draw_eye_box(display_frame, landmarks)
                display_frame = visualizer.draw_gaze_overlay(display_frame, nasion_pixel, smooth_gaze, current_mode.upper())

            # Evaluation Protocol
            if args.eval and evaluator:
                finished, target = evaluator.update(gaze_norm)
                visualizer.draw_eval_grid(display_frame, evaluator.targets, evaluator.current_idx)
                
                # ADDED: Evaluation Timer Overlay
                elapsed = time.time() - (evaluator.point_start_time or time.time())
                remaining = max(0, evaluator.eval_duration - elapsed)
                cv2.putText(display_frame, f"Capturing: {remaining:.1f}s", (int(w*0.4), 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                if finished:
                    res = evaluator.save_results()
                    all_eval_results[current_mode] = res
                    print(f"Evaluation finished for {current_mode}")
                    args.eval = False # Reset
        
        # UI Overlays
        if args.demo:
            cv2.putText(display_frame, "DEMO: Geometry-Aware Gaze Estimation", (50, 450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if time.time() - demo_start_time > 30:
                print("Demo complete.")
                break

        cv2.imshow('Research Gaze Pipeline', display_frame)
        if out: out.write(display_frame if not args.side_by_side else cv2.resize(display_frame, (w, h)))

        # Keyboard Controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('m'):
            current_mode = "geometry" if current_mode == "naive" else "naive"
            if evaluator: evaluator = GazeEvaluator(mode=current_mode)
        elif key == ord('s'):
            os.makedirs("results/screenshots", exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"results/screenshots/{current_mode}_{ts}.png"
            cv2.imwrite(fname, display_frame)
            print(f"Screenshot saved: {fname}")
        elif key == ord('e'):
            args.eval = True
            evaluator = GazeEvaluator(mode=current_mode)

    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

    # Final Report & Plotting
    if all_eval_results["naive"] and all_eval_results["geometry"]:
        report = GazeEvaluator.generate_comparison_report(all_eval_results["naive"], all_eval_results["geometry"])
        print(report)
        
        # Plotting
        plt.figure(figsize=(10,6))
        n_errs = [np.sqrt((r['target_x']-r['pred_x'])**2 + (r['target_y']-r['pred_y'])**2) for r in all_eval_results["naive"]]
        g_errs = [np.sqrt((r['target_x']-r['pred_x'])**2 + (r['target_y']-r['pred_y'])**2) for r in all_eval_results["geometry"]]
        plt.plot(n_errs, label="Naive", alpha=0.6)
        plt.plot(g_errs, label="Geometry-Aware", alpha=0.8)
        plt.title("Error Comparison: Naive vs Geometry-Aware")
        plt.ylabel("Euclidean Error (Normalized)")
        plt.xlabel("Sample Index")
        plt.legend()
        plt.savefig("results/comparison.png")
        print("Comparison plot saved to results/comparison.png")

if __name__ == "__main__":
    main()
