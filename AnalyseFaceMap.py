import subprocess
import os
import numpy as np
import pickle
import cv2
from facemap import utils
from facemap.pose import refine_pose
import matplotlib.pyplot as plt

path = input("Common path to the files containing the video : ")

def find_files(root_dir):
    """Recursively search root_dir for all FaceMap-related files.
Uses the system 'find' command for fast recursive search.
Returns three sorted lists: (npy_files, h5_files, video_files).
npy_files: FaceMap proc files (*_proc.npy) — pupil area, center of mass
h5_files: FaceMap Pose files (*_FacemapPose.h5) — keypoints (whiskers, eye)
video_files: raw video files (*.avi, *.mp4)
"""
    find_command = ['find', root_dir, '-type', 'f', '(',
        '-iname', '*_proc.npy', '-o',
        '-iname', '*_FacemapPose.h5', '-o',
        '-iname', '*.mp4', '-o',
        '-iname', '*.avi', ')']
    try:
        result = subprocess.run(find_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        files = result.stdout.strip().split('\n')
        npy_files   = sorted([f for f in files if f.endswith('_proc.npy')])
        h5_files    = sorted([f for f in files if f.endswith('_FacemapPose.h5')])
        video_files = sorted([f for f in files if f.endswith('.mp4') or f.endswith('.avi')])
        return npy_files, h5_files, video_files
    except subprocess.CalledProcessError as e:
        print("An error occurred:", e.stderr)
        return [], [], []

def load_npy(path):
    """Load a FaceMap _proc.npy file.
    Handles both numpy scalar array format (standard) and raw pickle format (legacy).
    Returns a dictionary containing pupil data, ROI settings, and SVD traces.
    """
    try:
        raw = np.load(path, allow_pickle=True)
        if isinstance(raw, np.ndarray):
            return raw.item()
        return raw
    except:
        with open(path, 'rb') as f:
            return pickle.load(f)

def get_npy_frames(path):
    """Return the number of frames in a FaceMap proc.npy file.
 Uses the pupil area array length as frame count reference."""
    data = load_npy(path)
    return len(data['pupil'][0]['area'])

def get_h5_frames(path):
    """Return the number of frames in a FaceMap Pose h5 file.
    Keypoints array has shape (bodyparts, coords, frames) — returns axis 2."""
    kp = utils.load_keypoints(refine_pose.BODYPARTS, path)
    return kp.shape[2]

def get_avi_frames(path):
    """Return the number of frames in a video file using OpenCV.
    Used to verify synchronization between npy, h5 and video files."""
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n

npy_files, h5_files, video_files = find_files(path)
print(f'\nFound: {len(npy_files)} npy, {len(h5_files)} h5, {len(video_files)} videos')

# Index files by base name (without extension) to enable matching across file types
# e.g. 'F04032025-0000_Ivabradine' -> path to its .npy, .h5 and .avi
npy_dict = {os.path.basename(f).replace('_proc.npy', ''): f for f in npy_files}
h5_dict  = {os.path.basename(f).replace('_FacemapPose.h5', ''): f for f in h5_files}
vid_dict = {os.path.basename(f).replace('.avi', '').replace('.mp4', ''): f for f in video_files}


# Frame verification: ensure npy, h5 and video have identical frame counts
# This is a critical safety check — mismatched files produce corrupted pkl output
# Only videos where all 3 files match are passed to the pipeline
# Common causes of mismatch: incomplete FaceMap processing, interrupted recording
print(f'\n=== Frame verification ===')
matched_npy, matched_h5, matched_vid = [], [], []
for name, vid_path in sorted(vid_dict.items()):
    if name not in npy_dict:
        print(f'⚠️  {name}.avi — no matching npy, skipping')
        continue
    if name not in h5_dict:
        print(f'⚠️  {name}.avi — no matching h5, skipping')
        continue
    try:
        n_avi = get_avi_frames(vid_path)
        n_npy = get_npy_frames(npy_dict[name])
        n_h5  = get_h5_frames(h5_dict[name])
        if n_avi == n_npy == n_h5:
            print(f'✅ {name}.avi — {n_avi} frames')
            matched_npy.append(npy_dict[name])
            matched_h5.append(h5_dict[name])
            matched_vid.append(vid_path)
        else:
            print(f'❌ {name}.avi — frame mismatch: npy={n_npy} h5={n_h5} video={n_avi}')
    except Exception as e:
        print(f'❌ {name}.avi — error: {e}')

# Summary of frame verification
# Exit early if no valid videos found — avoids launching pipeline with empty input
print(f'\n{len(matched_npy)}/{len(vid_dict)} videos passed frame verification')
print(len(matched_npy))  # debug line 

if len(matched_npy) == 0:
    print('No valid videos to process.')
    exit(1)

# ── Pre-pass: calibration GUI for all videos ─────────────────────────────────
# Collects pixel/mm calibration for all videos BEFORE launching overnight analysis.
# The user clicks 2 points on the respiratory apparatus diameter on the first frame.
# Calibrations are saved to /tmp/facemap_calibrations_TIMESTAMP.json
# and loaded automatically by PipelineFaceMap.py during processing.
# If calibration is skipped for a video, the eye keypoints fallback is used in ComparaisonFaceMap.py to estimate px/mm.
print(f'\n=== Calibration GUI — {len(matched_vid)} videos ===')
print('Please position 2 points on the respiratory apparatus for each video.')
print('Close each figure window when done.\n')

calibrations = {}  # {vid_path: (px_per_mm, dist_px, diam_mm)}

for vid_path in matched_vid:
    name = os.path.basename(vid_path).replace('.avi', '').replace('.mp4', '')
    print(f'\n[{matched_vid.index(vid_path)+1}/{len(matched_vid)}] {name}')

    cap = cv2.VideoCapture(vid_path)
    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if not ret:
        print(f'⚠️  Cannot read first frame — skipping calibration for {name}')
        calibrations[vid_path] = None
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fig_cal, ax_cal = plt.subplots(figsize=(10, 8))
    ax_cal.imshow(frame_rgb, cmap='gray')
    ax_cal.set_title(f'[{matched_vid.index(vid_path)+1}/{len(matched_vid)}] {name}\nClick 2 points on respiratory apparatus — Right click to reset', fontsize=9)
    ax_cal.axis('off')

    points_cal = []

   # Interactive callback — left click to place points, right click to undo
    # Accepts exactly 2 points defining the respiratory apparatus diameter
    def onclick_cal(event):
        if not event.inaxes:
            return
        if event.button == 3:  # right click = reset
            if points_cal:
                points_cal.pop()
                ax_cal.cla()
                ax_cal.imshow(frame_rgb, cmap='gray')
                ax_cal.set_title(f'[{matched_vid.index(vid_path)+1}/{len(matched_vid)}] {name}\nClick 2 points — Right click to reset', fontsize=9)
                ax_cal.axis('off')
                for pt in points_cal:
                    ax_cal.plot(pt[0], pt[1], 'r+', markersize=15, markeredgewidth=2)
                fig_cal.canvas.draw()
            return
        if len(points_cal) < 2:
            points_cal.append((event.xdata, event.ydata))
            ax_cal.plot(event.xdata, event.ydata, 'r+', markersize=15, markeredgewidth=2)
            if len(points_cal) == 2:
                dist_px = np.sqrt((points_cal[1][0]-points_cal[0][0])**2 +
                                  (points_cal[1][1]-points_cal[0][1])**2)
                ax_cal.plot([points_cal[0][0], points_cal[1][0]],
                           [points_cal[0][1], points_cal[1][1]], 'r-', linewidth=1.5)
                ax_cal.set_title(f'Distance: {dist_px:.1f} px — Close window to continue', fontsize=10)
            fig_cal.canvas.draw()

    cid = fig_cal.canvas.mpl_connect('button_press_event', onclick_cal)
    plt.show()
    fig_cal.canvas.mpl_disconnect(cid)
    plt.close(fig_cal)

    if len(points_cal) == 2:
        dist_px = np.sqrt((points_cal[1][0]-points_cal[0][0])**2 +
                          (points_cal[1][1]-points_cal[0][1])**2)
        diam_mm = None
        while diam_mm is None:
            try:
                val = input(f'   Diameter in mm (distance: {dist_px:.1f} px): ')
                diam_mm = float(val.strip())
            except ValueError:
                print('   Please enter a valid number.')
        px_per_mm = dist_px / diam_mm
        calibrations[vid_path] = (px_per_mm, dist_px, diam_mm)
        print(f'   ✅ {dist_px:.1f} px = {diam_mm} mm → {px_per_mm:.2f} px/mm')
    else:
        print(f'   ⚠️  Calibration skipped for {name}')
        calibrations[vid_path] = None
        
    # After window closes: compute px/mm ratio from clicked distance and user-entered diameter
    # px_per_mm is stored per video and saved to JSON for PipelineFaceMap.py
    # If fewer than 2 points were clicked, calibration is skipped (None stored) → eye keypoints fallback will be used in ComparaisonFaceMap.py


# Save calibrations to a timestamped JSON file in /tmp/

# Timestamp ensures each terminal running a batch has its own file — no conflicts
# PipelineFaceMap.py searches all /tmp/facemap_calibrations_*.json and merges them
# so multiple overnight batches in parallel all find their calibrations correctly.
import json
import time as _time
cal_file = f'/tmp/facemap_calibrations_{int(_time.time())}.json'
cal_data = {}
for vid_path, cal in calibrations.items():
    if cal is not None:
        cal_data[vid_path] = {'px_per_mm': cal[0], 'dist_px': cal[1], 'diam_mm': cal[2]}
with open(cal_file, 'w') as f:
    json.dump(cal_data, f)
print(f'\n✅ Calibrations saved to {cal_file}')
print(f'\n=== Starting analysis ===\n')

npy_str   = ','.join(matched_npy)
h5_str    = ','.join(matched_h5)
video_str = ','.join(matched_vid)

# Launch PipelineFaceMap.py with matched files as comma-separated arguments
# Each list contains only files that passed frame verification
subprocess.run(['python', 'PipelineFaceMap.py', npy_str, h5_str, video_str])
