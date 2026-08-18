import os, sys, pickle, glob, numpy as np, copy
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Slider, Button, RadioButtons

# ── Find files ────────────────────────────────────────────────────────────────
# Searches for the original FacemapPose .pkl (excludes post_processing and metadata)
# Looks first in VF1/ subfolder (new structure), then directly in session folder (legacy)
# Searches for .avi in session folder first, then parent folder (new structure)
# total_frames = min(pkl, video) as safety in case of partial desynchronization
path = input('Path to session folder (containing VF1/ or avi): ').strip()

pkl_candidates = glob.glob(os.path.join(path, 'VF1', '*FacemapPose.pkl'))
pkl_candidates = [p for p in pkl_candidates if '_metadata' not in p and 'post_processing' not in p]
if not pkl_candidates:
    pkl_candidates = glob.glob(os.path.join(path, '*FacemapPose.pkl'))
    pkl_candidates = [p for p in pkl_candidates if '_metadata' not in p and 'post_processing' not in p]
if not pkl_candidates:
    print('❌ No FacemapPose pkl found.')
    sys.exit(1)
pkl_path = pkl_candidates[0]
print(f'✅ PKL: {pkl_path}')

avi_candidates = glob.glob(os.path.join(path, '*.avi'))
if not avi_candidates:
    avi_candidates = glob.glob(os.path.join(os.path.dirname(path), '*.avi'))
if not avi_candidates:
    print('❌ No avi found.')
    sys.exit(1)
avi_path = avi_candidates[0]
print(f'✅ AVI: {avi_path}')

with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

fps = data['fps']
pupil_area = np.array(data['pupil_area'], dtype=float)
pupil_vel = np.array(data['absolute_velocity'], dtype=float) * fps
frames_time = np.array(data.get('frames', np.arange(len(pupil_area)) / fps))
total_frames_pkl = len(pupil_area)

cap = cv2.VideoCapture(avi_path)
total_frames_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
total_frames = min(total_frames_pkl, total_frames_vid)
duration_s = total_frames / fps

print(f'Total frames: {total_frames} | Duration: {duration_s/60:.1f} min | FPS: {fps}')

# ── Optional: copy segments from another post-processed pkl ──────────────────
# Useful when two videos share the same recording (e.g. with/without corneal mask)
# and should have identical liquid/eye_closed segments marked.
# Also copies the injection timepoint if one was marked in the source pkl.
# Requires confirmation before applying to avoid accidental overwrites.
print('\n=== Copy segments from another pkl? ===')
copy_path = input('Path to source post-processed pkl (or press Enter to skip): ').strip()
segments_to_copy = []
injection_to_copy = None
if copy_path and os.path.exists(copy_path):
    try:
        with open(copy_path, 'rb') as _f:
            _src = pickle.load(_f)
        segments_to_copy = _src.get('post_processing_segments', [])
        injection_to_copy = _src.get('injection_frame', None)
        print(f'✅ Found {len(segments_to_copy)} segments to copy.')
        for seg in segments_to_copy:
            print(f'   [{seg["t_start_s"]:.1f}s — {seg["t_end_s"]:.1f}s] {seg["category"]}')
        if injection_to_copy is not None:
            print(f'   Injection frame: {injection_to_copy}')
        confirm = input('Confirm copy? (y/n): ').strip().lower()
        if confirm != 'y':
            segments_to_copy = []
            injection_to_copy = None
            print('   Skipped.')
    except Exception as _e:
        print(f'⚠️  Could not load source pkl: {_e}')
        segments_to_copy = []
elif copy_path:
    print('⚠️  File not found — skipping.')

# ── State ─────────────────────────────────────────────────────────────────────
# Lists used instead of simple variables to allow mutation inside nested callback functions
# (Python closures cannot reassign variables from outer scope without 'nonlocal')
segments = []          # list of (start_frame, end_frame, category) tuples
current_start = [None] # frame index where the current segment starts (None = no segment in progress)
current_frame_idx = [0] # current frame displayed in the GUI
injection_frame = [None] # frame index of injection start (None = no injection marked)
                         # if set, triggers split mode on SAVE: creates 2 separate pkl files

# ── Figure layout ─────────────────────────────────────────────────────────────
# All axes use normalized coordinates [left, bottom, width, height] (0 to 1)
# Left panel: video display
# Right panel: pupil velocity (top) and pupil area (bottom) time series
# Bottom: time slider, action buttons, category radio buttons, segment info text
fig = plt.figure(figsize=(20, 11))
fig.suptitle(f'Post-processing GUI — {os.path.basename(pkl_path)}', fontsize=10)

ax_video  = fig.add_axes([0.02, 0.35, 0.33, 0.58])
ax_vel    = fig.add_axes([0.42, 0.60, 0.55, 0.33])
ax_area   = fig.add_axes([0.42, 0.35, 0.55, 0.20])
ax_slider = fig.add_axes([0.10, 0.25, 0.78, 0.03])
ax_btn_start   = fig.add_axes([0.02, 0.15, 0.12, 0.06])
ax_btn_end     = fig.add_axes([0.16, 0.15, 0.12, 0.06])
ax_btn_undo    = fig.add_axes([0.30, 0.15, 0.10, 0.06])
ax_btn_save    = fig.add_axes([0.42, 0.15, 0.10, 0.06])
ax_btn_inject  = fig.add_axes([0.75, 0.15, 0.15, 0.06])
ax_radio     = fig.add_axes([0.55, 0.10, 0.18, 0.12])
ax_info      = fig.add_axes([0.02, 0.02, 0.95, 0.10])
ax_info.axis('off')

# ── Signal plots ───────────────────────────────────────────────────────────────
# Orange dashed line at 800 px/s = automatic velocity filter threshold from PipelineFaceMap
# Red vertical lines (vline_vel, vline_area) follow the slider to show current frame position
# X-ticks every 500s for consistency with pipeline figures
# segment_patches_* store colored spans that are redrawn when segments are added/removed
t_s = frames_time
ax_vel.plot(t_s, pupil_vel, color='#3498db', linewidth=0.4, alpha=0.8)
ax_vel.set_ylabel('Velocity (px/s)', fontsize=8)
ax_vel.set_title('Pupil velocity', fontsize=8)
ax_vel.axhline(y=800, color='orange', linewidth=0.8, linestyle='--', alpha=0.7, label='800px/s threshold')
ax_vel.legend(fontsize=7)
ax_vel.grid(False)
ax_vel.set_xticklabels([])
tick_vals = np.arange(0, duration_s + 500, 500)
ax_vel.set_xticks(tick_vals)

ax_area.plot(t_s, pupil_area, color='#2ecc71', linewidth=0.4, alpha=0.8)
ax_area.set_ylabel('Area (px²)', fontsize=8)
ax_area.set_xlabel('Time (s)', fontsize=8)
ax_area.set_title('Pupil area', fontsize=8)
ax_area.grid(False)
ax_area.set_xticks(tick_vals)
ax_area.set_xticklabels([str(int(t)) for t in tick_vals], fontsize=7)

vline_vel  = ax_vel.axvline(x=0, color='red', linewidth=1.2)
vline_area = ax_area.axvline(x=0, color='red', linewidth=1.2)
segment_patches_vel = []
segment_patches_area = []

# ── Video display ──────────────────────────────────────────────────────────────
# get_frame() seeks directly to any frame using CAP_PROP_POS_FRAMES (not sequential)
# OpenCV reads BGR — converted to RGB for matplotlib display
# Returns black frame (zeros) if frame cannot be read (e.g. corrupted or end of file)
# img_display is updated in-place via set_data() for performance (avoids re-creating axes)
# time_title and info_text are updated on each slider move
def get_frame(idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.zeros((100, 100, 3), dtype=np.uint8)

img_display = ax_video.imshow(get_frame(0))
ax_video.axis('off')
time_title = ax_video.set_title('t=0.00s | frame 0', fontsize=8)
info_text = ax_info.text(0.01, 0.7, 'Segments: none', fontsize=8, transform=ax_info.transAxes)

# ── Controls ───────────────────────────────────────────────────────────────────
# Slider: navigate in time (seconds) — keyboard arrows also work (±10s / ±1s with Shift)
# Mark START + Mark END: define a segment to exclude
# Radio buttons: select segment category before clicking Mark END
#   - liquid: sets pupil + whiskers + eye keypoints to NaN
#   - eye_closed: sets only pupil + eye keypoints to NaN (whiskers kept)
# Start Injection: marks injection timepoint — on SAVE creates 2 pkl instead of 1:
#   *_Pre-injection_post_processing_FacemapPose.pkl and *_Injection_post_processing_FacemapPose.pkl
# SAVE: writes post-processed pkl — if no changes and no injection, saves a control PNG only


# Slider 
slider = Slider(ax_slider, 'Time (s)', 0, duration_s, valinit=0)

# Buttons
btn_start  = Button(ax_btn_start,  'Mark START',       color='#2ecc71', hovercolor='#27ae60')
btn_end    = Button(ax_btn_end,    'Mark END',         color='#e74c3c', hovercolor='#c0392b')
btn_undo   = Button(ax_btn_undo,   'Undo',             color='#f39c12', hovercolor='#e67e22')
btn_save   = Button(ax_btn_save,   'SAVE',             color='#3498db', hovercolor='#2980b9')
btn_inject = Button(ax_btn_inject, 'Start Injection',  color='#8e44ad', hovercolor='#6c3483')
radio = RadioButtons(ax_radio, ['liquid', 'eye_closed'], active=0,
                    activecolor='#e74c3c')

def on_category_change(label):
    """Update radio button active color to match the selected category.
    liquid → red (#e74c3c), eye_closed → purple (#9b59b6).
    Provides immediate visual feedback on which category is selected."""
    colors_radio = {'liquid': '#e74c3c', 'eye_closed': '#9b59b6'}
    radio.activecolor = colors_radio.get(label, '#e74c3c')
    fig.canvas.draw_idle()

radio.on_clicked(on_category_change)

def redraw_segments():
    """Redraw all segment spans on velocity and area plots to keep display synchronized with segments list.
    Called after any segment is added or removed.
    Clears all existing patches and redraws from scratch — simpler than tracking individual changes.
    try/except on remove() handles cases where a patch was already detached from the axes.
    Also updates the info text at the bottom with all segment timecodes and categories."""
    for p in segment_patches_vel + segment_patches_area:
        try: p.remove()
        except: pass
    segment_patches_vel.clear()
    segment_patches_area.clear()
    colors = {'liquid': '#e74c3c', 'eye_closed': '#9b59b6'}
    for s, e, cat in segments:
        t_s_seg = frames_time[s]
        t_e_seg = frames_time[min(e, total_frames-1)]
        c = colors.get(cat, 'grey')
        segment_patches_vel.append(ax_vel.axvspan(t_s_seg, t_e_seg, alpha=0.3, color=c))
        segment_patches_area.append(ax_area.axvspan(t_s_seg, t_e_seg, alpha=0.3, color=c))
    # Update info
    info_str = f'Segments ({len(segments)}): ' + ' | '.join(
        [f'[{frames_time[s]:.0f}s-{frames_time[min(e,total_frames-1)]:.0f}s {cat}]' for s,e,cat in segments]
    ) if segments else 'Segments: none'
    info_text.set_text(info_str)
    fig.canvas.draw_idle()

def update_frame(val):
    """Called on every slider move — updates video display and cursor lines.
    Converts slider time (seconds) to frame index via fps.
    Shows current START timestamp in title if a segment is in progress.
    Uses frames_time[idx] instead of slider value for accurate timestamp
    (frame timestamps may be slightly irregular due to recording jitter)."""
    t = slider.val
    idx = min(int(t * fps), total_frames - 1)
    current_frame_idx[0] = idx
    img_display.set_data(get_frame(idx))
    actual_t = frames_time[idx] if idx < len(frames_time) else t
    start_str = f' | START={frames_time[current_start[0]]:.1f}s' if current_start[0] is not None else ''
    time_title.set_text(f't={actual_t:.1f}s ({actual_t/60:.2f}min) | frame {idx}{start_str}')
    vline_vel.set_xdata([actual_t])
    vline_area.set_xdata([actual_t])
    fig.canvas.draw_idle()

slider.on_changed(update_frame)

def on_mark_start(event):
    """Mark the current frame as the start of a new segment.
    Stores the frame index in current_start[0].
    Does not validate anything — Mark END will check that end > start."""
    current_start[0] = current_frame_idx[0]
    t = frames_time[current_start[0]]
    print(f'▶️  Start: frame {current_start[0]} (t={t:.1f}s = {t/60:.2f}min)')

def on_mark_end(event):
    """Validate and finalize the current segment.
    Checks that START was marked and that END > START.
    Reads the selected category from radio buttons at the moment END is clicked.
    Resets current_start to None — ready for the next segment."""

    if current_start[0] is None:
        print('⚠️  Mark START first.')
        return
    end = current_frame_idx[0]
    if end <= current_start[0]:
        print('⚠️  END must be after START.')
        return
    cat = radio.value_selected
    segments.append((current_start[0], end, cat))
    t_s_seg = frames_time[current_start[0]]
    t_e_seg = frames_time[end]
    print(f'✅ [{t_s_seg:.1f}s — {t_e_seg:.1f}s] ({cat}) | Total segments: {len(segments)}')
    current_start[0] = None
    redraw_segments()

def on_undo(event):
    """Remove the last marked segment (LIFO order).
    Pops from segments list and redraws to synchronize display."""
    if segments:
        s, e, cat = segments.pop()
        print(f'↩️  Removed: [{frames_time[s]:.1f}s — {frames_time[min(e,total_frames-1)]:.1f}s] ({cat})')
        redraw_segments()
    else:
        print('⚠️  No segments to undo.')

def on_save(event):
    
   # ── Split mode ────────────────────────────────────────────────────────────
    # Triggered when 'Start Injection' was clicked
    # Creates 2 pkl by deep-copying the original data and masking frames outside each segment:
    #   - Pre-injection pkl: frames 0 → injection_frame (all frames after = NaN)
    #   - Injection pkl: frames injection_frame → end (all frames before = NaN)
    # Segments (liquid/eye_closed) are assigned to the correct pkl based on their timeframe
    # split_type, split_start_frame, split_end_frame stored for downstream identification
    
    # ── No changes mode ───────────────────────────────────────────────────────
   # If no segments and no injection marked: no pkl created
   # Saves a control PNG with watermark "No segments have been placed"
   # Provides a visual trace that the session was reviewed and deemed clean

 # ── Normal mode ───────────────────────────────────────────────────────────
    # Applies NaN to marked frames with differentiated logic:
    #   liquid → NaN on ALL signals (pupil area, velocity, center, eye keypoints, whiskers)
    #   eye_closed → NaN on pupil + eye keypoints ONLY — whiskers kept (physiologically valid)
    # Stores segment metadata in pkl for traceability
    # Generates control PNG overlaying original (grey) vs post-processed (colored) signals
    # Purple shaded zone = pre-injection period if injection was marked
    # Colored spans = liquid (red) and eye_closed (purple) segments

    if injection_frame[0] is not None and segments is not None:
        inj_f = injection_frame[0]
        print(f'\n💉 Split mode: injection at frame {inj_f} (t={frames_time[inj_f]:.1f}s)')

        base = pkl_path.replace('_FacemapPose.pkl', '')

        for suffix, f_start, f_end, seg_cat in [
            ('_Pre-injection', 0, inj_f, 'pre'),
            ('_Injection', inj_f, total_frames, 'post'),
        ]:
            data_split = copy.deepcopy(data)

            # Mask frames outside this segment
            mask = np.zeros(total_frames, dtype=bool)
            mask[f_start:f_end] = True
            outside = ~mask

            # Set outside frames to NaN
            pup_arr = np.array(data_split['pupil_area'], dtype=float)
            pup_arr[outside] = np.nan
            data_split['pupil_area'] = pup_arr

            if 'absolute_velocity' in data_split:
                vel_arr = np.array(data_split['absolute_velocity'], dtype=float)
                vel_arr[outside] = np.nan
                data_split['absolute_velocity'] = vel_arr

            if 'pupil_center_[x,y]' in data_split:
                center = np.array(data_split['pupil_center_[x,y]'], dtype=float)
                center[outside] = np.nan
                data_split['pupil_center_[x,y]'] = center

            for kp in ['eye(front)', 'eye(back)', 'eye(top)', 'eye(bottom)',
                       'whisker(I)', 'whisker(II)', 'whisker(III)']:
                if kp in data_split:
                    for coord in ['x', 'y', 'likelihood']:
                        if coord in data_split[kp]:
                            arr = np.array(data_split[kp][coord], dtype=float)
                            arr[outside] = np.nan
                            data_split[kp][coord] = arr

            # Apply segments that fall within this split
            liquid_frames_split = set()
            eye_closed_frames_split = set()
            for s, e, cat in segments:
                for f in range(max(s, f_start), min(e + 1, f_end)):
                    if cat == 'liquid':
                        liquid_frames_split.add(f)
                    else:
                        eye_closed_frames_split.add(f)

            all_marked = liquid_frames_split | eye_closed_frames_split
            if all_marked:
                pup_arr = np.array(data_split['pupil_area'], dtype=float)
                pup_arr[list(all_marked)] = np.nan
                data_split['pupil_area'] = pup_arr
                for kp in ['eye(front)', 'eye(back)', 'eye(top)', 'eye(bottom)']:
                    if kp in data_split:
                        for coord in ['x', 'y', 'likelihood']:
                            if coord in data_split[kp]:
                                arr = np.array(data_split[kp][coord], dtype=float)
                                arr[list(all_marked)] = np.nan
                                data_split[kp][coord] = arr
                for kp in ['whisker(I)', 'whisker(II)', 'whisker(III)']:
                    if kp in data_split:
                        for coord in ['x', 'y', 'likelihood']:
                            if coord in data_split[kp]:
                                arr = np.array(data_split[kp][coord], dtype=float)
                                arr[list(liquid_frames_split)] = np.nan
                                data_split[kp][coord] = arr

            data_split['post_processing_segments'] = [
                {'start': s, 'end': e, 'category': cat,
                 't_start_s': float(frames_time[s]),
                 't_end_s': float(frames_time[min(e, total_frames-1)])}
                for s, e, cat in segments
                if s < f_end and e >= f_start
            ]
            data_split['split_start_frame'] = f_start
            data_split['split_end_frame'] = f_end
            data_split['injection_frame'] = inj_f
            data_split['injection_time_s'] = float(frames_time[inj_f])
            data_split['split_type'] = seg_cat

            out_path = base + suffix + '_post_processing_FacemapPose.pkl'
            with open(out_path, 'wb') as f_out:
                pickle.dump(data_split, f_out, protocol=4)
            print(f'✅ Saved: {out_path}')

        plt.close()
        cap.release()
        return

    if not segments and injection_frame[0] is None:
        # Generate a control PNG showing no changes were made
        fig_no, axs_no = plt.subplots(2, 1, figsize=(25, 6), sharex=True)
        t_arr = data.get('frames', np.arange(len(pupil_area)) / fps)
        axs_no[0].plot(t_arr, pupil_vel, color='#3498db', linewidth=0.4, alpha=0.8)
        axs_no[0].set_ylabel('Velocity (px/s)', fontsize=8)
        axs_no[0].set_title('Pupil velocity', fontsize=8)
        axs_no[0].grid(False)
        axs_no[1].plot(t_arr, pupil_area, color='#2ecc71', linewidth=0.4, alpha=0.8)
        axs_no[1].set_ylabel('Area (px²)', fontsize=8)
        axs_no[1].set_xlabel('Time (s)', fontsize=8)
        axs_no[1].set_title('Pupil area', fontsize=8)
        axs_no[1].grid(False)
        tick_vals_no = np.arange(0, t_arr[-1] + 500, 500)
        for ax in axs_no:
            ax.set_xticks(tick_vals_no)
            ax.set_xticklabels([str(int(t)) for t in tick_vals_no], fontsize=7)
        fig_no.text(0.5, 0.5, 'No segments have been placed — original data unchanged',
                   ha='center', va='center', fontsize=14, color='#27ae60',
                   fontweight='bold', transform=fig_no.transFigure, alpha=0.6)
        fig_no.suptitle(f'Post-processing review — {os.path.basename(pkl_path)}', fontsize=10)
        png_path = pkl_path.replace('_FacemapPose.pkl', '_post_processing_no_changes_control.png')
        plt.savefig(png_path, dpi=120, bbox_inches='tight')
        plt.close(fig_no)
        print(f'✅ No changes — control PNG saved: {png_path}')
        plt.close()
        cap.release()
        return

    data_pp = copy.deepcopy(data)
    liquid_frames = set()
    eye_closed_frames = set()
    for s, e, cat in segments:
        for f in range(s, min(e + 1, total_frames)):
            if cat == 'liquid':
                liquid_frames.add(f)
            else:
                eye_closed_frames.add(f)

    # Liquid → NaN all parameters (pupil + whiskers)
    # Eye closed → NaN only pupil parameters, keep whiskers
    all_marked = liquid_frames | eye_closed_frames

    # Pupil area — NaN for both liquid and eye_closed
    pup_area_arr = np.array(data_pp['pupil_area'], dtype=float)
    pup_area_arr[list(all_marked)] = np.nan
    data_pp['pupil_area'] = pup_area_arr

    # Pupil velocity — NaN for both
    if 'absolute_velocity' in data_pp:
        vel_arr = np.array(data_pp['absolute_velocity'], dtype=float)
        vel_arr[list(all_marked)] = np.nan
        data_pp['absolute_velocity'] = vel_arr

    if 'pupil_velocity[vx,vy]' in data_pp:
        vel2 = np.array(data_pp['pupil_velocity[vx,vy]'], dtype=float)
        vel2[list(all_marked)] = np.nan
        data_pp['pupil_velocity[vx,vy]'] = vel2

    # Pupil center — NaN for both
    if 'pupil_center_[x,y]' in data_pp:
        center = np.array(data_pp['pupil_center_[x,y]'], dtype=float)
        center[list(all_marked)] = np.nan
        data_pp['pupil_center_[x,y]'] = center

    # Eye keypoints — NaN for both liquid and eye_closed
    for kp in ['eye(front)', 'eye(back)', 'eye(top)', 'eye(bottom)']:
        if kp in data_pp:
            for coord in ['x', 'y', 'likelihood']:
                if coord in data_pp[kp]:
                    arr = np.array(data_pp[kp][coord], dtype=float)
                    arr[list(all_marked)] = np.nan
                    data_pp[kp][coord] = arr

    # Whisker keypoints — NaN only for liquid frames (keep for eye_closed)
    for kp in ['whisker(I)', 'whisker(II)', 'whisker(III)']:
        if kp in data_pp:
            for coord in ['x', 'y', 'likelihood']:
                if coord in data_pp[kp]:
                    arr = np.array(data_pp[kp][coord], dtype=float)
                    arr[list(liquid_frames)] = np.nan
                    data_pp[kp][coord] = arr

    data_pp['injection_frame'] = injection_frame[0]
    data_pp['injection_time_s'] = float(frames_time[injection_frame[0]]) if injection_frame[0] is not None else None
    data_pp['post_processing_segments'] = [
        {'start': s, 'end': e, 'category': cat,
         't_start_s': float(frames_time[s]),
         't_end_s': float(frames_time[min(e, total_frames-1)])}
        for s, e, cat in segments
    ]
    data_pp['post_processing_liquid_frames'] = sorted(liquid_frames)
    data_pp['post_processing_eye_closed_frames'] = sorted(eye_closed_frames)

    base = pkl_path.replace('_FacemapPose.pkl', '')
    out_path = base + '_post_processing_FacemapPose.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(data_pp, f, protocol=4)
    print(f'✅ Saved: {out_path}')
    print(f'   Liquid: {len(liquid_frames)} frames | Eye closed: {len(eye_closed_frames)} frames')

    # Generate control figure
    import matplotlib.pyplot as plt2
    fig2, axs2 = plt2.subplots(2, 1, figsize=(25, 8), sharex=True)
    fig2.subplots_adjust(hspace=0.3)

    t_arr = data_pp.get('frames', np.arange(len(pupil_area)) / fps)
    pup_area_pp = np.array(data_pp['pupil_area'], dtype=float)
    pup_vel_pp = np.array(data_pp['absolute_velocity'], dtype=float) * fps

    axs2[0].plot(t_arr, pupil_vel, color='#95a5a6', linewidth=0.4, alpha=0.5, label='Original')
    axs2[0].plot(t_arr, pup_vel_pp, color='#3498db', linewidth=0.4, alpha=0.8, label='Post-processed')
    axs2[0].set_ylabel('Velocity (px/s)', fontsize=8)
    axs2[0].set_title('Pupil velocity', fontsize=8)
    axs2[0].grid(False)

    axs2[1].plot(t_arr, pupil_area, color='#95a5a6', linewidth=0.4, alpha=0.5, label='Original')
    axs2[1].plot(t_arr, pup_area_pp, color='#2ecc71', linewidth=0.4, alpha=0.8, label='Post-processed')
    axs2[1].set_ylabel('Area (px²)', fontsize=8)
    axs2[1].set_xlabel('Time (s)', fontsize=8)
    axs2[1].set_title('Pupil area', fontsize=8)
    axs2[1].grid(False)

    # Grey out pre-injection zone
    if injection_frame[0] is not None:
        t_inj = frames_time[injection_frame[0]]
        for ax in axs2:
            ax.axvspan(0, t_inj, alpha=0.15, color='#8e44ad')
            ax.axvline(x=t_inj, color='#8e44ad', linewidth=1.5, linestyle='--')
            ax.text(t_inj/2, ax.get_ylim()[1]*0.9, 'Pre-injection',
                   ha='center', fontsize=7, color='#8e44ad', style='italic')

    # Mark segments
    colors_seg = {'liquid': '#e74c3c', 'eye_closed': '#9b59b6'}
    legend_patches = []
    for s, e, cat in segments:
        t_s_seg = frames_time[s]
        t_e_seg = frames_time[min(e, total_frames-1)]
        c = colors_seg.get(cat, 'grey')
        for ax in axs2:
            ax.axvspan(t_s_seg, t_e_seg, alpha=0.25, color=c)
        legend_patches.append(mpatches.Patch(color=c, alpha=0.5, label=cat))

    # X ticks every 500s
    tick_vals2 = np.arange(0, t_arr[-1] + 500, 500)
    for ax in axs2:
        ax.set_xticks(tick_vals2)
        ax.set_xticklabels([str(int(t)) for t in tick_vals2], fontsize=7)
        ax.legend(fontsize=7)

    if legend_patches:
        axs2[0].legend(handles=axs2[0].get_lines() + legend_patches, fontsize=7)

    fig2.suptitle(f'Post-processing control — {os.path.basename(out_path)}', fontsize=10)
    png_path = out_path.replace('.pkl', '_control.png')
    plt2.savefig(png_path, dpi=120, bbox_inches='tight')
    plt2.close()
    print(f'✅ Control figure saved: {png_path}')

    plt.close()
    cap.release()

def on_mark_injection(event):
    """Mark the current frame as the injection timepoint.
    Draws a purple dashed vertical line and shades the pre-injection zone on both signal plots.
    Activates split mode on SAVE — creates 2 pkl instead of 1.
    Can only be set once (overwrites if clicked again)."""
# Connect all button callbacks to their respective functions
    injection_frame[0] = current_frame_idx[0]
    t = frames_time[injection_frame[0]]
    print(f'💉 Injection marked: frame {injection_frame[0]} (t={t:.1f}s = {t/60:.2f}min)')
    # Draw vertical line on signal plots
    for ax in [ax_vel, ax_area]:
        ax.axvline(x=t, color='#8e44ad', linewidth=1.5, linestyle='--', alpha=0.8)
        ax.axvspan(0, t, alpha=0.08, color='#8e44ad')
        ax.text(t, ax.get_ylim()[1]*0.95, 'injection', color='#8e44ad',
               fontsize=7, ha='left', va='top')
    fig.canvas.draw_idle()

btn_inject.on_clicked(on_mark_injection)
btn_start.on_clicked(on_mark_start)
btn_end.on_clicked(on_mark_end)
btn_undo.on_clicked(on_undo)
btn_save.on_clicked(on_save)

def on_key(event):
    """Keyboard navigation — arrows move ±10s, Shift+arrows move ±1s.
    Triggers slider update which calls update_frame() automatically."""
    idx = current_frame_idx[0]
    if event.key == 'right':
        slider.set_val(min(slider.val + 10, duration_s))
    elif event.key == 'left':
        slider.set_val(max(slider.val - 10, 0))
    elif event.key == 'shift+right':
        slider.set_val(min(slider.val + 1, duration_s))
    elif event.key == 'shift+left':
        slider.set_val(max(slider.val - 1, 0))

fig.canvas.mpl_connect('key_press_event', on_key)

# Mouse wheel zoom on video
def on_scroll(event):
    """Mouse wheel zoom on video panel only (ignored on signal plots).
   Scroll up = zoom in (scale 0.85), scroll down = zoom out (scale 1.15).
   Zooms centered on cursor position for precise inspection of pupil."""
   # Apply copied segments if any
   # Converts time in seconds from source pkl to frame indices for this video
   # Clamps to total_frames-1 in case source video was longer
   # Also copies injection timepoint and draws purple pre-injection zone on plots

   # Print controls summary to terminal for quick reference
   # plt.show() opens the GUI — blocks until window is closed
   
    if event.inaxes != ax_video:
        return
    cur_xlim = ax_video.get_xlim()
    cur_ylim = ax_video.get_ylim()
    xdata = event.xdata
    ydata = event.ydata
    if xdata is None or ydata is None:
        return
    scale = 0.85 if event.button == 'up' else 1.15
    ax_video.set_xlim([xdata - (xdata - cur_xlim[0]) * scale,
                       xdata + (cur_xlim[1] - xdata) * scale])
    ax_video.set_ylim([ydata - (ydata - cur_ylim[0]) * scale,
                       ydata + (cur_ylim[1] - ydata) * scale])
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('scroll_event', on_scroll)

# Apply copied segments if any
if segments_to_copy:
    for seg in segments_to_copy:
        # Convert time in seconds to frame indices
        s_frame = int(seg['t_start_s'] * fps)
        e_frame = int(seg['t_end_s'] * fps)
        s_frame = min(s_frame, total_frames - 1)
        e_frame = min(e_frame, total_frames - 1)
        segments.append((s_frame, e_frame, seg['category']))
    redraw_segments()
    print(f'✅ {len(segments_to_copy)} segments copied and applied.')
if injection_to_copy is not None:
    injection_frame[0] = min(injection_to_copy, total_frames - 1)
    t_inj = frames_time[injection_frame[0]]
    for ax in [ax_vel, ax_area]:
        ax.axvline(x=t_inj, color='#8e44ad', linewidth=1.5, linestyle='--', alpha=0.8)
        ax.axvspan(0, t_inj, alpha=0.08, color='#8e44ad')
        ax.text(t_inj * 0.5, ax.get_ylim()[1] * 0.92, 'Pre-injection',
               color='#8e44ad', fontsize=7, ha='center', va='top', style='italic')
    print(f'✅ Injection frame copied: t={t_inj:.1f}s')

print('\n=== Controls ===')
print('← / → : ±10 seconds | Shift+← / Shift+→ : ±1 second')
print('1. Navigate to start of segment → Mark START')
print('2. Navigate to end → Mark END')
print('3. Select category (liquid / eye_closed)')
print('4. SAVE when done\n')

plt.show()
