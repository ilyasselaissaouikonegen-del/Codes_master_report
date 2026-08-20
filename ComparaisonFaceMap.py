#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ComparaisonFaceMap.py compares FaceMap pupil and whisker signals across experimental conditions.

Generates figures for:
  - Temporal median ± IQR per condition
  - Individual session curves
  - Summary box plots with statistics
  - Pre-injection vs injection comparison
  - Saline vs Ivabradine comparison
  - Raw (post-AnalyseFaceMap) vs Post-processed comparison

Run directly to generate all figures:
  python ComparaisonFaceMap.py
Output saved to: /media/nas8-2/ProjectCardioSense/Data_for_Ilyass/Comparaisons_FaceMap/TO RENAME_YYYYMMDD_HHMMSS/
"""
import os
import sys
import pickle
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# Add script directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) 
from path_for_expe_facemap import get_facemap_data, EXPERIMENTS, MICE, MOUSE_COLORS #calls the dict EXPERIMENTS formed by get_facemap_data.py that compiles all the paths to the latest .pkl

from datetime import datetime

RUN_DIR = None
OUTPUT_DIR = None 
OUTPUT_DIR_FILTERED = None # Comparison_of_filtered_data/
OUTPUT_DIR_RAW = None # Raw_vs_Postprocessed/
OUTPUT_DIR_PRE_VS_TEST = None # Pre_injection_vs_Injection/
OUTPUT_DIR_SALINE_IVA = None # Saline_vs_Ivabradine/

def _init_output_dirs():
    """
    Initialize and create all output directories for this analysis run.
   
   Called once at the start of run_comparisons() — not on import.
   Creates a timestamped 'TO RENAME_YYYYMMDD_HHMMSS' folder in Comparaisons_FaceMap/
   so the user can rename it to describe the analysis after reviewing the figures.
   
   Subdirectory structure:
     TO RENAME_YYYYMMDD_HHMMSS/
     ├── Comparison_of_filtered_data/   — temporal and summary figures per condition
     ├── Raw_vs_Postprocessed/          — raw vs post-processed comparison figures
     ├── Pre_injection_vs_Injection/    — pre vs injection temporal and ratio figures
     └── Saline_vs_Ivabradine/         — saline vs ivabradine temporal figures
   
   Uses global variables so all plot functions can access output paths without arguments.
   """
   
    global RUN_DIR, OUTPUT_DIR, OUTPUT_DIR_FILTERED, OUTPUT_DIR_RAW, OUTPUT_DIR_PRE_VS_TEST, OUTPUT_DIR_SALINE_IVA
    RUN_DIR = os.path.join('/media/nas8-2/ProjectCardioSense/Data_for_Ilyass/Comparaisons_FaceMap',
                           'TO RENAME_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    OUTPUT_DIR = RUN_DIR
    OUTPUT_DIR_FILTERED = os.path.join(RUN_DIR, 'Comparison_of_filtered_data')
    OUTPUT_DIR_RAW = os.path.join(RUN_DIR, 'Raw_vs_Postprocessed')
    OUTPUT_DIR_PRE_VS_TEST = os.path.join(RUN_DIR, 'Pre_injection_vs_Injection')
    OUTPUT_DIR_SALINE_IVA = os.path.join(RUN_DIR, 'Saline_vs_Ivabradine')
    for d in [RUN_DIR, OUTPUT_DIR_FILTERED, OUTPUT_DIR_RAW, OUTPUT_DIR_PRE_VS_TEST, OUTPUT_DIR_SALINE_IVA]:
        os.makedirs(d, exist_ok=True)

# ── Colors ──────────────────────────────────────────────────────────────────
# MOUSE_COLORS imported from path_for_expe_facemap
CONDITION_COLORS = {
    'Basal_Pre_Injection':         '#95a5a6',
    'Injection_Saline':            '#3498db',
    'Injection_Ivabradine_5mgkg':  '#f39c12',
    'Injection_Ivabradine_10mgkg': '#e67e22',
    'Injection_Ivabradine_20mgkg': '#e74c3c',
}
CONDITION_LABELS = {
    'Basal_Pre_Injection':         'Pre-injection',
    'Injection_Saline':            'Saline',
    'Injection_Ivabradine_5mgkg':  'Iva 5mg/kg',
    'Injection_Ivabradine_10mgkg': 'Iva 10mg/kg',
    'Injection_Ivabradine_20mgkg': 'Iva 20mg/kg',
}

# Temporal binning — 60s bins for temporal figures
# Reduces noise and makes curves comparable across sessions of different lengths
# Each bin = median of all non-NaN frames within that 60s window
BIN_SIZE_S = 60  

SIGNAL_LABELS = {
    'pupil_area_mm2':       'Pupil area (mm²)',  # calibrated using respiratory apparatus or eye keypoints
    'pupil_area_px2':       'Pupil area (px²)',  # raw pixel area
    'pupil_velocity':       'Pupil velocity (px/s)',
    'pupil_center_x':       'Pupil displacement X (px/s)',
    'pupil_center_y':       'Pupil displacement Y (px/s)',
    'whisker_I_velocity':   'Whisker I velocity (px/s)',
    'whisker_II_velocity':  'Whisker II velocity (px/s)',
    'whisker_III_velocity': 'Whisker III velocity (px/s)',
}
SIGNAL_KEYS = list(SIGNAL_LABELS.keys())  # includes pupil_area_norm

# ── Data extraction ──────────────────────────────────────────────────────────
def find_h5_from_pkl(pkl_path):
    """
    Find the h5 file corresponding to a given pkl path.
    
    Navigates 2 levels up from VF1/ folder to reach the session directory:
      pkl: .../Analyse pupille/VF1/*_FacemapPose.pkl
      h5:  .../*_FacemapPose.h5 (in session_dir, 2 levels above VF1/)
    
    Returns first matching h5 path, or None if not found.
    Used by extract_signals_raw() to load unfiltered keypoint data.
    """
    session_dir = os.path.dirname(os.path.dirname(os.path.dirname(pkl_path)))
    h5_files = glob.glob(os.path.join(session_dir, '*_FacemapPose.h5'))
    if h5_files:
        return h5_files[0]
    return None

def plot_box_or_bar(ax, values, position, color, width=0.4, alpha=0.6, hatch=None):
    """
    Plot boxplot if n>1, bar plot if n=1.
    Avoids displaying a lonely isolated point when only one session is available for a condition.
    hatch: optional fill pattern (e.g. '///' for raw data) to visually distinguish series.
    """
    if len(values) > 1:
        bp = ax.boxplot(values, positions=[position], widths=width,
                       patch_artist=True,
                       medianprops=dict(color='black', linewidth=2))
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(alpha)
        if hatch:
            bp['boxes'][0].set_hatch(hatch)
        return bp
    elif len(values) == 1:
        bar = ax.bar(position, values[0], width=width, color=color, alpha=alpha)
        if hatch:
            bar[0].set_hatch(hatch)
        ax.text(position, values[0] * 1.02, f'{values[0]:.2f}',
               ha='center', va='bottom', fontsize=7, color='black')
        return None
    return None

def extract_signals_raw(session):
    """Extract signals from the post-AnalyseFaceMap pkl (before PostProcessingGUI).
    If a post_processing pkl exists, this loads the original pkl instead.
    If no post_processing pkl exists, returns same as extract_signals.
    """
    pkl_path = session['pkl_path']
    if 'post_processing' in pkl_path:
        original_path = pkl_path.replace('_post_processing_FacemapPose.pkl', '_FacemapPose.pkl')
        if os.path.exists(original_path):
            try:
                with open(original_path, 'rb') as f:
                    data_raw = pickle.load(f)
                return extract_signals(data_raw)
            except Exception as e:
                print(f'⚠️  Could not load original pkl: {e}')
    return extract_signals(session['data'])

def compute_px_per_mm(data, session_name='unknown session'):
    """Fallback calibration — only used when px_per_mm_calibration is absent from pkl.
    Primary calibration uses the respiratory apparatus diameter GUI in PipelineFaceMap.py.
    Computes px/mm from median distance eye(front) to eye(back) divided by EYE_DIAMETER_MM (3mm).
    Returns (px_per_mm, warning_message or None)."""
    def get_coords(data, key):
        if key in data:
            x = np.array(data[key]['x'], dtype=float)
            y = np.array(data[key]['y'], dtype=float)
            return x, y
        return None, None

    xf, yf = get_coords(data, 'eye(front)')
    xb, yb = get_coords(data, 'eye(back)')
    xt, yt = get_coords(data, 'eye(top)')
    xbot, ybot = get_coords(data, 'eye(bottom)')

    # Horizontal axis: eye(front) to eye(back)
    EYE_DIAMETER_MM = 3
    if xf is not None and xb is not None:
        dist_h = np.sqrt((xf - xb)**2 + (yf - yb)**2) #euclidian distance
        valid_h = dist_h[~np.isnan(dist_h)]
        if len(valid_h) > 0:
            px_per_mm_h = np.median(valid_h) / EYE_DIAMETER_MM

    # Horizontal axis only: eye(front) to eye(back) = EYE_DIAMETER_MM (~3mm)
   
    warning = None
    if px_per_mm_h is not None:
        px_per_mm = px_per_mm_h
    else:
        px_per_mm = None
        warning = '❌ eye(front) or eye(back) keypoints unavailable — fallback calibration failed'

    return px_per_mm, warning

#────────────END OF FALLBACK FUNCTION────────────────────────────────────────────────────────────



def extract_signals(data):
    """
    Extract and compute all signals from a loaded pkl dictionary.
    
    Handles t=0 shift if injection_time_s is set (PostProcessingGUI split mode).
    Calibration priority: px_per_mm_calibration (respiratory apparatus GUI) > eye keypoints fallback.
    pupil_area_mm2 = NaN for all frames if calibration fails.
    Whisker velocity: frame-to-frame euclidean displacement * fps — frame 0 set to NaN.
    Returns (fps, frames, signals_dict).
    """
    fps = data['fps']
    frames = np.array(data['frames'])
    # Shift frames so t=0 corresponds to injection time if marked in PostProcessingGUI
    injection_time_s = data.get('injection_time_s', None)
    if injection_time_s is not None:
        frames = frames - (injection_time_s * fps)  # shift so injection = frame 0
    session_name = data.get('pkl_name', 'unknown session')  # set by caller if available

    # ── Calibration: respiratory apparatus (primary) or eye keypoints (fallback) ──
    px_per_mm_cal = data.get('px_per_mm_calibration', None)  # from pipeline GUI calibration
    if px_per_mm_cal is not None and px_per_mm_cal > 0:
        px_per_mm = px_per_mm_cal
        # print(f'  [{session_name}] Calibration: respiratory apparatus ({px_per_mm:.2f} px/mm)')
    else:
        # Fallback: eye keypoints method
        px_per_mm, _ = compute_px_per_mm(data, session_name=session_name)
        # print(f'  [{session_name}] Calibration: eye keypoints fallback ({px_per_mm:.2f} px/mm)')

    if px_per_mm is not None and px_per_mm > 0:
        pupil_area_mm2 = np.array(data['pupil_area'], dtype=float) / (px_per_mm ** 2)
    else:
        pupil_area_mm2 = np.full(len(np.array(data['pupil_area'])), np.nan)
    signals = {
        'pupil_area_mm2':  pupil_area_mm2,  # calibrated using respiratory apparatus or eye keypoints
        'pupil_area_px2':  np.array(data['pupil_area'], dtype=float),  # raw pixel area
        'pupil_velocity':  np.array(data['absolute_velocity'], dtype=float) * fps,  # px/frame → px/s
        'pupil_center_x': np.abs(np.diff(data['pupil_center_[x,y]'][:, 1], prepend=data['pupil_center_[x,y]'][0, 1])) * data['fps'],  # absolute horizontal displacement px/s
        'pupil_center_y': np.abs(np.diff(data['pupil_center_[x,y]'][:, 0], prepend=data['pupil_center_[x,y]'][0, 0])) * data['fps'],  # absolute vertical displacement px/s
    }
    for wh, key in [('whisker(I)', 'whisker_I_velocity'),
                    ('whisker(II)', 'whisker_II_velocity'),
                    ('whisker(III)', 'whisker_III_velocity')]:
        if wh in data:
            x = np.array(data[wh]['x'], dtype=float)
            y = np.array(data[wh]['y'], dtype=float)
            dx = np.diff(x, prepend=x[0])
            dy = np.diff(y, prepend=y[0])
            vel = np.sqrt(dx**2 + dy**2) * fps
            vel[0] = np.nan
            signals[key] = vel
        else:
            signals[key] = np.full(len(frames), np.nan) #if point doesn't exists or is filtered = replaces by NaN
    return fps, frames, signals

def bin_signal(signal, frames, bin_size_s=BIN_SIZE_S):
    """
    Bin a signal into fixed-size time windows and compute median per bin.
    
    Used to reduce noise and make temporal curves comparable across sessions of different lengths.
    Bins with no valid (non-NaN) frames remain NaN — no interpolation.
    Returns (bin_centers_minutes, binned_medians).
    Last bin may be smaller than bin_size_s if session length is not a multiple of bin_size_s.
    """
   
    if len(frames) == 0:
        return np.array([]), np.array([]) #Protection against empty videos 
    max_t = frames[-1]
    bin_edges = np.arange(0, max_t + bin_size_s, bin_size_s) #creating the bins in the video to sum up information of close frames, by defining the edges
    #NB : the last bin will be smaller because it ends at max_t 
    bin_centers = bin_edges[:-1] + bin_size_s / 2 # Define the center of the bins but without the last edge from the last bin to have nb_of_bins = nb_of_centers
    binned = np.full(len(bin_centers), np.nan) #allows the bin to keep the NaN value if they contain no valid datas
    for i, (t0, t1) in enumerate(zip(bin_edges[:-1], bin_edges[1:])): #loops on every bins by creating the pairs with the coordinates of the edges, ex :(0, 60), (60, 120), (120, 180)
        mask = (frames >= t0) & (frames < t1) #Mask that filters the last frame from the n-bin in order to count it just once as the first frame of the (n+1)-bin
        vals = signal[mask]  
        valid = vals[~np.isnan(vals)] #invert the NaN on all the valid frames
        if len(valid) > 0: 
            binned[i] = np.nanmedian(valid) #if no data is valid in the i-bin binned[i] remains NaN but if there's at least 1 valid value = calculates the median of the bin
    return bin_centers / 60, binned #return the coordinates of each bin center

#In litterature the bins used for long videos 

def compute_median_iqr(sessions, sig_key, bin_size_s=BIN_SIZE_S):
    """
    Compute median and IQR of a signal across sessions on a common time grid.
    
    Each session is binned then interpolated onto the common grid (max session duration).
    Interpolation allows comparison despite different FPS, bin centers, or session lengths.
    Bins beyond a session's end are set to NaN — not extrapolated.
    n_per_bin: number of sessions contributing to each bin — decreases toward end of grid.
    Returns (t_grid_minutes, median, iqr, n_per_bin).
    """
    max_dur_min = 0
    for session in sessions:
        fps, frames, signals = extract_signals(session['data'])
        if len(frames) > 0:
            dur = frames[-1] / 60
            if dur > max_dur_min: #Search across all session the longest to define the x-axis
                max_dur_min = dur

    t_grid = np.arange(bin_size_s / 2 / 60, max_dur_min, bin_size_s / 60) #start the grid in the first bin center
    if len(t_grid) == 0: #protects against errors if every session is empty
        return np.array([]), np.array([]), np.array([]), np.array([])

    all_binned = np.full((len(sessions), len(t_grid)), np.nan) #grid starts by setting everything to NaN because it's easier to "un-NaN" the valid data since the majority of the short videos datas will be at NaN 
    for i, session in enumerate(sessions):
        fps, frames, signals = extract_signals(session['data'])
        if sig_key not in signals: #if the keypoints doesn't exist in this session, continue
            continue
        t_min, binned = bin_signal(signals[sig_key], frames, bin_size_s)
        valid = ~np.isnan(binned)
        if np.sum(valid) < 2: #if there're less than 2 valid bins, no interpolation will be made
            continue
        interp = np.interp(t_grid, t_min[valid], binned[valid], left=np.nan, right=np.nan) #linear interpolation that allows to compare videos even if : the bin center aren't matching, the FPS are changing, the bin size are variable
        max_t_session = frames[-1] / 60 if len(frames) > 0 else 0
        interp[t_grid > max_t_session] = np.nan
        all_binned[i] = interp
    n_per_bin = np.sum(~np.isnan(all_binned), axis=0) #counts number of sessions in each bins : n=nb of session a the beginning and n=1 at the end (only the longest video remains)

    median = np.nanmedian(all_binned, axis=0)
    iqr = np.array([np.percentile(all_binned[:, j][~np.isnan(all_binned[:, j])], 75) - np.percentile(all_binned[:, j][~np.isnan(all_binned[:, j])], 25) if np.sum(~np.isnan(all_binned[:, j])) > 1 else np.nan for j in range(all_binned.shape[1])])
    #using median and IQR = InterQuartile Range
    median[n_per_bin == 0] = np.nan
    iqr[n_per_bin == 0] = np.nan #forces the bins without any valid data to be NaN 
    return t_grid, median, iqr, n_per_bin

def session_median(signal):
    """Compute median of a signal ignoring NaN values.
       Returns NaN if all values are NaN (empty or fully filtered session)."""
    valid = signal[~np.isnan(signal)]
    return np.nanmedian(valid) if len(valid) > 0 else np.nan

# ── Figure 1 — Temporal curves ────────────────────────────────────────────────
def plot_temporal(sessions_by_condition, title, output_path, color_map=None, label_map=None):
    """Plot median ± IQR/2 temporal curves for each condition and signal.
    One subplot per signal (SIGNAL_KEYS), all conditions overlaid per subplot.
    X-axis: time in minutes (binned at BIN_SIZE_S seconds).
    Shaded area: IQR/2 around median (not standard error — robust to outliers).
    Legend deduplicated — only one entry per condition label.
    X tick labels hidden on all subplots except the last for readability."""
    if color_map is None:
        color_map = CONDITION_COLORS
    if label_map is None:
        label_map = CONDITION_LABELS

    n_signals = len(SIGNAL_KEYS)
    fig, axs = plt.subplots(n_signals, 1, figsize=(20, 3 * n_signals))
    fig.subplots_adjust(top=0.94, hspace=0.35)
    if n_signals == 1: #protection if n=1
        axs = [axs]

    for si, (ax, sig_key) in enumerate(zip(axs, SIGNAL_KEYS)): #loops on every subplots to extract the data from each keypoint
        for condition, sessions in sessions_by_condition.items():
            color = color_map.get(condition, '#000000')
            label = label_map.get(condition, condition)

            t_grid, median, iqr, n_per_bin = compute_median_iqr(sessions, sig_key)
            if len(t_grid) == 0: #if no data, passes
                continue

            ax.plot(t_grid, median, color=color, linewidth=1.5, label=label)
            ax.fill_between(t_grid,
                           np.where(np.isnan(iqr), np.nan, median - iqr / 2), #draws the spreading zone of the temporal data 
                           np.where(np.isnan(iqr), np.nan, median + iqr / 2),
                           color=color, alpha=0.2)

        ax.set_ylabel(SIGNAL_LABELS[sig_key], fontsize=8)
        ax.grid(False)
        # X label only on last subplot
        if si == n_signals - 1:
            ax.set_xlabel('Time (min)', fontsize=8)
        else:
            ax.set_xticklabels([])

    # Legend on first subplot only
    handles, labels = axs[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axs[0].legend(unique.values(), unique.keys(), fontsize=7,
                  bbox_to_anchor=(1.01, 1), loc='upper left')

    fig.suptitle(title, fontsize=12, y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


# ── Figure 5 — Individual temporal curves ────────────────────────────────────
def plot_temporal_individual(sessions_by_condition, title, output_path,
                              color_map=None, label_map=None):
    """Plot individual session curves with condition median overlaid.
        Individual curves: alpha=0.3, linewidth=0.8 — faded to show spread.
        Median curve: alpha=1.0, linewidth=2.0 — prominent on top.
        Same layout as plot_temporal — one subplot per signal, conditions overlaid.
        Useful for identifying outlier sessions that distort the median."""
    if color_map is None:
        color_map = CONDITION_COLORS
    if label_map is None:
        label_map = CONDITION_LABELS

    n_signals = len(SIGNAL_KEYS)
    fig, axs = plt.subplots(n_signals, 1, figsize=(20, 3 * n_signals))
    fig.subplots_adjust(top=0.94, hspace=0.35)
    if n_signals == 1:
        axs = [axs]

    for si, (ax, sig_key) in enumerate(zip(axs, SIGNAL_KEYS)):
        for condition, sessions in sessions_by_condition.items():
            color = color_map.get(condition, '#000000')
            label = label_map.get(condition, condition)

            # Individual curves
            for session in sessions:
                fps, frames, signals = extract_signals(session['data'])
                if sig_key not in signals:
                    continue
                t_min, binned = bin_signal(signals[sig_key], frames)
                ax.plot(t_min, binned, color=color, linewidth=0.8,
                       alpha=0.3)

            # Median on top
            t_grid, median, iqr, n_per_bin = compute_median_iqr(sessions, sig_key)
            if len(t_grid) > 0:
                ax.plot(t_grid, median, color=color, linewidth=2.0,
                       label=label)

        ax.set_ylabel(SIGNAL_LABELS[sig_key], fontsize=8)
        ax.grid(False)
        if si == n_signals - 1:
            ax.set_xlabel('Time (min)', fontsize=8)
        else:
            ax.set_xticklabels([])

    handles, labels = axs[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axs[0].legend(unique.values(), unique.keys(), fontsize=7,
                  bbox_to_anchor=(1.01, 1), loc='upper left')

    fig.suptitle(title + ' — Individual sessions', fontsize=12, y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')

# ── Figure 2 — Box plots + error bars ────────────────────────────────────────
def plot_summary(sessions_by_condition, title, output_path, color_map=None,
                 label_map=None, point_color_key='mouse'):
    """Plot summary box plots with individual session points and statistical tests.
    
    One subplot per signal, one box plot per condition.
    Individual points: colored by mouse (MOUSE_COLORS) — each point = one session median.
    Diamond marker (D): median ± IQR of session medians across all sessions of the condition.
    
    Statistical tests:
      - Aggregated by mouse first (median of sessions per mouse) — mouse is the statistical unit
      - Wilcoxon signed-rank (paired) or Mann-Whitney (independent) depending on point_color_key
      - FDR correction applied across all comparisons
      - p-values saved as CSV alongside the figure
    
    Session count annotation per condition shown below x-axis (mouse: n_sessions format).
    Legend: condition colors (patches) + mouse colors (points)."""
    if color_map is None:
        color_map = CONDITION_COLORS
    if label_map is None:
        label_map = CONDITION_LABELS

    n_signals = len(SIGNAL_KEYS)
    fig, axs = plt.subplots(n_signals, 1, figsize=(14, 3 * n_signals))
    fig.subplots_adjust(top=0.94, hspace=0.5)
    if n_signals == 1:
        axs = [axs]

    conditions = list(sessions_by_condition.keys())
    x_pos = np.arange(len(conditions)) #position of the box plots on the X-axis

    for ax, sig_key in zip(axs, SIGNAL_KEYS):
        for xi, condition in enumerate(conditions):
            sessions = sessions_by_condition[condition]
            color = color_map.get(condition, '#000000')

            medians = []
            mouse_labels = []
            for session in sessions:
                fps, frames, signals = extract_signals(session['data'])
                if sig_key in signals:
                    m = session_median(signals[sig_key])
                    if not np.isnan(m):
                        medians.append(m)
                        mouse_labels.append(session['mouse'])

            if len(medians) == 0:
                continue

            medians = np.array(medians)

            if len(medians) > 1:
                bp = plot_box_or_bar(ax, list(medians), xi, color, width=0.4, alpha=0.6)
            
            # Individual points colored by mouse
            for j, (m, mouse) in enumerate(zip(medians, mouse_labels)):
                jitter = np.random.uniform(-0.08, 0.08)
                pt_color = MOUSE_COLORS.get(mouse, '#000000') if point_color_key == 'mouse' else color
                ax.scatter(xi + jitter, m, color=pt_color, s=25, zorder=5, alpha=0.9)

            # Median ± IQR
            med = np.median(medians)
            q25, q75 = np.percentile(medians, [25, 75]) if len(medians) > 1 else (med, med)
            ax.errorbar(xi + 0.3, med, yerr=[[med - q25], [q75 - med]],
                       fmt='D', color='black', markersize=6, capsize=5,
                       linewidth=2, zorder=6)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([label_map.get(c, c) for c in conditions],
                           fontsize=8, rotation=15)
        ax.set_ylabel(SIGNAL_LABELS[sig_key], fontsize=8)
        ax.grid(False)

        # Statistical tests — aggregate by mouse first (one median per mouse per condition)
        # This ensures independence: mouse is the statistical unit, not the session
        group_values = {}
        session_counts = {}  # {condition: {mouse: n_sessions}}
        for condition, sessions in sessions_by_condition.items():
            mouse_medians = {}
            mouse_n = {}
            for session in sessions:
                mouse = session['mouse']
                fps, frames, signals = extract_signals(session['data'])
                if sig_key in signals:
                    m = session_median(signals[sig_key])
                    if not np.isnan(m):
                        if mouse not in mouse_medians:
                            mouse_medians[mouse] = []
                            mouse_n[mouse] = 0
                        mouse_medians[mouse].append(m)
                        mouse_n[mouse] += 1
            # One median per mouse
            group_values[condition] = [np.median(v) for v in mouse_medians.values()]
            session_counts[condition] = mouse_n
        paired = (point_color_key == 'mouse')
        stat_results = compute_stats_from_values(group_values, paired=paired)
        stat_results = apply_fdr_correction(stat_results)
        add_stats_to_ax(ax, list(x_pos), stat_results, sig_key, label_map)
        stats_csv_path = output_path.replace('.png', f'_{sig_key}_stats.csv')
        save_stats_table(stat_results, stats_csv_path, sig_key)
        # Add session count annotation per condition
        for xi, condition in enumerate(conditions):
            if condition in session_counts:
                counts_str = ' '.join([f'{m}:n={n}' for m, n in sorted(session_counts[condition].items())])
                ax.annotate(counts_str, xy=(xi, ax.get_ylim()[0]),
                           xycoords=('data', 'data'),
                           ha='center', va='top', fontsize=5, color='grey', style='italic')

    # Legend — conditions (colored boxes) + mice (colored points)
    condition_handles = [mpatches.Patch(color=color_map.get(c, '#000000'),
                        label=label_map.get(c, c)) for c in conditions]
    mouse_handles = [plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=MOUSE_COLORS[m], markersize=7, label=m)
                    for m in MOUSE_COLORS if m in set(
                    ml for s in sessions_by_condition.values()
                    for ml in [sess['mouse'] for sess in s])]
    axs[0].legend(handles=condition_handles + mouse_handles, fontsize=7,
                 bbox_to_anchor=(1.01, 1), loc='upper left')

    fig.suptitle(title, fontsize=12, y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')

# ── Figure 3 — Post-processed vs Post-AnalyseFaceMap temporal ─────────────────────────────────────
def plot_temporal_raw_vs_postprocessed(sessions_by_condition, title, output_path,
                                   color_map=None, label_map=None):
    """Compare temporal curves between post-AnalyseFaceMap (raw) and post-processed data.
    
    Solid line (—): post-processed data (extract_signals) with IQR/2 shading.
    Dashed line (--): post-AnalyseFaceMap data (extract_signals_raw) — no shading.
    Both computed as median across sessions on independent time grids.
    
    Purpose: assess impact of PostProcessingGUI on the signals.
    If raw == post-processed for a session: no post-processing was applied (same pkl used)."""
    if color_map is None:
        color_map = CONDITION_COLORS
    if label_map is None:
        label_map = CONDITION_LABELS

    n_signals = len(SIGNAL_KEYS)
    fig, axs = plt.subplots(n_signals, 1, figsize=(20, 3 * n_signals))
    fig.subplots_adjust(top=0.94, hspace=0.35)
    if n_signals == 1:
        axs = [axs]

    legend_entries = {}

    for si, (ax, sig_key) in enumerate(zip(axs, SIGNAL_KEYS)):
        for condition, sessions in sessions_by_condition.items():
            color = color_map.get(condition, '#000000')
            label = label_map.get(condition, condition)

            # Filtered
            t_grid, median, iqr, _ = compute_median_iqr(sessions, sig_key)
            if len(t_grid) > 0:
                ax.plot(t_grid, median, color=color, linewidth=1.5,
                       linestyle='-', label=f'{label} (filtered)')
                ax.fill_between(t_grid,
                               np.where(np.isnan(iqr), np.nan, median - iqr / 2),
                               np.where(np.isnan(iqr), np.nan, median + iqr / 2),
                               color=color, alpha=0.15)

            # Raw — compute mean/sem manually using extract_signals_raw
            all_binned_raw = []
            max_dur_min = 0
            for session in sessions:
                fps_r, frames_r, signals_r = extract_signals_raw(session)
                if fps_r is None or sig_key not in signals_r:
                    continue
                dur = frames_r[-1] / 60 if len(frames_r) > 0 else 0
                if dur > max_dur_min:
                    max_dur_min = dur

            if max_dur_min > 0:
                t_grid_r = np.arange(BIN_SIZE_S / 2 / 60, max_dur_min, BIN_SIZE_S / 60)
                all_binned_r = np.full((len(sessions), len(t_grid_r)), np.nan)
                for i, session in enumerate(sessions):
                    fps_r, frames_r, signals_r = extract_signals_raw(session)
                    if fps_r is None or sig_key not in signals_r:
                        continue
                    t_r, b_r = bin_signal(signals_r[sig_key], frames_r)
                    valid = ~np.isnan(b_r)
                    if np.sum(valid) < 2:
                        continue
                    interp = np.interp(t_grid_r, t_r[valid], b_r[valid],
                                      left=np.nan, right=np.nan)
                    max_t_s = frames_r[-1] / 60
                    interp[t_grid_r > max_t_s] = np.nan
                    all_binned_r[i] = interp

                mean_r = np.nanmedian(all_binned_r, axis=0)
                ax.plot(t_grid_r, mean_r, color=color, linewidth=1.0,
                       linestyle='--', alpha=0.7, label=f'{label} (raw)')

        ax.set_ylabel(SIGNAL_LABELS[sig_key], fontsize=8)
        ax.grid(False)
        if si == n_signals - 1:
            ax.set_xlabel('Time (min)', fontsize=8)
        else:
            ax.set_xticklabels([])

    handles, labels_leg = axs[0].get_legend_handles_labels()
    unique = dict(zip(labels_leg, handles))
    axs[0].legend(unique.values(), unique.keys(), fontsize=6,
                  bbox_to_anchor=(1.01, 1), loc='upper left')

    fig.suptitle(title + ' — Filtered (—) vs Raw (--)', fontsize=12, y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')

# ── Figure 4 — Post-processed vs Post-AnalyseFaceMap summary (box plots) ──────────────────────────
def plot_summary_raw_vs_postprocessed(sessions_by_condition, title, output_path,
                                  color_map=None, label_map=None):
    """Compare session median box plots between post-processed and post-AnalyseFaceMap data.
 
    For each condition: two adjacent box plots (filtered solid, raw hatched ///).
    Individual points: colored by mouse for filtered, grey for raw.
    Statistical test: paired Wilcoxon signed-rank between filtered and raw medians per condition.
 
    Layout: conditions spaced by 3 units, filtered at xi*3, raw at xi*3+1.
    Purpose: quantify impact of PostProcessingGUI — if filtered == raw, no changes were made."""
    if color_map is None:
        color_map = CONDITION_COLORS
    if label_map is None:
        label_map = CONDITION_LABELS

    n_signals = len(SIGNAL_KEYS)
    fig, axs = plt.subplots(n_signals, 1, figsize=(14, 3 * n_signals))
    fig.subplots_adjust(top=0.94, hspace=0.5)
    if n_signals == 1:
        axs = [axs]

    conditions = list(sessions_by_condition.keys())
    n_cond = len(conditions)
    # For each condition: filtered box at xi*3, raw box at xi*3+1, gap of 1
    x_filtered = np.arange(n_cond) * 3
    x_raw = np.arange(n_cond) * 3 + 1

    for ax, sig_key in zip(axs, SIGNAL_KEYS):
        for xi, condition in enumerate(conditions):
            sessions = sessions_by_condition[condition]
            color = color_map.get(condition, '#000000')

            # Filtered medians
            medians_filt = []
            mouse_labels = []
            for session in sessions:
                fps, frames, signals = extract_signals(session['data'])
                if sig_key in signals:
                    m = session_median(signals[sig_key])
                    if not np.isnan(m):
                        medians_filt.append(m)
                        mouse_labels.append(session['mouse'])

            # Raw medians
            medians_raw = []
            for session in sessions:
                fps_r, frames_r, signals_r = extract_signals_raw(session)
                if fps_r is not None and sig_key in signals_r:
                    m = session_median(signals_r[sig_key])
                    if not np.isnan(m):
                        medians_raw.append(m)

            for xpos, medians, alpha, hatch, lbl in [
                (x_filtered[xi], medians_filt, 0.7, None, 'filtered'),
                (x_raw[xi], medians_raw, 0.3, '///', 'raw'),
            ]:
                if len(medians) == 0:
                    continue
                medians = np.array(medians)
                if len(medians) > 1:
                    bp = plot_box_or_bar(ax, list(medians), xpos, color, width=0.7, alpha=alpha, hatch=hatch)
                # Individual points
                for m, mouse in zip(medians, mouse_labels if lbl == 'filtered' else ['']*len(medians)):
                    jitter = np.random.uniform(-0.1, 0.1)
                    pt_color = MOUSE_COLORS.get(mouse, color) if lbl == 'filtered' else '#888888'
                    ax.scatter(xpos + jitter, m, color=pt_color, s=20, zorder=5, alpha=0.8)
                # Median ± IQR
                med = np.median(medians)
                q25, q75 = np.percentile(medians, [25, 75]) if len(medians) > 1 else (med, med)
                ax.errorbar(xpos + 0.4, med, yerr=[[med - q25], [q75 - med]], fmt="D",
                           color="black", markersize=5, capsize=4, linewidth=1.5, zorder=6)

        # X ticks between filtered and raw pairs
        tick_pos = (x_filtered + x_raw) / 2
        ax.set_xticks(tick_pos)
        ax.set_xticklabels([label_map.get(c, c) for c in conditions], fontsize=8, rotation=15)
        ax.set_ylabel(SIGNAL_LABELS[sig_key], fontsize=8)
        ax.grid(False)

        # Stats — filtered vs raw (paired Wilcoxon per condition)
        for xi, condition in enumerate(conditions):
            sessions = sessions_by_condition[condition]
            vals_filt, vals_raw = [], []
            for session in sessions:
                fps, frames, signals = extract_signals(session['data'])
                if sig_key in signals:
                    m_f = session_median(signals[sig_key])
                    if not np.isnan(m_f): vals_filt.append(m_f)
                fps_r, frames_r, signals_r = extract_signals_raw(session)
                if fps_r is not None and sig_key in signals_r:
                    m_r = session_median(signals_r[sig_key])
                    if not np.isnan(m_r): vals_raw.append(m_r)
            if len(vals_filt) >= 2 and len(vals_raw) >= 2 and len(vals_filt) == len(vals_raw):
                from scipy import stats as _stats
                _, pval = _stats.wilcoxon(vals_filt, vals_raw)
                stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
                y_max = ax.get_ylim()[1]
                y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                x_mid = (x_filtered[xi] + x_raw[xi]) / 2
                ax.plot([x_filtered[xi], x_filtered[xi], x_raw[xi], x_raw[xi]],
                       [y_max + y_range*0.03, y_max + y_range*0.06,
                        y_max + y_range*0.06, y_max + y_range*0.03],
                       color='black', linewidth=0.8)
                ax.text(x_mid, y_max + y_range*0.07, stars, ha='center', fontsize=8)
                ax.set_ylim(top=y_max + y_range*0.15)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', alpha=0.7, label='Post-processed'),
        Patch(facecolor='gray', alpha=0.3, hatch='///', label='Post-AnalyseFaceMap'),
    ]
    mouse_handles = [mpatches.Patch(color=MOUSE_COLORS[m], label=m) for m in MOUSE_COLORS]
    axs[0].legend(handles=legend_elements + mouse_handles, fontsize=7,
                 bbox_to_anchor=(1.01, 1), loc='upper left')

    fig.suptitle(title + ' — Post-processed vs Post-AnalyseFaceMap (box plots)', fontsize=12, y=0.98)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


# ── Figure 6 — All sessions individual curves ────────────────────────────────
def plot_all_sessions(all_data, output_dir):
    """Plot one curve per session for each signal — all conditions and mice combined.
    
    One figure per signal key, all sessions overlaid on the same axis.
    Color = condition color, label = mouse + session name.
    Useful for identifying outlier sessions or verifying data completeness.
    One PNG per signal saved in output_dir."""
    
    # Flatten all sessions with condition info
    all_sessions = []
    for condition, sessions in all_data.items():
        for s in sessions:
            s_copy = dict(s)
            s_copy['condition'] = condition
            all_sessions.append(s_copy)

    for sig_key, sig_label in SIGNAL_LABELS.items():
        fig, ax = plt.subplots(figsize=(25, 6))
        for s in all_sessions:
            fps, frames, signals = extract_signals(s['data'])
            if sig_key not in signals or len(frames) == 0:
                continue
            t_min, binned = bin_signal(signals[sig_key], frames)
            label = f"{s['mouse']} {os.path.basename(s['pkl_path']).replace('_FacemapPose.pkl', '')}"
            color = CONDITION_COLORS.get(s['condition'], '#000000')
            ax.plot(t_min, binned, linewidth=0.8, alpha=0.6, color=color, label=label)
        ax.set_xlabel('Time (min)')
        ax.set_ylabel(sig_label)
        ax.set_title(f'All sessions — {sig_label}')
        ax.legend(fontsize=5, bbox_to_anchor=(1.01, 1), loc='upper left')
        ax.grid(False)
        out = os.path.join(output_dir, f'all_sessions_{sig_key}.png')
        plt.savefig(out, dpi=120, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out}')


# ── Figure 7 — Interactive HTML all sessions ─────────────────────────────────
def plot_all_sessions_html(all_data, output_dir):
    """Interactive Plotly version of plot_all_sessions — one curve per session, toggleable.
   
   All signals in shared-x subplots — zooming one zooms all simultaneously.
   Legend grouped by session (legendgroup) — click to show/hide individual sessions.
   Hover shows session name, time and value.
   Saved as all_sessions_interactive.html — open in browser, no Python required."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    all_sessions = []
    for condition, sessions in all_data.items():
        for s in sessions:
            s_copy = dict(s)
            s_copy['condition'] = condition
            all_sessions.append(s_copy)

    n_signals = len(SIGNAL_KEYS)
    fig = make_subplots(rows=n_signals, cols=1,
                        subplot_titles=list(SIGNAL_LABELS.values()),
                        shared_xaxes=True,
                        vertical_spacing=0.03)

    # Track which labels already have a legend entry
    legend_shown = set()

    for s in all_sessions:
        fps, frames, signals = extract_signals(s['data'])
        condition = s['condition']
        color = CONDITION_COLORS.get(condition, '#000000')
        label = CONDITION_LABELS.get(condition, condition)
        session_name = os.path.basename(s['pkl_path']).replace('_FacemapPose.pkl', '')
        full_label = f"{s['mouse']} — {session_name}"
        show_legend = label not in legend_shown
        if show_legend:
            legend_shown.add(label)

        for ri, sig_key in enumerate(SIGNAL_KEYS):
            if sig_key not in signals or len(frames) == 0:
                continue
            t_min, binned = bin_signal(signals[sig_key], frames)
            valid = ~np.isnan(binned)
            fig.add_trace(
                go.Scatter(
                    x=t_min[valid],
                    y=binned[valid],
                    mode='lines',
                    name=full_label,
                    legendgroup=full_label,
                    showlegend=(ri == 0),
                    line=dict(color=color, width=1),
                    opacity=0.7,
                    hovertemplate=f'{full_label}<br>t=%{{x:.1f}} min<br>val=%{{y:.2f}}<extra></extra>',
                ),
                row=ri + 1, col=1
            )

    fig.update_layout(
        height=250 * n_signals,
        title_text='All sessions — Interactive (click legend to toggle)',
        legend=dict(
            groupclick='toggleitem',  # click once = toggle individual session
            font=dict(size=9),
            tracegroupgap=2,
        ),
        # Zoom with mouse wheel + autorange on double-click
        dragmode='zoom',
        hovermode='x unified',  # show all traces at same x on hover
        modebar_add=['drawrect', 'eraseshape'],  # allow drawing comparison zones
    )
    # Enable scroll zoom and autorange per subplot
    fig.update_xaxes(
        title_text='Time (min)', row=n_signals, col=1,
        rangeslider=dict(visible=True, thickness=0.04),  # mini-slider at bottom for navigation
    )
    for ri in range(1, n_signals + 1):
        fig.update_xaxes(
            fixedrange=False,  # allow zoom on x
            row=ri, col=1
        )
        fig.update_yaxes(
            fixedrange=False,  # allow zoom on y
            autorange=True,    # auto-adjust y range when zooming x
            row=ri, col=1
        )

    # Add buttons to toggle all/none and compare two conditions
    conditions_available = list(set(s['condition'] for s in all_sessions))
    buttons = [
        dict(label='Show all', method='restyle',
             args=[{'visible': True}]),
        dict(label='Hide all', method='restyle',
             args=[{'visible': 'legendonly'}]),
    ]
    for cond in conditions_available:
        cond_label = CONDITION_LABELS.get(cond, cond)
        # Build visibility list: True for matching condition, legendonly for others
        vis = []
        for s in all_sessions:
            v = True if s['condition'] == cond else 'legendonly'
            vis.extend([v] * n_signals)
        buttons.append(dict(
            label=f'Only {cond_label}',
            method='restyle',
            args=[{'visible': vis}]
        ))

    fig.update_layout(
        updatemenus=[dict(
            type='buttons',
            direction='left',
            buttons=buttons,
            pad=dict(r=10, t=10),
            showactive=False,
            x=0.0, xanchor='left',
            y=1.02, yanchor='bottom',
        )]
    )

    out = os.path.join(output_dir, 'all_sessions_interactive.html')
    fig.write_html(out, config={
        'scrollZoom': True,         # enable mouse wheel zoom
        'displayModeBar': True,
        'modeBarButtonsToAdd': ['drawrect'],
        'toImageButtonOptions': {'format': 'png', 'scale': 2},
    })
    print(f'Saved: {out}')


import re as _re_global

def get_date_key(pkl_path):
    # Extract date key F<YYYYMMDD> from pkl filename for day-level session matching
    m = _re_global.search(r'F(\d{8})', os.path.basename(pkl_path))
    return m.group(0) if m else None

def get_parent_folder(pkl_path):
    # Extract timestamped folder (YYYY-MM-DD_HH-MM-SS) from pkl path for exact session matching
    for part in reversed(pkl_path.split('/')):
        if _re_global.match(r'\d{4}-\d{2}-\d{2}_', part):
            return part
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(pkl_path)))))

# ── Pre-injection vs Test comparison ─────────────────────────────────────────
def plot_pre_vs_test(all_data, output_dir):
    """Plot paired Pre-injection vs Test session comparison per mouse.
    
    Pairing logic: matches sessions by (mouse, date key F<YYYYMMDD>, parent timestamp folder).
    This ensures Pre-injection and Test from the same recording day are paired together.
    
    TFor each paired session (Pre-injection + Test on same day):
    - Figure 1: Temporal curves superimposed (Pre in grey, Test in condition color)
    - Figure 2: Ratio Test/Pre-injection per signal (bar chart)
    One PNG per mouse grouping all sessions.
    
    get_date_key(): extracts F<YYYYMMDD> from pkl filename — day-level matching.
    get_parent_folder(): extracts timestamped folder (YYYY-MM-DD_HH-MM-SS) for exact day matching
      when multiple sessions exist on the same date.
    
    Only mice present in Basal_Pre_Injection AND at least one test condition are plotted."""


    out_dir = output_dir

    # Index Pre-injection sessions by (mouse, date, parent_folder)
    pre_index = {}
    for s in all_data['Basal_Pre_Injection']:
        date_key = get_date_key(s['pkl_path'])
        parent = get_parent_folder(s['pkl_path'])
        if date_key:
            pre_index[(s['mouse'], date_key, parent)] = s

    # Find pairs
    test_conditions = [c for c in EXPERIMENTS if c != 'Basal_Pre_Injection']
    pairs = []
    for condition in test_conditions:
        for s in all_data[condition]:
            date_key = get_date_key(s['pkl_path'])
            parent = get_parent_folder(s['pkl_path'])
            key = (s['mouse'], date_key, parent)
            if key in pre_index:
                pairs.append((pre_index[key], s, condition))

    print(f'\n{len(pairs)} Pre/Test pairs found')

    for mouse in MICE:
        mouse_pairs = [(pre, test, cond) for pre, test, cond in pairs if pre['mouse'] == mouse]
        if not mouse_pairs:
            continue

        n_sessions = len(mouse_pairs)
        n_signals = len(SIGNAL_KEYS)

        # ── Figure 1 — Temporal ──────────────────────────────────────────
        # Grid: n_signals rows × n_sessions columns
        # axs reshape handles edge cases: 1 signal or 1 session (2D array always)
        # Pre-injection: grey (#95a5a6) — Test: condition color
        # Y-axis unified per signal row (same scale across all sessions of the mouse)
        # for comparability — ymin/ymax taken from all subplots in the row
        # Legend deduplicated — shown once in top-left subplot
        fig, axs = plt.subplots(n_signals, n_sessions,
                                figsize=(6 * n_sessions, 3 * n_signals))
        fig.subplots_adjust(top=0.93, hspace=0.4, wspace=0.3)
        if n_sessions == 1 and n_signals == 1:
            axs = np.array([[axs]])
        elif n_sessions == 1:
            axs = axs.reshape(n_signals, 1)
        elif n_signals == 1:
            axs = axs.reshape(1, n_sessions)

        for col, (pre_s, test_s, condition) in enumerate(mouse_pairs):
            fps_p, frames_p, signals_p = extract_signals(pre_s['data'])
            fps_t, frames_t, signals_t = extract_signals(test_s['data'])
            color = CONDITION_COLORS.get(condition, '#000000')
            cond_label = CONDITION_LABELS.get(condition, condition)
            date_key = get_date_key(test_s['pkl_path'])
            axs[0, col].set_title(f'{date_key}\n{cond_label}', fontsize=8)

            for row, sig_key in enumerate(SIGNAL_KEYS):
                ax = axs[row, col]
                if sig_key in signals_p and len(frames_p) > 0:
                    t_p, b_p = bin_signal(signals_p[sig_key], frames_p)
                    ax.plot(t_p, b_p, color='#95a5a6', linewidth=1.2, label='Pre-injection')
                if sig_key in signals_t and len(frames_t) > 0:
                    t_t, b_t = bin_signal(signals_t[sig_key], frames_t)
                    ax.plot(t_t, b_t, color=color, linewidth=1.2, label=cond_label)
                if row == n_signals - 1:
                    ax.set_xlabel('Time (min)', fontsize=7)
                if col == 0:
                    ax.set_ylabel(SIGNAL_LABELS[sig_key], fontsize=7)
                ax.tick_params(labelsize=6)
                ax.grid(False)

        # Force same y-axis scale per signal row
        for row in range(n_signals):
            all_ylims = [axs[row, col].get_ylim() for col in range(n_sessions)]
            ymin = min(y[0] for y in all_ylims)
            ymax = max(y[1] for y in all_ylims)
            for col in range(n_sessions):
                axs[row, col].set_ylim(ymin, ymax)

        handles, labels_leg = axs[0, 0].get_legend_handles_labels()
        unique = dict(zip(labels_leg, handles))
        axs[0, 0].legend(unique.values(), unique.keys(), fontsize=6,
                         bbox_to_anchor=(0, 1.3), loc='upper left')
        fig.suptitle(f'{mouse} — Pre-injection vs Test', fontsize=12, y=0.98)
        out = os.path.join(out_dir, f'{mouse}_pre_vs_test_temporal.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out}')

         # ── Figure 2: Ratio Test/Pre-injection ───────────────────────────────────
        # One bar per session pair — height = session_median(test) / session_median(pre)
        # ratio = 1.0 (dashed line) = no effect vs pre-injection
        # Bar color = condition color, value shown above bar
        # Stats: Wilcoxon signed-rank of all ratios vs 1.0 (H0: no effect)
        # — tests whether the condition systematically changes the signal vs pre-injection
        # X labels: date + condition name, rotated 30° for readability
        # Ratios computed twice (bar + stats) — could be optimized but kept for clarity
        fig, axs_sum = plt.subplots(1, n_signals, figsize=(3 * n_signals, 5))
        if n_signals == 1:
            axs_sum = [axs_sum]
        fig.subplots_adjust(top=0.88, wspace=0.4)

        for si, sig_key in enumerate(SIGNAL_KEYS):
            ax = axs_sum[si]
            for xi, (pre_s, test_s, condition) in enumerate(mouse_pairs):
                fps_p, frames_p, signals_p = extract_signals(pre_s['data'])
                fps_t, frames_t, signals_t = extract_signals(test_s['data'])
                color = CONDITION_COLORS.get(condition, '#000000')
                date_key = get_date_key(test_s['pkl_path'])
                cond_label = CONDITION_LABELS.get(condition, condition)
                pre_med = session_median(signals_p[sig_key]) if sig_key in signals_p else np.nan
                test_med = session_median(signals_t[sig_key]) if sig_key in signals_t else np.nan
                if not np.isnan(pre_med) and pre_med > 0 and not np.isnan(test_med):
                    ratio = test_med / pre_med
                    ax.bar(xi, ratio, color=color, alpha=0.7, width=0.6)
                    ax.text(xi, ratio + 0.01, f'{ratio:.2f}', ha='center', fontsize=6)
            # Stats — compare ratio=1 (no effect) vs actual ratios using Wilcoxon signed-rank
            from scipy import stats as _stats
            ratios_all = []
            for pre_s, test_s, condition in mouse_pairs:
                fps_p2, frames_p2, signals_p2 = extract_signals(pre_s['data'])
                fps_t2, frames_t2, signals_t2 = extract_signals(test_s['data'])
                pre_m = session_median(signals_p2[sig_key]) if sig_key in signals_p2 else np.nan
                test_m = session_median(signals_t2[sig_key]) if sig_key in signals_t2 else np.nan
                if not np.isnan(pre_m) and pre_m > 0 and not np.isnan(test_m):
                    ratios_all.append(test_m / pre_m)
            if len(ratios_all) >= 2:
                _, pval = _stats.wilcoxon(ratios_all, [1.0]*len(ratios_all))
                stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
                y_max = ax.get_ylim()[1]
                ax.text(n_sessions/2 - 0.5, y_max * 0.95, f'vs 1.0: {stars}',
                       ha='center', fontsize=7, color='black')
            ax.axhline(y=1.0, color='black', linewidth=0.8, linestyle='--')
            ax.set_xticks(range(n_sessions))
            xlabels = [f'{get_date_key(t["pkl_path"])}\n{CONDITION_LABELS.get(c, c)}' for _, t, c in mouse_pairs]
            ax.set_xticklabels(xlabels, fontsize=5, rotation=30)
            ax.set_ylabel('Ratio vs Pre-injection', fontsize=7)
            ax.set_title(SIGNAL_LABELS[sig_key], fontsize=7)
            ax.grid(False)

        fig.suptitle(f'{mouse} — Ratio Test / Pre-injection', fontsize=11, y=1.01)
        plt.tight_layout()
        out = os.path.join(out_dir, f'{mouse}_pre_vs_test_ratio.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out}')


# ── Saline vs Ivabradine comparison ──────────────────────────────────────────
# START_MIN_STAT: minutes after injection before statistics are computed
# First 10 minutes excluded to avoid the transient stress response from the injection itself
# (sympathetic activation → pupil dilation regardless of drug effect)
# See literature: saline injection stress lasts ~5-10 min
START_MIN_STAT = 10  # minutes before this = greyed out, not counted in stats

def get_common_end(sessions_list):
    """Return the minimum session end time (minutes) across all provided session lists.
    Used to align x-axis across conditions — all curves end at the shortest session.
    Returns None if all session lists are empty."""
    min_end = np.inf
    for sessions in sessions_list:
        for s in sessions:
            frames = np.array(s['data']['frames'])
            if len(frames) > 0:
                end_min = frames[-1] / 60
                if end_min < min_end:
                    min_end = end_min
    return min_end if min_end != np.inf else None

def plot_saline_vs_ivabradine(all_data, output_dir):
    """Plot Saline vs Ivabradine individual session curves per mouse — 4 column layout.
   
    For each mouse, generate a figure with 4 subplots columns:
        1. All conditions (Saline + all Ivabradine doses)
        2. Saline vs Iva 5mg/kg
        3. Saline vs Iva 10mg/kg
        4. Saline vs Iva 20mg/kg
        First 15 min shown in grey (not counted in stats).
        X axis truncated to common minimum end time.
   
   Grey zone (before START_MIN_STAT=10min): injection stress period — reduced alpha, not in stats.
   Active zone (after 10min): full alpha — used for statistical comparisons.
   Vertical dashed line at START_MIN_STAT separates the two zones.
   
   X-axis clipped to common minimum end time (get_common_end) — all curves end at same point.
   Empty conditions (no sessions for this mouse) silently skipped.
   One PNG per mouse."""

    out_dir = output_dir

    for mouse in MICE:
        saline_sessions = [s for s in all_data['Injection_Saline'] if s['mouse'] == mouse]
        iva5_sessions   = [s for s in all_data['Injection_Ivabradine_5mgkg'] if s['mouse'] == mouse]
        iva10_sessions  = [s for s in all_data['Injection_Ivabradine_10mgkg'] if s['mouse'] == mouse]
        iva20_sessions  = [s for s in all_data['Injection_Ivabradine_20mgkg'] if s['mouse'] == mouse]

        if not saline_sessions and not (iva5_sessions or iva10_sessions or iva20_sessions):
            continue

        comparisons = [
            ('All conditions', {
                'Injection_Saline': saline_sessions,
                'Injection_Ivabradine_5mgkg': iva5_sessions,
                'Injection_Ivabradine_10mgkg': iva10_sessions,
                'Injection_Ivabradine_20mgkg': iva20_sessions,
            }),
            ('Saline vs Iva 5mg/kg', {
                'Injection_Saline': saline_sessions,
                'Injection_Ivabradine_5mgkg': iva5_sessions,
            }),
            ('Saline vs Iva 10mg/kg', {
                'Injection_Saline': saline_sessions,
                'Injection_Ivabradine_10mgkg': iva10_sessions,
            }),
            ('Saline vs Iva 20mg/kg', {
                'Injection_Saline': saline_sessions,
                'Injection_Ivabradine_20mgkg': iva20_sessions,
            }),
        ]

        n_signals = len(SIGNAL_KEYS)
        fig, axs = plt.subplots(n_signals, 4, figsize=(24, 3 * n_signals))
        fig.subplots_adjust(top=0.93, hspace=0.4, wspace=0.3)
        if n_signals == 1:
            axs = axs.reshape(1, 4)

        for col, (comp_title, sessions_by_condition) in enumerate(comparisons):
            sessions_by_condition = {k: v for k, v in sessions_by_condition.items() if v}
            common_end = get_common_end(list(sessions_by_condition.values()))
            axs[0, col].set_title(comp_title, fontsize=8)

            for row, sig_key in enumerate(SIGNAL_KEYS):
                ax = axs[row, col]

                for condition, sessions in sessions_by_condition.items():
                    color = CONDITION_COLORS.get(condition, '#000000')
                    label = CONDITION_LABELS.get(condition, condition)
                    first = True
                    for session in sessions:
                        fps, frames, signals = extract_signals(session['data'])
                        if sig_key not in signals or len(frames) == 0:
                            continue
                        t_min, binned = bin_signal(signals[sig_key], frames)

                        # Clip to common end
                        if common_end is not None:
                            mask = t_min <= common_end
                            t_min = t_min[mask]
                            binned = binned[mask]

                        grey_mask = t_min < START_MIN_STAT
                        active_mask = t_min >= START_MIN_STAT

                        # Grey zone (before 10 min) — same color, reduced alpha
                        if np.any(grey_mask):
                            ax.plot(t_min[grey_mask], binned[grey_mask],
                                   color=color, linewidth=0.8, alpha=0.3)

                        # Active zone (after 15 min)
                        if np.any(active_mask):
                            ax.plot(t_min[active_mask], binned[active_mask],
                                   color=color, linewidth=0.8, alpha=0.7,
                                   label=label if first else None)
                            first = False

                # Vertical line at 15 min
                ax.axvline(x=START_MIN_STAT, color='grey', linewidth=0.8, linestyle='--', alpha=0.5)

                if row == n_signals - 1:
                    ax.set_xlabel('Time (min)', fontsize=7)
                if col == 0:
                    ax.set_ylabel(SIGNAL_LABELS[sig_key], fontsize=7)
                ax.tick_params(labelsize=6)
                ax.grid(False)

            handles, labels_leg = axs[0, col].get_legend_handles_labels()
            unique = dict(zip(labels_leg, handles))
            if unique:
                axs[0, col].legend(unique.values(), unique.keys(), fontsize=6,
                                  bbox_to_anchor=(0, 1.2), loc='upper left')

        fig.suptitle(f'{mouse} — Saline vs Ivabradine (grey = first {START_MIN_STAT} min, not counted in stats)',
                    fontsize=11, y=0.98)
        out = os.path.join(out_dir, f'{mouse}_saline_vs_ivabradine.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out}')

        # ── Figure trajectoire centre de masse ───────────────────────────
        # Plots raw frame-by-frame pupil center position (not binned) for all sessions
        # x = center[:,1], y = center[:,0] — FaceMap returns [y, x] convention
        # y-axis inverted to match image coordinate system (origin top-left)
        # linewidth=0.3, alpha=0.5 — very thin lines to show density without overplotting
        # All conditions overlaid — useful for detecting systematic pupil position shifts
        # between Saline and Ivabradine that would bias area measurements
        fig, ax = plt.subplots(figsize=(8, 7))
        all_conditions_sessions = {
            'Injection_Saline': saline_sessions,
            'Injection_Ivabradine_5mgkg': iva5_sessions,
            'Injection_Ivabradine_10mgkg': iva10_sessions,
            'Injection_Ivabradine_20mgkg': iva20_sessions,
        }
        for condition, sessions in all_conditions_sessions.items():
            if not sessions:
                continue
            color = CONDITION_COLORS.get(condition, '#000000')
            label = CONDITION_LABELS.get(condition, condition)
            for session in sessions:
                center = session['data']['pupil_center_[x,y]']
                x = center[:, 1].astype(float)
                y = center[:, 0].astype(float)
                valid = ~(np.isnan(x) | np.isnan(y))
                if np.sum(valid) > 0:
                    ax.plot(x[valid], y[valid], color=color, linewidth=0.3,
                           alpha=0.5, label=label)

        ax.set_xlabel('x (px)', fontsize=9)
        ax.set_ylabel('y (px)', fontsize=9)
        ax.invert_yaxis()
        ax.grid(False)
        # Deduplicate legend
        handles, labels_leg = ax.get_legend_handles_labels()
        unique = dict(zip(labels_leg, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=8)
        ax.set_title(f'{mouse} — Pupil center trajectory (Saline vs Ivabradine)', fontsize=10)
        out = os.path.join(out_dir, f'{mouse}_saline_vs_ivabradine_trajectory.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out}')


# ── Without vs With corneal mask comparison ──────────────────────────────────
# Compares signals between videos processed without and with the FaceMap corneal reflector mask.
# Purpose: assess whether the corneal mask improves or biases pupil detection.

# MASK_PAIRS: list of tuples (mouse, session_name, nomask_relative_path, mask_relative_path)
# Each pair = same video processed twice in FaceMap (with and without the corneal reflector mask)
# Paths are relative to NAS_BASE — both pkl files must exist for the pair to be valid.

# HOW TO ADD A NEW PAIR:
#   ('KXXXX', 'FDDMMYYYY_Condition',
#    'KXXXX/path/to/VF1/without_mask_FacemapPose.pkl',
#    'KXXXX_Corneal_mask_files/date condition/with_mask/with_mask_FacemapPose.pkl')

NAS_BASE = '/media/nas8-2/ProjectCardioSense'

MASK_PAIRS = [
    ('K1711', 'F04032025_Pre-injection',
     'K1711/2025-03-04_11-59-07/1711_250304_Basal_Pre-Injection/SLEEP-Mouse-1711-04032025-Sleep_00/Analyse pupille/VF1/F04032025-0000_Pre-injection_FacemapPose.pkl',
     'K1711_Corneal_mask_files_and_data/04.03 P/with_mask/F04032025-0000_with_mask_Pre-injection_FacemapPose.pkl'),
    ('K1711', 'F04032025_Saline',
     'K1711/2025-03-04_11-59-07/1711_250304_Injection_Saline/SLEEP-Mouse-1711-04032025-Sleep_01/Analyse pupille/VF1/F04032025-0000_Saline_FacemapPose.pkl',
     'K1711_Corneal_mask_files_and_data/04.03 S/with_mask/F04032025-0000_with_mask_Saline_FacemapPose.pkl'),
    ('K1711', 'F05032025_Pre-injection',
     'K1711/2025-03-05_14-29-26/1711_250305_Basal_Pre-Injection/SLEEP-Mouse-1711-05032025-Sleep_00/Analyse pupille/VF1/F05032025-0000_Pre-injection_FacemapPose.pkl',
     'K1711_Corneal_mask_files_and_data/05.03 P/with_mask/F05032025-0000_with_mask_Pre-injection_FacemapPose.pkl'),
    ('K1711', 'F05032025_Ivabradine_20mg',
     'K1711/2025-03-05_14-29-26/1711_250305_Injection_Ivabradine_20mgkg/SLEEP-Mouse-1711-05032025-Sleep_01/Analyse pupille/VF1/F05032025-0000_Ivabradine_FacemapPose.pkl',
     'K1711_Corneal_mask_files_and_data/05.03 Iva 20mg/with_mask/F05032025-0000_with_mask_Ivabradine_FacemapPose.pkl'),
    ('K1712', 'F04032025_Pre-injection',
     'K1712/Injection_Pre_Saline_Ivabradine/2025-03-04_17-18-54/1712_250304_Basal_Pre-Injection/SLEEP-Mouse-1712-04032025-Sleep_00/Analyse pupille/VF1/F04032025-0000_Pre-injection_FacemapPose.pkl',
     'K1712_Corneal_mask_files/04.03 P/with_mask/F04032025-0000_with_mask_Pre-injection_FacemapPose.pkl'),
    ('K1712', 'F04032025_Saline',
     'K1712/Injection_Pre_Saline_Ivabradine/2025-03-04_17-18-54/1712_250304_Injection_Saline/SLEEP-Mouse-1712-04032025-Sleep_01/Analyse pupille/VF1/F04032025-0000_Saline_FacemapPose.pkl',
     'K1712_Corneal_mask_files/04.03 S/with_mask/F04032025-0000_with_mask_Saline_FacemapPose.pkl'),
    ('K1712', 'F05032025_Pre-injection',
     'K1712/Injection_Pre_Saline_Ivabradine/2025-03-05_16-32-38/1712_250305_Basal_Pre-Injection/SLEEP-Mouse-1712-05032025-Sleep_00/Analyse pupille/VF1/F05032025-0000_Pre-injection_FacemapPose.pkl',
     'K1712_Corneal_mask_files/05.03 P/with_mask/F05032025-0000_with_mask_Pre-injection_FacemapPose.pkl'),
    ('K1712', 'F05032025_Ivabradine_20mg',
     'K1712/Injection_Pre_Saline_Ivabradine/2025-03-05_16-32-38/1712_250305_Injection_Ivabradine_20mgkg/SLEEP-Mouse-1712-05032025-Sleep_01/Analyse pupille/VF1/F05032025-0000_Ivabradine_FacemapPose.pkl',
     'K1712_Corneal_mask_files/05.03 Iva 20mg/with_mask/F05032025-0000_with_mask_Ivabradine_FacemapPose.pkl'),
    ('K1712', 'F11032025_Pre-injection',
     'K1712/Injection_Pre_Saline_Ivabradine/2025-03-11_14-19-28/1712_250311_Basal_Pre-Injection/SLEEP-Mouse-1712-11032025-Sleep_00/Analyse pupille/VF1/F11032025-0000-141928_Pre-injection_FacemapPose.pkl',
     'K1712_Corneal_mask_files/11.03 p/with_mask/F11032025-0000-141928_with_mask_Pre-injection_FacemapPose.pkl'),
    ('K1712', 'F11032025_Ivabradine_5mg',
     'K1712/Injection_Pre_Saline_Ivabradine/2025-03-11_14-19-28/1712_250311_Injection_Ivabradine_5mgkg/SLEEP-Mouse-1712-11032025-Sleep_01/Analyse pupille/VF1/F11032025-0000-141928_Ivabradine_FacemapPose.pkl',
     'K1712_Corneal_mask_files/11.03 Iva 5mg/with_mask/F11032025-0000-141928_with_mask_Ivabradine_FacemapPose.pkl'),
    ('K1712', 'F12032025_Pre-injection',
     'K1712/Injection_Pre_Saline_Ivabradine/2025-03-12_15-07-02/1712_250312_Basal_Pre-Injection/SLEEP-Mouse-1712-12032025-Sleep_00/Analyse pupille/VF1/F12032025-0000_Pre-injection_FacemapPose.pkl',
     'K1712_Corneal_mask_files/12.03 p/with_mask/F12032025-0000_with_mask_Pre-injection_FacemapPose.pkl'),
    ('K1712', 'F12032025_Ivabradine_10mg',
     'K1712/Injection_Pre_Saline_Ivabradine/2025-03-12_15-07-02/1712_250312_Injection_Ivabradine_10mgkg/SLEEP-Mouse-1712-12032025-Sleep_01/Analyse pupille/VF1/F12032025-0000_Ivabradine_FacemapPose.pkl',
     'K1712_Corneal_mask_files/12.03 Iva 10mg/with_mask/F12032025-0000_with_mask_Ivabradine_FacemapPose.pkl'),
    ('K1712', 'F13032025_Pre-injection',
     'K1712/Injection_Pre_Saline_Ivabradine/2025-03-13_15-28-37/1712_250313_Basal_Pre-Injection/SLEEP-Mouse-1712-13032025-Sleep_00/Analyse pupille/VF1/F13032025-0000_Pre-injection_FacemapPose.pkl',
     'K1712_Corneal_mask_files/13.03 P/with_mask/F13032025-0000_with_mask_Pre-injection_FacemapPose.pkl'),
    ('K1712', 'F13032025_Saline',
     'K1712/Injection_Pre_Saline_Ivabradine/2025-03-13_15-28-37/1712_250313_Injection_Saline/SLEEP-Mouse-1712-13032025-Sleep_01/Analyse pupille/VF1/F13032025-0000_Saline_FacemapPose.pkl',
     'K1712_Corneal_mask_files/13.03 S/with_mask/F13032025-0000_with_mask_Saline_FacemapPose.pkl'),
    ('K1712', 'F31032025_Pre-injection',
     'K1712/Injection_Pre_Saline_Ivabradine/2025-03-31_13-56-30/1712_250331_Basal_Pre-Injection/SLEEP-Mouse-1712-31032025-Sleep_00/Analyse pupille/VF1/F31032025-0000-135630_Pre-injection_FacemapPose.pkl',
     'K1712_Corneal_mask_files/31.03 P/with_mask/F31032025-0000-135630_with_mask_Pre-injection_FacemapPose.pkl'),
    ('K1712', 'F31032025_Saline',
     'K1712/Injection_Pre_Saline_Ivabradine/2025-03-31_13-56-30/1712_250331_Injection_Saline/SLEEP-Mouse-1712-31032025-Sleep_01/Analyse pupille/VF1/F31032025-0000-135630_Saline_FacemapPose.pkl',
     'K1712_Corneal_mask_files/31.03 S/with_mask/F31032025-0000-135630_with_mask_Saline_FacemapPose.pkl'),
    ('K1712', 'F01042025_Pre-injection',
     'K1712/Injection_Pre_Saline_Ivabradine/2025-04-01_14-05-15/1712_250401_Basal_Pre-Injection/SLEEP-Mouse-1712-01042025-Sleep_01/Analyse pupille/VF1/F01042025-0000_Pre-injection_FacemapPose.pkl',
     'K1712_Corneal_mask_files/01.04 P/with_mask/F01042025-0000_with_mask_Pre-injection_FacemapPose.pkl'),
    ('K1712', 'F01042025_Ivabradine_5mg',
     'K1712/Injection_Pre_Saline_Ivabradine/2025-04-01_14-05-15/1712_250401_Injection_Ivabradine_5mgkg/SLEEP-Mouse-1712-01042025-Sleep_02/Analyse pupille/VF1/F01042025-0000_Ivabradine_FacemapPose.pkl',
     'K1712_Corneal_mask_files/01.04 Iva 5mg/with_mask/F01042025-0000_with_mask_Ivabradine_FacemapPose.pkl'),
    ('K1712', 'F02042025_Pre-injection',
     'K1712/Injection_Pre_Saline_Ivabradine/2025-04-02_14-10-36/1712_250402_Basal_Pre-Injection/SLEEP-Mouse-1712-02042025-Sleep_00/Analyse pupille/VF1/F02042025-0000_Pre-injection_FacemapPose.pkl',
     'K1712_Corneal_mask_files/02.04 P/with_mask/F02042025-0000_with_mask_Pre-injection_FacemapPose.pkl'),
    ('K1712', 'F02042025_Ivabradine_10mg',
     'K1712/Injection_Pre_Saline_Ivabradine/2025-04-02_14-10-36/1712_250402_Injection_Ivabradine_10mgkg/SLEEP-Mouse-1712-02042025-Sleep_01/Analyse pupille/VF1/F02042025-0000_Ivabradine_FacemapPose.pkl',
     'K1712_Corneal_mask_files/02.04 Iva 10mg/with_mask/F02042025-0000_with_mask_Ivabradine_FacemapPose.pkl'),
]

def plot_mask_comparison(base_dir, output_dir):
    """Placeholder — mask comparison figures generated by MaskVSnomask.py.
    Run MaskVSnomask.py separately to avoid RAM issues with full session loading."""

    print('   Please launch: python MaskVSnomask.py')


# ── Statistical tests ─────────────────────────────────────────────────────────
# All tests use pre-aggregated values (one median per mouse) — mouse is the statistical unit
# Shapiro-Wilk normality test run but not used for test selection (always non-parametric)
# FDR correction applied downstream via apply_fdr_correction()

from scipy import stats
from itertools import combinations

def compute_stats_from_values(group_values, paired=False):

    """Compute pairwise non-parametric tests between all condition pairs.
    
    group_values: dict {condition_label: [one_value_per_mouse]}
    Avoids reloading pkl data — use when medians are already computed.
    Skips pairs with fewer than 2 values in either group.
    
    Test selection:
      - Wilcoxon signed-rank: paired=True AND equal n (same mice in both conditions)
      - Mann-Whitney: otherwise (unequal n or unpaired)
    
    Shapiro-Wilk run for reference only — not used for test selection.
    Returns list of result dicts with group names, n, statistic, p_value, test_used."""
    
    from itertools import combinations as _combinations
    groups = list(group_values.keys())
    results = []
    for g1, g2 in _combinations(groups, 2):
        vals1 = np.array([v for v in group_values[g1] if not np.isnan(v)])
        vals2 = np.array([v for v in group_values[g2] if not np.isnan(v)])
        if len(vals1) < 2 or len(vals2) < 2:
            continue
        _, p_norm1 = stats.shapiro(vals1) if len(vals1) >= 3 else (None, 0.0)
        _, p_norm2 = stats.shapiro(vals2) if len(vals2) >= 3 else (None, 0.0)
        normal1 = p_norm1 > 0.05 if p_norm1 is not None else None
        normal2 = p_norm2 > 0.05 if p_norm2 is not None else None
        # Test selection:
        # Wilcoxon signed-rank — used when groups are paired AND have equal n
        #   (same mouse in both conditions, same number of sessions)
        # Mann-Whitney — used when n differs between groups
        #   (unequal number of sessions per condition, cannot pair)
        # NOTE: when all conditions have equal n per mouse (future dataset),
        #   consider truncating to min n and forcing Wilcoxon for all comparisons
        if paired and len(vals1) == len(vals2):
            stat, pval = stats.wilcoxon(vals1, vals2)
            test_used = 'Wilcoxon signed-rank (paired, equal n)'
        else:
            stat, pval = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
            test_used = 'Mann-Whitney (independent, unequal n)'
        results.append({
            'group1': g1, 'group2': g2,
            'n1': len(vals1), 'n2': len(vals2),
            'statistic': stat, 'p_value': pval,
            'test_used': test_used,
            'normal1': normal1, 'normal2': normal2,
        })
    return results

def compute_stats(sessions_by_group, sig_key, paired=False):
    """Compute pairwise non-parametric tests between groups — loads signals from sessions.
    
    Wrapper around compute_stats_from_values() that extracts session medians first.
    Use compute_stats_from_values() instead when medians are already computed (faster).
    
    Parameters:
      sessions_by_group: dict {group_label: [session_dicts]}
      sig_key: signal to test (must be in SIGNAL_KEYS)
      paired: True = Wilcoxon signed-rank (if equal n), False = Mann-Whitney
    
    Skips pairs with fewer than 2 values.
   Returns list of dicts with keys: group1, group2, n1, n2, statistic, p_value, test_used, normal1, normal2"""

    # Collect medians per group
    group_medians = {}
    for group, sessions in sessions_by_group.items():
        medians = []
        for s in sessions:
            fps, frames, signals = extract_signals(s['data'])
            if sig_key in signals:
                m = session_median(signals[sig_key])
                if not np.isnan(m):
                    medians.append(m)
        group_medians[group] = np.array(medians)

    groups = list(group_medians.keys())
    results = []

    for g1, g2 in combinations(groups, 2):
        vals1 = group_medians[g1]
        vals2 = group_medians[g2]

        if len(vals1) < 2 or len(vals2) < 2:
            continue

        # Shapiro-Wilk normality test — informative only, not used to select test
        _, p_norm1 = stats.shapiro(vals1) if len(vals1) >= 3 else (None, 0.0)
        _, p_norm2 = stats.shapiro(vals2) if len(vals2) >= 3 else (None, 0.0)
        normal1 = p_norm1 > 0.05 if p_norm1 is not None else None
        normal2 = p_norm2 > 0.05 if p_norm2 is not None else None

        # Always use non-parametric tests regardless of normality
        # (small n, non-independent observations — see methodological note)
        # Paired (same mouse, different conditions) → Wilcoxon signed-rank
        # Independent (different mice, same condition) → Mann-Whitney
        # Test selection:
        # Wilcoxon signed-rank — used when groups are paired AND have equal n
        #   (same mouse in both conditions, same number of sessions)
        # Mann-Whitney — used when n differs between groups
        #   (unequal number of sessions per condition, cannot pair)
        # NOTE: when all conditions have equal n per mouse (future dataset),
        #   consider truncating to min n and forcing Wilcoxon for all comparisons
        if paired and len(vals1) == len(vals2):
            stat, pval = stats.wilcoxon(vals1, vals2)
            test_used = 'Wilcoxon signed-rank (paired, equal n)'
        else:
            stat, pval = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
            test_used = 'Mann-Whitney (independent, unequal n)'

        results.append({
            'group1': g1, 'group2': g2,
            'n1': len(vals1), 'n2': len(vals2),
            'statistic': stat, 'p_value': pval,
            'test_used': test_used,
            'normal1': normal1, 'normal2': normal2,
        })

    return results

def apply_fdr_correction(results_list):
    """Apply Benjamini-Hochberg False Discovery Rate correction to a list of test results.
    
    Corrects for multiple comparisons across all pairwise tests — reduces false positives when testing multiple signals and condition pairs simultaneously.
    
    Adds 'p_corrected' and 'significant' keys to each result dict in-place.
    Returns the same list with corrections added."""

    from scipy.stats import false_discovery_control
    if not results_list:
        return results_list
    p_values = np.array([r['p_value'] for r in results_list])
    # BH correction
    reject, p_corrected = fdr_bh(p_values)
    for i, r in enumerate(results_list):
        r['p_corrected'] = p_corrected[i]
        r['significant'] = reject[i]
    return results_list

def fdr_bh(p_values):
    """Benjamini-Hochberg FDR correction.
   
   Algorithm:
     1. Sort p-values in ascending order: p(1) <= p(2) <= ... <= p(n)
     2. Compare each p(i) to its BH threshold: (i/n) * alpha (alpha=0.05)
     3. Find the largest i where p(i) <= (i/n)*alpha — reject all H0 up to that rank
     4. Monotone enforcement: if rank i is rejected, all ranks below i are also rejected
     5. Corrected p-values: p_corr(i) = min(p(i) * n/i, 1.0), computed right-to-left
        to ensure monotonicity of corrected values
   
   Returns (reject: bool array, p_corrected: float array) in original order."""
    
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    alpha = 0.05
    threshold = (np.arange(1, n+1) / n) * alpha
    reject_sorted = sorted_p <= threshold
    # Make rejection monotone
    for i in range(n-2, -1, -1):
        if reject_sorted[i+1]:
            reject_sorted[i] = True
    reject = np.empty(n, dtype=bool)
    reject[sorted_idx] = reject_sorted
    # Corrected p-values
    p_corrected = np.minimum(1.0, sorted_p * n / np.arange(1, n+1))
    p_corrected = np.minimum.accumulate(p_corrected[::-1])[::-1]
    p_corr_out = np.empty(n)
    p_corr_out[sorted_idx] = p_corrected
    return reject, p_corr_out

def pval_to_stars(p):
    """
    Convert p-value to significance stars.
    - p < 0.001 → '***' 
    - p < 0.01 → '**' 
    - p < 0.05 → '*'
    - p >= 0.05 → 'ns'.
    """
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'

def add_ratio_stats_to_legend(ax, condition_label, pval):
    # Add ratio vs 1.0 stats as a legend entry — avoids text overlap on plot
    stars = pval_to_stars(pval)
    color = 'black' if pval < 0.05 else 'grey'
    return plt.Line2D([0], [0], linestyle='none', marker='',
                      label=f'{condition_label}: vs 1.0 {stars} (p={pval:.3f})',
                      color=color)

def add_stats_to_ax(ax, x_positions, results, sig_key, label_map, y_offset_factor=0.05):
    """Draw significance bars and stars above boxplots on a matplotlib axis.
    
    Draws a bar between each tested pair with stars above:
      - Significant (p<0.05): black solid bar
      - Non-significant: grey dashed bar
      
    Bars stacked vertically with step = y_range * y_offset_factor to avoid overlap.
    Uses FDR-corrected p-value if available, raw p-value otherwise.
    Footnote added below axis: test name + FDR correction + star legend.
    y_offset_factor: increase if bars overlap, decrease if figure is too tall."""

    y_max = ax.get_ylim()[1]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    step = y_range * y_offset_factor
    current_y = y_max + step
    tests_used = set()

    for r in results:
        tests_used.add(r['test_used'])
        g1, g2 = r['group1'], r['group2']
        labels = list(label_map.keys())
        if g1 not in labels or g2 not in labels:
            continue
        x1 = x_positions[labels.index(g1)]
        x2 = x_positions[labels.index(g2)]
        pval = r.get('p_corrected', r['p_value'])
        stars = pval_to_stars(pval)
        # Draw bar for all comparisons (significant or not)
        line_color = 'black' if pval < 0.05 else 'grey'
        line_style = '-' if pval < 0.05 else '--'
        ax.plot([x1, x1, x2, x2], [current_y, current_y + step*0.3,
                current_y + step*0.3, current_y],
                color=line_color, linewidth=0.8, linestyle=line_style)
        ax.text((x1+x2)/2, current_y + step*0.3, stars,
               ha='center', va='bottom', fontsize=8,
               color=line_color)
        current_y += step * 1.2
    ax.set_ylim(top=current_y + step)

    # Add test name as footnote
    if tests_used:
        test_str = ', '.join(sorted(tests_used))
        ax.annotate(f'Test: {test_str} | FDR corrected | *p<0.05 **p<0.01 ***p<0.001',
                   xy=(0.5, -0.25), xycoords='axes fraction',
                   ha='center', fontsize=6, color='grey', style='italic')

def save_stats_table(all_results, output_path, sig_key):
    """Save pairwise statistical test results to CSV alongside the figure.
   
   One row per comparison pair. Columns: signal, group1, group2, n1, n2,
   test_used, statistic, p_value, p_corrected (FDR), significant, normal1, normal2.
   Saved as *_<sig_key>_stats.csv in the same directory as the figure PNG."""
    
    import csv
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'signal', 'group1', 'group2', 'n1', 'n2',
            'test_used', 'statistic', 'p_value', 'p_corrected',
            'significant', 'normal1', 'normal2'
        ])
        writer.writeheader()
        for r in all_results:
            row = dict(r)
            row['signal'] = sig_key
            writer.writerow(row)


# ── Figure — All mice ratio (condition/Pre-injection) ────────────────────────
def plot_ratio_all_mice(all_data, output_dir): 
    """Plot ratio condition/Pre-injection box plots — all mice combined.
    
    For each session: ratio = session_median(condition) / session_median(pre-injection).
    Pre-injection matched by (mouse, date key F<YYYYMMDD>, parent timestamp folder).
    
    One box plot per condition — all mice pooled.
    Statistical test: Wilcoxon signed-rank vs 1.0 (H0: no effect vs pre-injection).
    
    get_date_key(): extracts F<YYYYMMDD> from pkl filename for day-level matching.
    get_parent_folder(): extracts YYYY-MM-DD_HH-MM-SS folder for exact session matching."""


    # Index Pre-injection sessions by (mouse, date, parent folder) for exact day matching
    # pre_index: {(mouse, F<YYYYMMDD>, parent_folder): session_dict}

    # Compute ratio = test_median / pre_median for each paired session
    # Skips sessions where no matching pre-injection found (date/mouse mismatch)
    # Skips if pre_median = 0 or NaN — avoids division by zero
    # ratios_by_condition: {condition: {sig_key: [ratio_per_session]}}
    # mouse_by_condition: {condition: [mouse_per_session]} — for point coloring

    # Reference line at ratio=1.0 (dashed grey) = no effect vs pre-injection
    # Individual points colored by MOUSE_COLORS — each point = one session
    # Diamond marker = median ± IQR of ratios across all sessions of the condition
    pre_index = {}
    for s in all_data['Basal_Pre_Injection']:
        date_key = get_date_key(s['pkl_path'])
        parent = get_parent_folder(s['pkl_path'])
        if date_key:
            pre_index[(s['mouse'], date_key, parent)] = s
    # Compute ratios per condition
    test_conditions = [c for c in EXPERIMENTS if c != 'Basal_Pre_Injection']
    ratios_by_condition = {c: {sig_key: [] for sig_key in SIGNAL_KEYS} for c in test_conditions}
    mouse_by_condition = {c: [] for c in test_conditions}
    for condition in test_conditions:
        for s in all_data[condition]:
            date_key = get_date_key(s['pkl_path'])
            parent = get_parent_folder(s['pkl_path'])
            key = (s['mouse'], date_key, parent)
            if key not in pre_index:
                continue
            pre_s = pre_index[key]
            fps_t, frames_t, signals_t = extract_signals(s['data'])
            fps_p, frames_p, signals_p = extract_signals(pre_s['data'])
            for sig_key in SIGNAL_KEYS:
                if sig_key in signals_t and sig_key in signals_p:
                    pre_med = session_median(signals_p[sig_key])
                    test_med = session_median(signals_t[sig_key])
                    if not np.isnan(pre_med) and pre_med > 0 and not np.isnan(test_med):
                        ratios_by_condition[condition][sig_key].append(test_med / pre_med)
            mouse_by_condition[condition].append(s['mouse'])
    n_signals = len(SIGNAL_KEYS)
    n_conditions = len(test_conditions)
    fig, axs = plt.subplots(n_signals, 1, figsize=(14, 3 * n_signals))
    fig.subplots_adjust(top=0.94, hspace=0.5)
    if n_signals == 1:
        axs = [axs]
    x_pos = np.arange(n_conditions)
    for ax, sig_key in zip(axs, SIGNAL_KEYS):
        for xi, condition in enumerate(test_conditions):
            ratios = ratios_by_condition[condition][sig_key]
            mice = mouse_by_condition[condition]
            color = CONDITION_COLORS.get(condition, '#000000')
            if len(ratios) == 0:
                continue
            ratios = np.array(ratios)
            if len(ratios) > 1:
                bp = plot_box_or_bar(ax, list(ratios), xi, color, width=0.4, alpha=0.6)
            # Individual points colored by mouse
            for j, (r, mouse) in enumerate(zip(ratios, mice)):
                jitter = np.random.uniform(-0.08, 0.08)
                pt_color = MOUSE_COLORS.get(mouse, '#000000')
                ax.scatter(xi + jitter, r, color=pt_color, s=25, zorder=5, alpha=0.9)
            # Median ± IQR marker
            med = np.median(ratios)
            q25, q75 = np.percentile(ratios, [25, 75]) if len(ratios) > 1 else (med, med)
            ax.errorbar(xi + 0.3, med, yerr=[[med - q25], [q75 - med]],
                       fmt='D', color='black', markersize=6, capsize=5,
                       linewidth=2, zorder=6)
        # Reference line at ratio=1
        ax.axhline(y=1.0, color='grey', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in test_conditions],
                           fontsize=8, rotation=15)
        ax.set_ylabel(f'Ratio {SIGNAL_LABELS[sig_key]}\n(condition/Pre-injection)', fontsize=7)
        ax.grid(False)



    # Index Pre-injection sessions
    pre_index = {}
    for s in all_data['Basal_Pre_Injection']:
        date_key = get_date_key(s['pkl_path'])
        parent = get_parent_folder(s['pkl_path'])
        if date_key:
            pre_index[(s['mouse'], date_key, parent)] = s

    # Compute ratios per condition
    test_conditions = [c for c in EXPERIMENTS if c != 'Basal_Pre_Injection']
    ratios_by_condition = {c: {sig_key: [] for sig_key in SIGNAL_KEYS} for c in test_conditions}
    mouse_by_condition = {c: [] for c in test_conditions}

    for condition in test_conditions:
        for s in all_data[condition]:
            date_key = get_date_key(s['pkl_path'])
            parent = get_parent_folder(s['pkl_path'])
            key = (s['mouse'], date_key, parent)
            if key not in pre_index:
                continue
            pre_s = pre_index[key]
            fps_t, frames_t, signals_t = extract_signals(s['data'])
            fps_p, frames_p, signals_p = extract_signals(pre_s['data'])
            for sig_key in SIGNAL_KEYS:
                if sig_key in signals_t and sig_key in signals_p:
                    pre_med = session_median(signals_p[sig_key])
                    test_med = session_median(signals_t[sig_key])
                    if not np.isnan(pre_med) and pre_med > 0 and not np.isnan(test_med):
                        ratios_by_condition[condition][sig_key].append(test_med / pre_med)
            mouse_by_condition[condition].append(s['mouse'])

    n_signals = len(SIGNAL_KEYS)
    n_conditions = len(test_conditions)
    fig, axs = plt.subplots(n_signals, 1, figsize=(14, 3 * n_signals))
    fig.subplots_adjust(top=0.94, hspace=0.5)
    if n_signals == 1:
        axs = [axs]

    x_pos = np.arange(n_conditions)

    for ax, sig_key in zip(axs, SIGNAL_KEYS):
        for xi, condition in enumerate(test_conditions):
            ratios = ratios_by_condition[condition][sig_key]
            mice = mouse_by_condition[condition]
            color = CONDITION_COLORS.get(condition, '#000000')
            if len(ratios) == 0:
                continue
            ratios = np.array(ratios)
            if len(ratios) > 1:
                bp = plot_box_or_bar(ax, list(ratios), xi, color, width=0.4, alpha=0.6)
            # Individual points colored by mouse
            for j, (r, mouse) in enumerate(zip(ratios, mice)):
                jitter = np.random.uniform(-0.08, 0.08)
                pt_color = MOUSE_COLORS.get(mouse, '#000000')
                ax.scatter(xi + jitter, r, color=pt_color, s=25, zorder=5, alpha=0.9)
            # Median ± IQR marker
            med = np.median(ratios)
            q25, q75 = np.percentile(ratios, [25, 75]) if len(ratios) > 1 else (med, med)
            ax.errorbar(xi + 0.3, med, yerr=[[med - q25], [q75 - med]],
                       fmt='D', color='black', markersize=6, capsize=5,
                       linewidth=2, zorder=6)

        # Reference line at ratio=1
        ax.axhline(y=1.0, color='grey', linewidth=0.8, linestyle='--', alpha=0.6)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([CONDITION_LABELS.get(c, c) for c in test_conditions],
                           fontsize=8, rotation=15)
        ax.set_ylabel(f'Ratio {SIGNAL_LABELS[sig_key]}\n(condition/Pre-injection)', fontsize=7)
        ax.grid(False)

        # Stats: Mann-Whitney between conditions on ratio values
        # sessions_for_stats: filter to only sessions that have a matching pre-injection
        # ratio_sessions: wraps ratio values as fake sessions to reuse compute_stats()
        # Fake session trick: creates a minimal session dict with ratio_val as signal value
        # so compute_stats() can call session_median() and extract_signals() normally
        # This avoids duplicating the stats computation logic
        # Alternative: use compute_stats_from_values() directly with ratios_by_condition
        sessions_for_stats = {}
        for condition in test_conditions:
            sessions_for_stats[condition] = [s for s in all_data[condition]
                                              if (s['mouse'], get_date_key(s['pkl_path']),
                                                  get_parent_folder(s['pkl_path'])) in pre_index]
        # Build ratio signals for stats
        ratio_sessions = {}
        for condition in test_conditions:
            ratio_sessions[condition] = []
            for s in sessions_for_stats[condition]:
                date_key = get_date_key(s['pkl_path'])
                parent = get_parent_folder(s['pkl_path'])
                pre_s = pre_index.get((s['mouse'], date_key, parent))
                if pre_s is None:
                    continue
                fps_t, frames_t, signals_t = extract_signals(s['data'])
                fps_p, frames_p, signals_p = extract_signals(pre_s['data'])
                if sig_key in signals_t and sig_key in signals_p:
                    pre_med = session_median(signals_p[sig_key])
                    test_med = session_median(signals_t[sig_key])
                    if not np.isnan(pre_med) and pre_med > 0 and not np.isnan(test_med):
                        # Create a fake session with ratio as signal
                        ratio_val = test_med / pre_med
                        fake_session = {'data': {'fps': 1, 'frames': np.array([0, 1]),
                                                   sig_key: np.array([ratio_val, ratio_val]),
                                                   'pupil_area': np.array([ratio_val, ratio_val]),
                                                   'absolute_velocity': np.array([ratio_val, ratio_val]),
                                                   'pupil_center_[x,y]': np.zeros((2, 2)),
                                                   'pkl_name': s.get('pkl_path', '')},
                                        'mouse': s['mouse'], 'pkl_path': s['pkl_path']}
                        ratio_sessions[condition].append(fake_session)

        # Check if same mice appear in both conditions → paired test
        mice_per_cond = {c: [s['mouse'] for s in ratio_sessions.get(c, [])] for c in test_conditions}
        common_mice = set.intersection(*[set(m) for m in mice_per_cond.values()]) if mice_per_cond else set()
        use_paired = len(common_mice) >= 2
        stat_results = compute_stats(ratio_sessions, sig_key, paired=use_paired)
        stat_results = apply_fdr_correction(stat_results)
        add_stats_to_ax(ax, list(x_pos), stat_results, sig_key,
                       {c: c for c in test_conditions})

    # Legend
    condition_handles = [mpatches.Patch(color=CONDITION_COLORS.get(c, '#000000'),
                        label=CONDITION_LABELS.get(c, c)) for c in test_conditions]
    mouse_handles = [plt.Line2D([0], [0], marker='o', color='w',
                    markerfacecolor=MOUSE_COLORS[m], markersize=7, label=m)
                    for m in MOUSE_COLORS]
    axs[0].legend(handles=condition_handles + mouse_handles, fontsize=7,
                 bbox_to_anchor=(1.01, 1), loc='upper left')
    fig.suptitle('All mice — Ratio condition / Pre-injection', fontsize=12, y=0.98)
    out = os.path.join(output_dir, 'all_mice_ratio_summary.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ── Figure — Raw vs Post-processed ratio comparison ──────────────────────────
def plot_ratio_raw_vs_postprocessed(all_data, output_dir):
    """
    Compare ratio condition/Pre-injection between post-AnalyseFaceMap and post-processed data.
    """



    def compute_ratios(sessions_by_condition, use_raw=False):
        """
        Inner function that computes ratios for all conditions.
          use_raw=False: uses extract_signals() — post-processed data
          use_raw=True: uses extract_signals_raw() — post-AnalyseFaceMap data
        
        One figure per mouse + one all mice figure.
        Two box plots per condition: grey (raw) vs green (post-processed).
        Stats: Wilcoxon signed-rank between raw and post-processed ratios (paired per session).
        Purpose: quantify impact of PostProcessingGUI on condition/pre-injection ratios.
        """
        pre_index = {}
        for s in sessions_by_condition.get('Basal_Pre_Injection', []):
            date_key = get_date_key(s['pkl_path'])
            parent = get_parent_folder(s['pkl_path'])
            if date_key:
                pre_index[(s['mouse'], date_key, parent)] = s

        test_conditions = [c for c in sessions_by_condition if c != 'Basal_Pre_Injection']
        ratios = {c: {sig_key: [] for sig_key in SIGNAL_KEYS} for c in test_conditions}
        mice = {c: [] for c in test_conditions}

        for condition in test_conditions:
            for s in sessions_by_condition[condition]:
                date_key = get_date_key(s['pkl_path'])
                parent = get_parent_folder(s['pkl_path'])
                key = (s['mouse'], date_key, parent)
                if key not in pre_index:
                    continue
                pre_s = pre_index[key]

                if use_raw:
                    fps_t, frames_t, signals_t = extract_signals_raw(s)
                    fps_p, frames_p, signals_p = extract_signals_raw(pre_s)
                else:
                    fps_t, frames_t, signals_t = extract_signals(s['data'])
                    fps_p, frames_p, signals_p = extract_signals(pre_s['data'])

                for sig_key in SIGNAL_KEYS:
                    pre_med = session_median(signals_p.get(sig_key, np.array([np.nan])))
                    test_med = session_median(signals_t.get(sig_key, np.array([np.nan])))
                    if not np.isnan(pre_med) and pre_med > 0 and not np.isnan(test_med):
                        ratios[condition][sig_key].append(test_med / pre_med)
                mice[condition].append(s['mouse'])
        return ratios, mice

    test_conditions = [c for c in all_data if c != 'Basal_Pre_Injection']
    n_signals = len(SIGNAL_KEYS)
    n_conditions = len(test_conditions)

    # ── Per mouse figures ────────────────────────────────────────────────────
    for mouse in MICE:
        sessions_mouse = {c: [s for s in all_data[c] if s['mouse'] == mouse]
                         for c in all_data}
        sessions_mouse['Basal_Pre_Injection'] = [s for s in all_data['Basal_Pre_Injection']
                                                   if s['mouse'] == mouse]
        if not any(sessions_mouse[c] for c in test_conditions):
            continue

        ratios_pp, _ = compute_ratios(sessions_mouse, use_raw=False)
        ratios_raw, _ = compute_ratios(sessions_mouse, use_raw=True)

        fig, axs = plt.subplots(n_signals, n_conditions, figsize=(4*n_conditions, 3*n_signals))
        fig.subplots_adjust(hspace=0.5, wspace=0.4)
        if n_signals == 1: axs = axs.reshape(1, -1)
        if n_conditions == 1: axs = axs.reshape(-1, 1)

        for ci, condition in enumerate(test_conditions):
            label = CONDITION_LABELS.get(condition, condition)
            for si, sig_key in enumerate(SIGNAL_KEYS):
                ax = axs[si, ci]
                vals_raw = [v for v in ratios_raw[condition][sig_key] if not np.isnan(v)]
                vals_pp  = [v for v in ratios_pp[condition][sig_key]  if not np.isnan(v)]

                # Hide subplot if no data
                if not vals_raw and not vals_pp:
                    ax.set_visible(False)
                    continue

                for xi, (vals, color, lbl) in enumerate([
                    (vals_raw, '#95a5a6', 'Post-AnalyseFaceMap'),
                    (vals_pp,  '#2ecc71', 'Post-processed')
                ]):
                    if vals:
                        if len(vals) > 1:
                            bp = plot_box_or_bar(ax, vals, xi, color, width=0.35, alpha=0.6)
                            bp['boxes'][0].set_alpha(0.6)
                        jitter = np.random.uniform(-0.06, 0.06, len(vals))
                        ax.scatter(np.full(len(vals), xi) + jitter, vals,
                                  color=color, s=20, zorder=5, alpha=0.8)

                ax.axhline(y=1.0, color='grey', linewidth=0.8, linestyle='--', alpha=0.6)
                ax.set_xticks([0, 1])
                ax.set_xticklabels(['Raw', 'PP'], fontsize=7)
                if si == 0:
                    ax.set_title(label, fontsize=8)
                if ci == 0:
                    ax.set_ylabel(f'Ratio\n{SIGNAL_LABELS[sig_key]}', fontsize=6)
                ax.grid(False)

                # Stats
                if len(vals_raw) >= 2 and len(vals_pp) >= 2 and len(vals_raw) == len(vals_pp):
                    from scipy import stats as _stats
                    _, pval = _stats.wilcoxon(vals_raw, vals_pp)
                    stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
                    y_max = ax.get_ylim()[1]
                    y_range = y_max - ax.get_ylim()[0]
                    ax.plot([0, 0, 1, 1], [y_max+y_range*0.03, y_max+y_range*0.06,
                            y_max+y_range*0.06, y_max+y_range*0.03], color='black', linewidth=0.8)
                    ax.text(0.5, y_max+y_range*0.07, stars, ha='center', fontsize=8)
                    ax.set_ylim(top=y_max+y_range*0.15)

        fig.suptitle(f'{mouse} — Ratio condition/Pre-injection: Post-AnalyseFaceMap vs Post-processed',
                    fontsize=10, y=1.01)
        plt.tight_layout()
        out = os.path.join(output_dir, f'{mouse}_ratio_raw_vs_postprocessed.png')
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved: {out}')

    # ── All mice figure ──────────────────────────────────────────────────────
    ratios_pp_all, _ = compute_ratios(all_data, use_raw=False)
    ratios_raw_all, _ = compute_ratios(all_data, use_raw=True)

    fig, axs = plt.subplots(n_signals, n_conditions, figsize=(4*n_conditions, 3*n_signals))
    fig.subplots_adjust(hspace=0.5, wspace=0.4)
    if n_signals == 1: axs = axs.reshape(1, -1)
    if n_conditions == 1: axs = axs.reshape(-1, 1)

    for ci, condition in enumerate(test_conditions):
        label = CONDITION_LABELS.get(condition, condition)
        for si, sig_key in enumerate(SIGNAL_KEYS):
            ax = axs[si, ci]
            vals_raw = [v for v in ratios_raw_all[condition][sig_key] if not np.isnan(v)]
            vals_pp  = [v for v in ratios_pp_all[condition][sig_key]  if not np.isnan(v)]

            if not vals_raw and not vals_pp:
                ax.set_visible(False)
                continue

            for xi, (vals, color) in enumerate([(vals_raw, '#95a5a6'), (vals_pp, '#2ecc71')]):
                if vals:
                    if len(vals) > 1:
                        bp = plot_box_or_bar(ax, vals, xi, color, width=0.35, alpha=0.6)
                    jitter = np.random.uniform(-0.06, 0.06, len(vals))
                    ax.scatter(np.full(len(vals), xi) + jitter, vals,
                              color=color, s=20, zorder=5, alpha=0.8)

            ax.axhline(y=1.0, color='grey', linewidth=0.8, linestyle='--', alpha=0.6)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Raw', 'PP'], fontsize=7)
            if si == 0:
                ax.set_title(label, fontsize=8)
            if ci == 0:
                ax.set_ylabel(f'Ratio\n{SIGNAL_LABELS[sig_key]}', fontsize=6)
            ax.grid(False)

            if len(vals_raw) >= 2 and len(vals_pp) >= 2 and len(vals_raw) == len(vals_pp):
                from scipy import stats as _stats
                _, pval = _stats.wilcoxon(vals_raw, vals_pp)
                stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
                y_max = ax.get_ylim()[1]
                y_range = y_max - ax.get_ylim()[0]
                ax.plot([0, 0, 1, 1], [y_max+y_range*0.03, y_max+y_range*0.06,
                        y_max+y_range*0.06, y_max+y_range*0.03], color='black', linewidth=0.8)
                ax.text(0.5, y_max+y_range*0.07, stars, ha='center', fontsize=8)
                ax.set_ylim(top=y_max+y_range*0.15)

    fig.suptitle('All mice — Ratio condition/Pre-injection: Post-AnalyseFaceMap vs Post-processed',
                fontsize=10, y=1.01)
    plt.tight_layout()
    out = os.path.join(output_dir, 'all_mice_ratio_raw_vs_postprocessed.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

# ── Main ─────────────────────────────────────────────────────────────────────
def run_comparisons():
    """Main function — loads all data and generates all comparison figures.
    
    Called when script is run directly (if __name__ == '__main__').
    Initializes output directories, loads all pkl via get_facemap_data(), then calls all plot functions in sequence.
   """
    _init_output_dirs()
    # Ask user for base data directory
    import path_for_expe_facemap as pfe
    pfe.BASE = NAS_BASE # Fix pfe.BASE to NAS — ensures all pkl paths resolve correctly


    print('Loading data...')
    all_data = {}
    for exp in EXPERIMENTS.keys():
        print(f'\n  {exp}')
        all_data[exp] = get_facemap_data(exp)

    
    all_conditions = list(EXPERIMENTS.keys())
    all_mice = ['K1690', 'K1711', 'K1712']

    # ── 1. Per mouse — all conditions ─────────────────────────────────────────
    # For each mouse: filter sessions to this mouse only, skip if no sessions found
    # Generates 5 figures per mouse:
    #   - temporal median ± IQR (all conditions overlaid)
    #   - temporal individual sessions + median
    #   - summary box plots + stats
    #   - raw vs post-processed temporal
    #   - raw vs post-processed summary box plots
    for mouse in all_mice:
        sessions_by_condition = {}
        for condition in all_conditions:
            sessions = [s for s in all_data[condition] if s['mouse'] == mouse]
            if sessions:
                sessions_by_condition[condition] = sessions

        if not sessions_by_condition: #if a mouse hasn't any sessions in any of the condition skips to the next one
            continue

        title = f'{mouse} — All conditions'
        plot_temporal(sessions_by_condition, title,
                     os.path.join(OUTPUT_DIR_FILTERED, f'{mouse}_all_conditions_temporal.png'),
                     color_map=CONDITION_COLORS, label_map=CONDITION_LABELS)
        plot_temporal_individual(sessions_by_condition, title,
                     os.path.join(OUTPUT_DIR_FILTERED, f'{mouse}_all_conditions_individual.png'),
                     color_map=CONDITION_COLORS, label_map=CONDITION_LABELS)
        plot_summary(sessions_by_condition, title,
                    os.path.join(OUTPUT_DIR_FILTERED, f'{mouse}_all_conditions_summary.png'),
                    color_map=CONDITION_COLORS, label_map=CONDITION_LABELS)
        plot_temporal_raw_vs_postprocessed(sessions_by_condition, title,
                    os.path.join(OUTPUT_DIR_RAW, f'{mouse}_all_conditions_raw_vs_postprocessed_temporal.png'),
                    color_map=CONDITION_COLORS, label_map=CONDITION_LABELS)
        plot_summary_raw_vs_postprocessed(sessions_by_condition, title,
                    os.path.join(OUTPUT_DIR_RAW, f'{mouse}_all_conditions_raw_vs_postprocessed_summary.png'),
                    color_map=CONDITION_COLORS, label_map=CONDITION_LABELS)

    # ── 2. Per condition — all mice ────────────────────────────────────────────
    # For each condition: group sessions by mouse — each mouse = one curve/box
    # mouse_color_map/mouse_label_map: pass mouse colors and names instead of condition colors
    # point_color_key='condition': individual points colored by condition (not mouse)
    #   because all points are from the same condition — mouse color would be more informative
    #   but condition color used here for consistency with the figure title
    # Same 5 figures as section 1 but grouped by condition instead of mouse
    for condition in all_conditions:
        label = CONDITION_LABELS.get(condition, condition)
        sessions_by_mouse = {}
        mouse_color_map = {}
        mouse_label_map = {}
        for mouse in all_mice:
            sessions = [s for s in all_data[condition] if s['mouse'] == mouse]
            if sessions:
                sessions_by_mouse[mouse] = sessions
                mouse_color_map[mouse] = MOUSE_COLORS[mouse]
                mouse_label_map[mouse] = mouse

        if not sessions_by_mouse:
            continue

        title = f'{label} — All mice'
        plot_temporal(sessions_by_mouse, title,
                     os.path.join(OUTPUT_DIR_FILTERED, f'{condition}_all_mice_temporal.png'),
                     color_map=mouse_color_map, label_map=mouse_label_map)
        plot_temporal_individual(sessions_by_mouse, title,
                     os.path.join(OUTPUT_DIR_FILTERED, f'{condition}_all_mice_individual.png'),
                     color_map=mouse_color_map, label_map=mouse_label_map)
        plot_summary(sessions_by_mouse, title,
                    os.path.join(OUTPUT_DIR_FILTERED, f'{condition}_all_mice_summary.png'),
                    color_map=mouse_color_map, label_map=mouse_label_map,
                    point_color_key='condition')
        plot_temporal_raw_vs_postprocessed(sessions_by_mouse, title,
                    os.path.join(OUTPUT_DIR_RAW, f'{condition}_all_mice_raw_vs_postprocessed_temporal.png'),
                    color_map=mouse_color_map, label_map=mouse_label_map)
        plot_summary_raw_vs_postprocessed(sessions_by_mouse, title,
                    os.path.join(OUTPUT_DIR_RAW, f'{condition}_all_mice_raw_vs_postprocessed_summary.png'),
                    color_map=mouse_color_map, label_map=mouse_label_map)

     # ── 3. Cross-condition figures ─────────────────────────────────────────────
    plot_ratio_all_mice(all_data, OUTPUT_DIR_FILTERED)          # ratio condition/pre — all mice
    plot_ratio_raw_vs_postprocessed(all_data, OUTPUT_DIR_RAW)   # raw vs PP ratio comparison
    plot_saline_vs_ivabradine(all_data, OUTPUT_DIR_SALINE_IVA)  # saline vs ivabradine per mouse
    plot_pre_vs_test(all_data, OUTPUT_DIR_PRE_VS_TEST)          # pre vs test paired temporal
    plot_all_sessions(all_data, OUTPUT_DIR_FILTERED)            # all sessions flat view (PNG)
    plot_all_sessions_html(all_data, OUTPUT_DIR_FILTERED)       # all sessions interactive (HTML)

    print(f'\nAll figures saved in {OUTPUT_DIR}')
    print('\n' + '='*50)
    print('ℹ️  To generate global mask comparison figures:')
    print(f'   python {os.path.join(os.path.dirname(os.path.abspath(__file__)), "MaskVSnomask.py")}')
    print('='*50)

# Entry point — try/except catches any unhandled error and prints full traceback
# Allows identifying the exact failing function without losing the error message

if __name__ == '__main__':
    try:
        run_comparisons()
    except Exception as e:
        import traceback
        print(f'\n❌ Fatal error: {e}')
        traceback.print_exc()

