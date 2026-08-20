import os, pickle, numpy as np, matplotlib.pyplot as plt
from collections import defaultdict
import gc, sys
# Add script directory to path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Import shared utilities from ComparaisonFaceMap — avoids code duplication
from path_for_expe_facemap import MICE, MOUSE_COLORS


# ── HOW TO ADD A NEW MOUSE ────────────────────────────────────────────────────
# 1. Add mouse to MICE and MOUSE_COLORS in path_for_expe_facemap.py
# 2. If the mouse has videos processed with AND without corneal mask:
#    Add pairs to MASK_PAIRS in ComparaisonFaceMap.py:
#    ('KXXXX', 'FDDMMYYYY_Condition',
#     'KXXXX/path/to/VF1/without_mask_FacemapPose.pkl',
#     'KXXXX_Corneal_mask_files/date condition/with_mask/with_mask_FacemapPose.pkl')
# 3. Re-run MaskVSnomask.py to regenerate all mask comparison figures
# ─────────────────────────────────────────────────────────────────────────────
from ComparaisonFaceMap import (extract_signals, session_median, compute_median_iqr,
                                 bin_signal, SIGNAL_KEYS, SIGNAL_LABELS, MASK_PAIRS, NAS_BASE,
                                 compute_stats, apply_fdr_correction, add_stats_to_ax)

# Output directory — fixed path on NAS
BASE_OUT = '/media/nas8-2/ProjectCardioSense/Data_for_Ilyass/Comparaisons_FaceMap'
OUT_NAME = 'Without_vs_With_corneal_mask'

# Duplicate protection: rename existing folder to _old_X before creating new one
# Allows re-running the script without losing previous results
out_dir = os.path.join(BASE_OUT, OUT_NAME)
if os.path.exists(out_dir):
    x = 1
    while os.path.exists(os.path.join(BASE_OUT, f'{OUT_NAME}_old_{x}')):
        x += 1
    os.rename(out_dir, os.path.join(BASE_OUT, f'{OUT_NAME}_old_{x}'))
    print(f'⚠️  Existing folder renamed to {OUT_NAME}_old_{x}')
os.makedirs(out_dir)

# Build list of valid pairs — check both pkl files exist before adding
# pair_paths: list of (mouse, session_name, nomask_absolute_path, mask_absolute_path)
# Paths stored only — pkl loaded lazily per figure to avoid RAM overload
# Pairs with missing files are skipped with a warning
pair_paths = []
for mouse, session_name, rel_nomask, rel_mask in MASK_PAIRS:
    nomask_path = os.path.join(NAS_BASE, rel_nomask)
    mask_path = os.path.join(NAS_BASE, rel_mask)
    if not os.path.exists(nomask_path):
        print(f'⚠️  Without mask missing: {mouse} {session_name}')
        continue
    if not os.path.exists(mask_path):
        print(f'⚠️  With mask missing: {mouse} {session_name}')
        continue
    pair_paths.append((mouse, session_name, nomask_path, mask_path))
    print(f'✅ Found: {mouse} {session_name}')

n_signals = len(SIGNAL_KEYS)

# ── Per session figures ───────────────────────────────────────────────────────
# Load pkl pair, extract signals, generate temporal figure
# Both pkl loaded together — freed at end of iteration via del + gc.collect()
# Blue solid = without mask | Red dashed = with mask
# try/except per session — errors don't stop the full run
for mouse, session_name, nomask_path, mask_path in pair_paths:
    try:
        with open(nomask_path, 'rb') as f:
            data_n = pickle.load(f)
        with open(mask_path, 'rb') as f:
            data_m = pickle.load(f)

        fps_n, frames_n, signals_n = extract_signals(data_n)
        fps_m, frames_m, signals_m = extract_signals(data_m)

        # Figure temporelle
        fig, axs = plt.subplots(n_signals, 1, figsize=(20, 3 * n_signals))
        fig.subplots_adjust(top=0.94, hspace=0.35)
        if n_signals == 1:
            axs = [axs]
        for si, (ax, sig_key) in enumerate(zip(axs, SIGNAL_KEYS)):
            if sig_key in signals_n and len(frames_n) > 0:
                t_n, b_n = bin_signal(signals_n[sig_key], frames_n)
                ax.plot(t_n, b_n, color='#3498db', linewidth=1.5, label='Without mask')
            if sig_key in signals_m and len(frames_m) > 0:
                t_m, b_m = bin_signal(signals_m[sig_key], frames_m)
                ax.plot(t_m, b_m, color='#e74c3c', linewidth=1.5, linestyle='--', label='With mask')
            ax.set_ylabel(SIGNAL_LABELS[sig_key], fontsize=8)
            ax.grid(False)
            if si == n_signals - 1:
                ax.set_xlabel('Time (min)', fontsize=8)
            else:
                ax.set_xticklabels([])
        handles, labels_leg = axs[0].get_legend_handles_labels()
        unique = dict(zip(labels_leg, handles))
        axs[0].legend(unique.values(), unique.keys(), fontsize=8,
                      bbox_to_anchor=(1.01, 1), loc='upper left')
        fig.suptitle(f'{mouse} {session_name} — Without vs With corneal mask', fontsize=12, y=0.98)
        plt.savefig(os.path.join(out_dir, f'{mouse}_{session_name}_mask_temporal.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # Per session summary: bar chart of session median — Without (blue) vs With (red) mask
        # Value displayed above each bar for quick reading
        # del + gc.collect() after each session — frees RAM before loading next pair
        # traceback.print_exc() — prints full error stack for debugging without stopping the run
        fig, axs_sum = plt.subplots(1, n_signals, figsize=(3 * n_signals, 5))
        if n_signals == 1:
            axs_sum = [axs_sum]
        for ax, sig_key in zip(axs_sum, SIGNAL_KEYS):
            values = {}
            if sig_key in signals_n:
                values['Without mask'] = session_median(signals_n[sig_key])
            if sig_key in signals_m:
                values['With mask'] = session_median(signals_m[sig_key])
            colors_bar = {'Without mask': '#3498db', 'With mask': '#e74c3c'}
            for xi, (lbl, val) in enumerate(values.items()):
                if not np.isnan(val):
                    ax.bar(xi, val, color=colors_bar[lbl], alpha=0.7, width=0.6)
                    ax.text(xi, val * 1.01, f'{val:.2f}', ha='center', fontsize=7)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(list(values.keys()), fontsize=7, rotation=15)
            ax.set_title(SIGNAL_LABELS[sig_key], fontsize=7)
            ax.grid(False)
        fig.suptitle(f'{mouse} {session_name} — Without vs With corneal mask (median)', fontsize=10, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{mouse}_{session_name}_mask_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()

        del data_n, data_m, fps_n, frames_n, signals_n, fps_m, frames_m, signals_m
        gc.collect()
        print(f'  Saved: {mouse} {session_name}')

    except Exception as e:
        import traceback
        print(f'\n❌ Error on {mouse} {session_name}: {e}')
        traceback.print_exc()
        continue

print('\nGenerating global figures...')

# ── Global temporal figure — median ± IQR across all sessions ─────────────────
# Loads pkl one by one (lazy loading) to avoid RAM overload
# path_idx=2: nomask pkl path | path_idx=3: mask pkl path in pair_paths tuple
# t_grid: 1-minute bins from 0.5min to max session duration
# Interpolation on common grid — same logic as compute_median_iqr() in ComparaisonFaceMap
# IQR/2 shading — robust spread measure across all sessions
# del + gc.collect() after each pkl load — critical for RAM management on long sessions
fig, axs = plt.subplots(n_signals, 1, figsize=(20, 3 * n_signals))
fig.subplots_adjust(top=0.94, hspace=0.35)
if n_signals == 1:
    axs = [axs]

for si, (ax, sig_key) in enumerate(zip(axs, SIGNAL_KEYS)):
    for color, label, path_idx in [
        ('#3498db', 'Without corneal mask', 2),
        ('#e74c3c', 'With corneal mask', 3),
    ]:
        all_binned = []
        max_t = 0
        for pair in pair_paths:
            pkl_path = pair[path_idx]
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
                fps, frames, signals = extract_signals(data)
                if sig_key in signals and len(frames) > 0:
                    t_min, binned = bin_signal(signals[sig_key], frames)
                    if len(t_min) > 0 and t_min[-1] > max_t:
                        max_t = t_min[-1]
                    all_binned.append((t_min, binned))
                del data, fps, frames, signals
                gc.collect()
            except Exception as e:
                print(f'⚠️  Error loading {pkl_path}: {e}')
                continue

        if not all_binned or max_t == 0:
            continue
        t_grid = np.arange(0.5/60, max_t, 1.0)
        all_interp = np.full((len(all_binned), len(t_grid)), np.nan)
        for i, (t_min, binned) in enumerate(all_binned):
            valid = ~np.isnan(binned)
            if np.sum(valid) >= 2:
                all_interp[i] = np.interp(t_grid, t_min[valid], binned[valid],
                                          left=np.nan, right=np.nan)
        median = np.nanmedian(all_interp, axis=0)
        iqr = np.nanpercentile(all_interp, 75, axis=0) - np.nanpercentile(all_interp, 25, axis=0)
        ax.plot(t_grid, median, color=color, linewidth=1.5, label=label)
        ax.fill_between(t_grid,
                       np.where(np.isnan(iqr), np.nan, median - iqr/2),
                       np.where(np.isnan(iqr), np.nan, median + iqr/2),
                       color=color, alpha=0.2)
    ax.set_ylabel(SIGNAL_LABELS[sig_key], fontsize=8)
    ax.grid(False)
    if si == n_signals - 1:
        ax.set_xlabel('Time (min)', fontsize=8)
    else:
        ax.set_xticklabels([])

handles, labels_leg = axs[0].get_legend_handles_labels()
unique = dict(zip(labels_leg, handles))
axs[0].legend(unique.values(), unique.keys(), fontsize=8,
              bbox_to_anchor=(1.01, 1), loc='upper left')
fig.suptitle('Global — Without vs With corneal mask (median ± IQR)', fontsize=12, y=0.98)
out = os.path.join(out_dir, 'GLOBAL_mask_temporal.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')



# ── Global ratio mask/nomask figure ───────────────────────────────────────────
# ratio = median(with_mask) / median(without_mask) per session
# ratio = 1.0 (dashed line) = mask has no effect
# ratio > 1.0 = mask increases signal | ratio < 1.0 = mask decreases signal
# NaN appended if either median is invalid — ensures ratios_mask_nomask length = n_pairs
# Stats: Wilcoxon signed-rank vs 1.0 — H0: mask has no effect on signal
# One point per session — purple box plot if n >= 2
# del + gc.collect() after each pkl — lazy loading for RAM management

print('\nGenerating ratio mask/nomask figure...')
ratios_mask_nomask = defaultdict(list)

for mouse, session_name, nomask_path, mask_path in pair_paths:
    try:
        with open(nomask_path, 'rb') as f:
            data_n = pickle.load(f)
        fps_n, frames_n, signals_n = extract_signals(data_n)
        del data_n; gc.collect()

        with open(mask_path, 'rb') as f:
            data_m = pickle.load(f)
        fps_m, frames_m, signals_m = extract_signals(data_m)
        del data_m; gc.collect()

        for sig_key in SIGNAL_KEYS:
            med_n = session_median(signals_n.get(sig_key, np.array([np.nan])))
            med_m = session_median(signals_m.get(sig_key, np.array([np.nan])))
            if not np.isnan(med_n) and med_n > 0 and not np.isnan(med_m):
                ratios_mask_nomask[sig_key].append(med_m / med_n)
            else:
                ratios_mask_nomask[sig_key].append(np.nan)
        del signals_n, signals_m; gc.collect()
    except Exception as e:
        print(f'⚠️  Error ratio {session_name}: {e}')
        for sig_key in SIGNAL_KEYS:
            ratios_mask_nomask[sig_key].append(np.nan)

fig, axs = plt.subplots(1, n_signals, figsize=(3 * n_signals, 6))
if n_signals == 1:
    axs = [axs]

for ax, sig_key in zip(axs, SIGNAL_KEYS):
    ratios = [v for v in ratios_mask_nomask[sig_key] if not np.isnan(v)]
    if len(ratios) >= 1:
        jitter = np.random.uniform(-0.08, 0.08, len(ratios))
        ax.scatter(np.zeros(len(ratios)) + jitter, ratios,
                  color='#9b59b6', s=25, zorder=5, alpha=0.8)
        if len(ratios) >= 2:
            bp = ax.boxplot(ratios, positions=[0], widths=0.4,
                           patch_artist=True,
                           medianprops=dict(color='black', linewidth=2))
            bp['boxes'][0].set_facecolor('#9b59b6')
            bp['boxes'][0].set_alpha(0.5)
        ax.axhline(y=1.0, color='grey', linewidth=0.8, linestyle='--', alpha=0.6)
        # Stats vs 1
        if len(ratios) >= 2:
            from scipy import stats as scipy_stats
            _, pval = scipy_stats.wilcoxon(ratios, [1.0]*len(ratios))
            stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
            y_max = ax.get_ylim()[1]
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            ax.text(0, y_max + y_range*0.05, f'vs 1: {stars}',
                   ha='center', fontsize=8)
            ax.set_ylim(top=y_max + y_range*0.15)
    ax.set_xticks([0])
    ax.set_xticklabels(['With/Without mask'], fontsize=8)
    ax.set_title(SIGNAL_LABELS[sig_key], fontsize=7)
    ax.set_ylabel('Ratio (with/without mask)', fontsize=7)
    ax.grid(False)

fig.suptitle('Global — Ratio With/Without corneal mask per session', fontsize=11, y=1.01)
plt.tight_layout()
out = os.path.join(out_dir, 'GLOBAL_mask_ratio.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved: {out}')


# ── Per mouse ratio figures ────────────────────────────────────────────────────
# get_date_key(): local version using session_name (not pkl_path) — format F<YYYYMMDD>
# Iterates over mice found in pair_paths (not hardcoded list)
# mouse_pairs: all pairs for the current mouse
print('\nGenerating per-mouse ratio figures...')
import re as _re2

def get_date_key(sn):
    m = _re2.search(r'F(\d{8})', sn)
    return m.group(0) if m else None

all_mice = list(set([p[0] for p in pair_paths]))

for mouse in [m for m in MICE if m in all_mice]:
    mouse_pairs = [(m, sn, np_path, mp_path) for m, sn, np_path, mp_path in pair_paths if m == mouse]

    # Index pre-injection pairs by date for this mouse
    # pre_n_by_date: {F<YYYYMMDD>: nomask_path} | pre_m_by_date: {F<YYYYMMDD>: mask_path}
    # test_pairs: all non-pre-injection sessions for this mouse
    # Skip mouse if no test sessions found

    # For each test session: load test + pre-injection pkl for both nomask and mask
    # ratio = median(test) / median(pre) — computed separately for nomask and mask
    # Lazy loading with del + gc.collect() after each pkl — RAM management
    # Bar chart: blue = no mask | red = with mask | dashed line = ratio 1.0
    # One column per test session, one row per signal
    
    pre_n_by_date = {}
    pre_m_by_date = {}
    for m, sn, np_path, mp_path in mouse_pairs:
        if 'Pre-injection' in sn or 'Pre_injection' in sn:
            date_key = get_date_key(sn)
            if date_key:
                pre_n_by_date[date_key] = np_path
                pre_m_by_date[date_key] = mp_path

    # Get test sessions
    test_pairs = [(m, sn, np_path, mp_path) for m, sn, np_path, mp_path in mouse_pairs
                  if 'Pre-injection' not in sn and 'Pre_injection' not in sn]
    if not test_pairs:
        continue

    n_sessions = len(test_pairs)
    fig, axs = plt.subplots(n_signals, n_sessions, figsize=(4 * n_sessions, 3 * n_signals))
    fig.subplots_adjust(hspace=0.5, wspace=0.4)
    if n_signals == 1: axs = axs.reshape(1, -1) if n_sessions > 1 else np.array([[axs]])
    if n_sessions == 1: axs = axs.reshape(-1, 1)

    for ci, (m, sn, np_path, mp_path) in enumerate(test_pairs):
        date_key = get_date_key(sn)
        pre_n = pre_n_by_date.get(date_key)
        pre_m = pre_m_by_date.get(date_key)

        for si, sig_key in enumerate(SIGNAL_KEYS):
            ax = axs[si, ci]
            ratio_n = ratio_m = np.nan

            if pre_n:
                try:
                    with open(np_path, 'rb') as f: d = pickle.load(f)
                    _, _, sig_t = extract_signals(d); del d; gc.collect()
                    with open(pre_n, 'rb') as f: d = pickle.load(f)
                    _, _, sig_p = extract_signals(d); del d; gc.collect()
                    med_t = session_median(sig_t.get(sig_key, np.array([np.nan])))
                    med_p = session_median(sig_p.get(sig_key, np.array([np.nan])))
                    if not np.isnan(med_p) and med_p > 0:
                        ratio_n = med_t / med_p
                    del sig_t, sig_p; gc.collect()
                except Exception as e:
                    print(f'⚠️  {sn} nomask: {e}')

            if pre_m:
                try:
                    with open(mp_path, 'rb') as f: d = pickle.load(f)
                    _, _, sig_t = extract_signals(d); del d; gc.collect()
                    with open(pre_m, 'rb') as f: d = pickle.load(f)
                    _, _, sig_p = extract_signals(d); del d; gc.collect()
                    med_t = session_median(sig_t.get(sig_key, np.array([np.nan])))
                    med_p = session_median(sig_p.get(sig_key, np.array([np.nan])))
                    if not np.isnan(med_p) and med_p > 0:
                        ratio_m = med_t / med_p
                    del sig_t, sig_p; gc.collect()
                except Exception as e:
                    print(f'⚠️  {sn} mask: {e}')

            # Plot
            for xi, (ratio, color, label) in enumerate([(ratio_n, '#3498db', 'No mask'),
                                                         (ratio_m, '#e74c3c', 'With mask')]):
                if not np.isnan(ratio):
                    ax.bar(xi, ratio, color=color, alpha=0.7, width=0.5)
                    ax.text(xi, ratio + 0.01, f'{ratio:.2f}', ha='center', fontsize=6)

            ax.axhline(y=1.0, color='grey', linewidth=0.8, linestyle='--', alpha=0.6)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['No mask', 'With mask'], fontsize=7)
            if si == 0:
                cond = sn.split('_', 1)[1] if '_' in sn else sn
                ax.set_title(f'{cond}', fontsize=8)
            if ci == 0:
                ax.set_ylabel(f'Ratio\n{SIGNAL_LABELS[sig_key]}', fontsize=6)
            ax.grid(False)

    fig.suptitle(f'{mouse} — Ratio condition/Pre-injection (Without vs With mask)', fontsize=11, y=1.01)
    plt.tight_layout()
    out = os.path.join(out_dir, f'{mouse}_ratio_vs_preinjection.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

# ── Global ratio condition/pre-injection (nomask vs mask) ─────────────────────
# For each test session: ratio = median(condition) / median(pre-injection)
# Computed separately for nomask and mask — same pre-injection matched by (mouse, date)
# pre_index_nomask/mask: {(mouse, F<YYYYMMDD>): pkl_path}
# condition_labels_ratio: unique condition names extracted from session_name (after first _)
# ratios_nomask_pre/ratios_mask_pre: {(cond, sig_key): [ratio_per_session]}

# NaN appended on error — keeps list length consistent across conditions
# Lazy loading with del + gc.collect() — separate try/except for nomask and mask
# Box plot: blue = no mask | red = with mask | one column per condition
# Stats: Wilcoxon signed-rank between nomask and mask ratios (paired, equal n only)

# Figure skipped entirely if no test conditions found (condition_labels_ratio empty)

print('\nGenerating ratio condition/pre-injection figure...')
import re as _re

def get_date_key(session_name):
    m = _re.search(r'F(\d{8})', session_name)
    return m.group(0) if m else None

# Index pre-injection pairs by (mouse, date)
pre_index_nomask = {}
pre_index_mask = {}
for mouse, session_name, nomask_path, mask_path in pair_paths:
    if 'Pre-injection' in session_name or 'Pre_injection' in session_name:
        date_key = get_date_key(session_name)
        if date_key:
            pre_index_nomask[(mouse, date_key)] = nomask_path
            pre_index_mask[(mouse, date_key)] = mask_path

# Compute ratios
ratios_nomask_pre = defaultdict(list)
ratios_mask_pre = defaultdict(list)
condition_labels_ratio = []

for mouse, session_name, nomask_path, mask_path in pair_paths:
    if 'Pre-injection' in session_name or 'Pre_injection' in session_name:
        continue
    date_key = get_date_key(session_name)
    if not date_key:
        continue
    pre_n = pre_index_nomask.get((mouse, date_key))
    pre_m = pre_index_mask.get((mouse, date_key))
    if not pre_n or not pre_m:
        continue

    cond = session_name.split('_', 1)[1] if '_' in session_name else session_name
    if cond not in condition_labels_ratio:
        condition_labels_ratio.append(cond)

    try:
        with open(nomask_path, 'rb') as f: data_t = pickle.load(f)
        fps_t, frames_t, signals_t = extract_signals(data_t)
        del data_t; gc.collect()

        with open(pre_n, 'rb') as f: data_p = pickle.load(f)
        fps_p, frames_p, signals_p = extract_signals(data_p)
        del data_p; gc.collect()

        for sig_key in SIGNAL_KEYS:
            med_t = session_median(signals_t.get(sig_key, np.array([np.nan])))
            med_p = session_median(signals_p.get(sig_key, np.array([np.nan])))
            ratio = med_t / med_p if not np.isnan(med_p) and med_p > 0 and not np.isnan(med_t) else np.nan
            ratios_nomask_pre[(cond, sig_key)].append(ratio)
        del signals_t, signals_p; gc.collect()
    except Exception as e:
        print(f'⚠️  Error nomask ratio {session_name}: {e}')
        for sig_key in SIGNAL_KEYS:
            ratios_nomask_pre[(cond, sig_key)].append(np.nan)

    try:
        with open(mask_path, 'rb') as f: data_t = pickle.load(f)
        fps_t, frames_t, signals_t = extract_signals(data_t)
        del data_t; gc.collect()

        with open(pre_m, 'rb') as f: data_p = pickle.load(f)
        fps_p, frames_p, signals_p = extract_signals(data_p)
        del data_p; gc.collect()

        for sig_key in SIGNAL_KEYS:
            med_t = session_median(signals_t.get(sig_key, np.array([np.nan])))
            med_p = session_median(signals_p.get(sig_key, np.array([np.nan])))
            ratio = med_t / med_p if not np.isnan(med_p) and med_p > 0 and not np.isnan(med_t) else np.nan
            ratios_mask_pre[(cond, sig_key)].append(ratio)
        del signals_t, signals_p; gc.collect()
    except Exception as e:
        print(f'⚠️  Error mask ratio {session_name}: {e}')
        for sig_key in SIGNAL_KEYS:
            ratios_mask_pre[(cond, sig_key)].append(np.nan)

if condition_labels_ratio:
    n_cond = len(condition_labels_ratio)
    fig, axs = plt.subplots(n_signals, n_cond, figsize=(4 * n_cond, 3 * n_signals))
    fig.subplots_adjust(hspace=0.5, wspace=0.4)
    if n_signals == 1: axs = axs.reshape(1, -1)
    if n_cond == 1: axs = axs.reshape(-1, 1)

    for si, sig_key in enumerate(SIGNAL_KEYS):
        for ci, cond in enumerate(condition_labels_ratio):
            ax = axs[si, ci]
            vals_n = [v for v in ratios_nomask_pre.get((cond, sig_key), []) if not np.isnan(v)]
            vals_m = [v for v in ratios_mask_pre.get((cond, sig_key), []) if not np.isnan(v)]

            if vals_n or vals_m:
                data_box = [d for d in [vals_n, vals_m] if d]
                pos = [i for i, d in enumerate([vals_n, vals_m]) if d]
                bp = ax.boxplot(data_box, positions=pos, widths=0.4,
                               patch_artist=True,
                               medianprops=dict(color='black', linewidth=2))
                colors_bp = ['#3498db', '#e74c3c']
                for bi, box in enumerate(bp['boxes']):
                    box.set_facecolor(colors_bp[pos[bi]])
                    box.set_alpha(0.6)
                for xi, vals in enumerate([vals_n, vals_m]):
                    if vals:
                        jitter = np.random.uniform(-0.06, 0.06, len(vals))
                        ax.scatter(np.full(len(vals), xi) + jitter, vals,
                                  color=colors_bp[xi], s=20, zorder=5, alpha=0.8)
                ax.axhline(y=1.0, color='grey', linewidth=0.8, linestyle='--', alpha=0.6)

                # Stats
                if len(vals_n) >= 2 and len(vals_m) >= 2 and len(vals_n) == len(vals_m):
                    from scipy import stats as scipy_stats
                    _, pval = scipy_stats.wilcoxon(vals_n, vals_m)
                    stars = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'
                    y_max = ax.get_ylim()[1]
                    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                    ax.plot([0, 0, 1, 1], [y_max + y_range*0.05, y_max + y_range*0.08,
                            y_max + y_range*0.08, y_max + y_range*0.05], color='black', linewidth=0.8)
                    ax.text(0.5, y_max + y_range*0.09, stars, ha='center', fontsize=8)
                    ax.set_ylim(top=y_max + y_range*0.15)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['No mask', 'With mask'], fontsize=7)
            ax.set_title(f'{cond}\n{SIGNAL_LABELS[sig_key]}', fontsize=7)
            if ci == 0:
                ax.set_ylabel('Ratio vs Pre-injection', fontsize=7)
            ax.grid(False)

    fig.suptitle('Ratio condition/Pre-injection — Without vs With corneal mask', fontsize=11, y=1.01)
    plt.tight_layout()
    out = os.path.join(out_dir, 'GLOBAL_mask_ratio_vs_preinjection.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')

print('\nDone.')
