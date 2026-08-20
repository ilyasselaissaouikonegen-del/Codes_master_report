#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FaceMap experiment path manager - VF1 pkl version

=== HOW TO ADD A NEW SESSION TO EXPERIMENTS ===

Each condition in EXPERIMENTS is a list of tuples: (mouse_id, relative_path)
- mouse_id: e.g. 'K1690', 'K2010'
- relative_path: path relative to BASE (/media/nas8-2/ProjectCardioSense)

Examples:
  Old structure (K1690/K1711/K1712):
    ('K1690', '2025-03-12_10-56-54/1690_250312_Injection_Ivabradine_10mgkg')
  
  New structure (Data_for_Ilyass):
    ('K2010', 'Data_for_Ilyass/K2010/20260716-HeadFixed-Mouse-2010-Ivabradine_00')

find_pkl() searches recursively for *_FacemapPose.pkl in any subfolder.
Priority: split post_processing pkl > regular post_processing pkl > regular pkl.

For new sessions with Pre-injection + Injection in the same video:
  1. Run PostProcessingGUI.py on the video
  2. Click 'Start Injection' at the injection timepoint
  3. Save → creates 2 pkl:
     - *_Pre-injection_post_processing_FacemapPose.pkl → Basal_Pre_Injection
     - *_Injection_post_processing_FacemapPose.pkl → Injection_Ivabradine_10mgkg
  4. Add the session to BOTH conditions in EXPERIMENTS.

=== BASE ===
BASE = '/media/nas8-2/ProjectCardioSense'
All paths are relative to this BASE.



FaceMap experiment path manager - VF1 pkl version
Maps each experimental condition to the corresponding VF1 pkl files (local paths).
"""

import os
import glob
import pickle
import numpy as np

# Base path for all data on NAS — all session paths in EXPERIMENTS are relative to this
BASE = '/media/nas8-2/ProjectCardioSense'

# ── Experiment definitions ──────────────────────────────────────────────────
# Maps each experimental condition to a list of (mouse_id, relative_session_path) tuples.

EXPERIMENTS = {
    'Basal_Pre_Injection': [
        ('K1690', '2025-03-04_09-48-02/1690_250304_Basal_Pre-Injection'),
        ('K1690', '2025-03-05_10-15-01/1690_250305_Basal_Pre-Injection'),
        ('K1690', '2025-03-11_10-16-14/1690_250311_Basal_Pre-Injection'),
        ('K1690', '2025-03-12_10-56-54/1690_250312_Basal_Pre-Injection'),
        ('K1690', '2025-03-13_10-48-26/1690_250313_Basal_Pre-Injection'),
        ('K1690', '2025-03-31_09-46-24/1690_250331_Basal_Pre-Injection'),
        ('K1690', '2025-04-01_09-56-18/1690_250401_Basal_Pre-Injection'),
        ('K1690', '2025-04-03_09-55-25/1690_250403_Basal_Pre-Injection'),
        ('K1711', '2025-03-04_11-59-07/1711_250304_Basal_Pre-Injection'),
        ('K1711', '2025-03-05_14-29-26/1711_250305_Basal_Pre-Injection'),
        ('K1712', 'Injection_Pre_Saline_Ivabradine/2025-03-04_17-18-54/1712_250304_Basal_Pre-Injection'),
        ('K1712', 'Injection_Pre_Saline_Ivabradine/2025-03-05_16-32-38/1712_250305_Basal_Pre-Injection'),
        ('K1712', 'Injection_Pre_Saline_Ivabradine/2025-03-11_14-19-28/1712_250311_Basal_Pre-Injection'),
        ('K1712', 'Injection_Pre_Saline_Ivabradine/2025-03-12_15-07-02/1712_250312_Basal_Pre-Injection'),
        ('K1712', 'Injection_Pre_Saline_Ivabradine/2025-03-13_15-28-37/1712_250313_Basal_Pre-Injection'),
        ('K1712', 'Injection_Pre_Saline_Ivabradine/2025-03-31_13-56-30/1712_250331_Basal_Pre-Injection'),
        ('K1712', 'Injection_Pre_Saline_Ivabradine/2025-04-01_14-05-15/1712_250401_Basal_Pre-Injection'),
        ('K1712', 'Injection_Pre_Saline_Ivabradine/2025-04-02_14-10-36/1712_250402_Basal_Pre-Injection'),
        ('K2010', 'Data_for_Ilyass/K2010/20260716-HeadFixed-Mouse-2010-Ivabradine_00'),
    ],
    'Injection_Saline': [
        ('K1690', '2025-03-04_09-48-02/1690_250304_Injection_Saline'),
        ('K1690', '2025-03-13_10-48-26/1690_250313_Injection_Saline'),
        ('K1690', '2025-03-31_09-46-24/1690_250331_Injection_Saline'),
        ('K1711', '2025-03-04_11-59-07/1711_250304_Injection_Saline'),
        ('K1712', 'Injection_Pre_Saline_Ivabradine/2025-03-04_17-18-54/1712_250304_Injection_Saline'),
        ('K1712', 'Injection_Pre_Saline_Ivabradine/2025-03-13_15-28-37/1712_250313_Injection_Saline'),
        ('K1712', 'Injection_Pre_Saline_Ivabradine/2025-03-31_13-56-30/1712_250331_Injection_Saline'),
    ],
    'Injection_Ivabradine_5mgkg': [
        ('K1690', '2025-03-11_10-16-14/1690_250311_Injection_Ivabradine_5mgkg'),
        ('K1690', '2025-04-01_09-56-18/1690_250401_Injection_Ivabradine_5mgkg'),
        ('K1712', 'Injection_Pre_Saline_Ivabradine/2025-03-11_14-19-28/1712_250311_Injection_Ivabradine_5mgkg'),
        ('K1712', 'Injection_Pre_Saline_Ivabradine/2025-04-01_14-05-15/1712_250401_Injection_Ivabradine_5mgkg'),
    ],
    'Injection_Ivabradine_10mgkg': [
        ('K1690', '2025-03-12_10-56-54/1690_250312_Injection_Ivabradine_10mgkg'),
        ('K1690', '2025-04-03_09-55-25/1690_250403_Injection_Ivabradine_10mgkg'),
        ('K1712', 'Injection_Pre_Saline_Ivabradine/2025-03-12_15-07-02/1712_250312_Injection_Ivabradine_10mgkg'),
        ('K1712', 'Injection_Pre_Saline_Ivabradine/2025-04-02_14-10-36/1712_250402_Injection_Ivabradine_10mgkg'),
        ('K2010', 'Data_for_Ilyass/K2010/20260716-HeadFixed-Mouse-2010-Ivabradine_00'),
    ],
    'Injection_Ivabradine_20mgkg': [
        ('K1690', '2025-03-05_10-15-01/1690_250305_Injection_Ivabradine_20mgkg'),
        ('K1711', '2025-03-05_14-29-26/1711_250305_Injection_Ivabradine_20mgkg'),
        ('K1712', 'Injection_Pre_Saline_Ivabradine/2025-03-05_16-32-38/1712_250305_Injection_Ivabradine_20mgkg'),
    ],
}

def find_pkl(mouse, session_folder):
    """Find the best pkl file for a given mouse and session folder.
    Fully recursive search - robust to any folder structure changes.
    Path construction: if session_folder already contains mouse name (new structure),
    searches BASE/session_folder/ directly to avoid path duplication.
    
    Exclusions: metadata, old_version, old_VF1 folders.
    
    Priority (most recent file wins within each tier):
      1. post_processing pkl (*_post_processing_FacemapPose.pkl) — user-reviewed data
      2. regular pkl (*_FacemapPose.pkl) — pipeline output without post-processing
    
    Returns None if no pkl found — caller prints warning and skips session.
    """
    # Fully recursive search - works regardless of subfolder structure
    # If session_folder already contains mouse name, don't duplicate it in path
    if mouse in session_folder:
        search_path = os.path.join(BASE, session_folder, '**', '*_FacemapPose.pkl')
    else:
        search_path = os.path.join(BASE, mouse, session_folder, '**', '*_FacemapPose.pkl')
    pkls = glob.glob(search_path, recursive=True)
    pkls = [p for p in pkls if '_metadata' not in p and 'old_version' not in p and 'old_VF1' not in p]
    if len(pkls) == 0:
        return None
    # Priority: split post_processing pkl > regular post_processing pkl > regular pkl
    # Split pkls are named *_Pre-injection_post_processing* or *_Injection_post_processing*
    # All post_processing pkls take priority over regular pkls
    post_pkls = [p for p in pkls if 'post_processing' in p]
    regular_pkls = [p for p in pkls if 'post_processing' not in p]

    if post_pkls:
        if len(post_pkls) > 1:
            post_pkls = sorted(post_pkls, key=os.path.getmtime)
        return post_pkls[-1]

    if len(regular_pkls) > 1:
        regular_pkls = sorted(regular_pkls, key=os.path.getmtime)
    return regular_pkls[-1] if regular_pkls else None

def get_facemap_data(experiment):
    """
    Load all VF1 pkl files for a given experimental condition.
    
    INPUT:
        experiment (str): One of the keys in EXPERIMENTS dict.
    
    OUTPUT:
        list of dicts, each containing:
            - 'mouse': mouse ID (e.g. 'K1690')
            - 'session': session folder name
            - 'pkl_path': full path to the pkl file
            - 'data': loaded pkl dictionary
    
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment '{experiment}'. Available: {list(EXPERIMENTS.keys())}")
    
    results = []
    for mouse, session_folder in EXPERIMENTS[experiment]:
        pkl_path = find_pkl(mouse, session_folder)
        if pkl_path is None:
            print(f'⚠️  No pkl found: {mouse}/{session_folder}')
            continue
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            data['pkl_name'] = os.path.basename(pkl_path).replace('_FacemapPose.pkl', '')
            results.append({
                'mouse': mouse,
                'session': session_folder,
                'pkl_path': pkl_path,
                'data': data,
            })
            print(f'✅ {mouse} - {os.path.basename(pkl_path)}')
        except Exception as e:
            print(f'❌ {mouse}/{session_folder}: {e}')
    
    print(f'\n{len(results)}/{len(EXPERIMENTS[experiment])} sessions loaded for {experiment}')
    return results

if __name__ == '__main__':
    # Test
    print('=== Test: Basal_Pre_Injection ===')
    sessions = get_facemap_data('Basal_Pre_Injection')
    