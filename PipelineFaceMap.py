#Importation of the required libraries
import os
import numpy as np 
import matplotlib.pyplot as plt
import sys
from facemap import utils
from facemap.pose import refine_pose
import matplotlib.patches as patches
import cv2
import cv2 as _cv2
from cycler import cycler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from PIL import Image
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter1d


#Importing colors
cmap=plt.get_cmap('tab20') #import 20 colors to distinguish 14 different keypoints
plt.rc('axes', prop_cycle=(cycler('color', cmap(np.linspace(0, 1, 20))))) # Create a color cycle using the colormap using automatically the next color from tab20 


#Retrieving the arguments passed from AnalyseFacemap : .npy, .h5 and .avi
npy_filepaths_str = sys.argv[1]
h5_filepaths_str = sys.argv[2]
video_filepaths_str = sys.argv[3]

#Converting strings back to lists
npy_filepaths = npy_filepaths_str.split(',')
h5_filepaths = h5_filepaths_str.split(',')
video_filepaths = video_filepaths_str.split(',')
#print(video_filepaths) = debug line
#print(h5_filepaths) = debug line
N=len(h5_filepaths) #Counts the .h5 
print(N) #Number of .h5 found = number of videos 
#print(npy_filepaths) = debug line
#print(h5_filepaths) = debug line

             

#Extraction for the pupil-related data (area and center of mass)

def load_and_process_npy(file_path):
    # Charging the .npy file — patched for Ubuntu 22.04 / NumPy >= 1.20
    # Original used np.nditer on shape=() array which behaves differently in recent NumPy
    """Load a FaceMap _proc.npy file and extract pupil and motion data.
   
   Handles 3 NumPy loading formats for cross-version compatibility (NumPy < 1.20 vs >= 1.20).
   Extracts:
     - corrected_areas: pupil area per frame (px²) — the function "remove_peaks" was suppressed, velocity filter used instead
     - centers: pupil center of mass (x,y) per frame
     - movSVD: motion SVD components — 'movSVD' for recent FaceMap, 'movSv' for legacy files
     - x_range_ROI, y_range_ROI: pixel coordinates of the pupil ROI bounding box
   
   Raises KeyError if required fields (pupil, rois, movSVD) are missing — likely FaceMap was
   not run with the correct ROI settings.
   """

    raw = np.load(file_path, allow_pickle=True)

    # Unwrap scalar array (shape == ()) directly to dict
    if isinstance(raw, np.ndarray) and raw.shape == () and raw.dtype == object:
        data = raw.item() #if raw is a NumPy scalar
    elif isinstance(raw, np.ndarray) and raw.dtype == object and raw.size == 1:
        data = raw.flat[0] #if raw is not a NumPy scalar with shape==()
    else:
        data = raw #if raw isn't a NumPy scalar 

    if not isinstance(data, dict): #verifies if the data is a dict, if not error message => Maybe corrupted data
        raise ValueError(f"Unexepected format in {file_path} : expected = dict, received = {type(data)}")
    
    #Lists of raw datas
    areas, centers = [], [] #raw pupil areas and raw pupil centers
    corrected_areas = [] #areas after correction of aberrant data 

    # Pupil
    if 'pupil' not in data: #verifies if the proc.npy contains data of the pupil => Error probably = No pupil ROI in Facemap
        raise KeyError(f"C 'pupil' do not exist in {file_path}, try reprocess batch in FaceMap with pupil ROI")
    Pupildico = data['pupil'][0] #extract the first pupil ROI from the list that Facemap created for a video, in majority it'll contain just one ROI
    areas = Pupildico['area'] #extract the pupil areas that Facemap calculated = number of pixels in the eliptical ROI
    centers = Pupildico['com'] #extract the 2D com = center of mass table that contains the x and y coordinates
    #print(areas) = debub
    corrected_areas = areas  # remove_peaks suppressed — velocity filter handles artifacts

    # movSVD — priorite a 'movSVD', fallback sur 'movSv' pour anciens fichiers
    if 'movSVD' in data:
        movSVD = data['movSVD'] #2D matrix (n_frames, n_components) in recent Facemap files
    elif 'movSv' in data: 
        movSVD = data['movSv'] #1D matrix in old Facemap files
    else:
        raise KeyError(f"'movSVD' absent and 'movSv' absent either in {file_path} = try to reprocess in facemap with 'movSVD' checked in")

    # ROI
    if 'rois' not in data:
        raise KeyError(f"C the 'rois' are absent in {file_path}, try to reprocess with the ROIs in FaceMap") #if the pupils ROI has not been placed
    ROIdico = data['rois'][0] #extract first ROI of the list = pupils
    x_range_ROI = ROIdico['xrange'] #extract x positions of the pupil
    y_range_ROI = ROIdico['yrange'] #extract y positions of the pupil

    return corrected_areas, centers, movSVD, x_range_ROI, y_range_ROI #extracted data found in the list : NPY = load_and_process_npy(...)



#Extraction for the keypoints-related data (whiskers, eyes, mouth, nose,...)

# Likelihood thresholds calibrated on 47 sessions (K1690/K1711/K1712 post-processed pkl)
# Criterion: threshold = 0.60 if pct<0.6 <= 10%, else 0.50 (minimum)
# Corneal mask sessions excluded from calibration
LIKELIHOOD_THRESHOLDS = {
    'eye(back)':    0.50,  # median=0.589 pct<0.6=54.9% -> minimum
    'eye(bottom)':  0.60,  # median=0.762 pct<0.6=9.0%  -> 0.60
    'eye(front)':   0.50,  # median=0.712 pct<0.6=17.0% -> minimum
    'eye(top)':     0.50,  # median=0.739 pct<0.6=18.5% -> minimum
    'lowerlip':     0.50,  # median=0.598 pct<0.6=50.4% -> minimum
    'mouth':        0.50,  # median=0.599 pct<0.6=50.3% -> minimum
    'nose(bottom)': 0.50,  # median=0.519 pct<0.6=84.2% -> minimum
    'nose(r)':      0.50,  # median=0.505 pct<0.6=98.5% -> minimum
    'nose(tip)':    0.50,  # median=0.522 pct<0.6=91.3% -> minimum
    'nose(top)':    0.50,  # median=0.511 pct<0.6=97.3% -> minimum
    'nosebridge':   0.50,  # median=0.519 pct<0.6=94.9% -> minimum
    'paw':          0.50,  # median=0.564 pct<0.6=70.1% -> minimum
    'whisker(I)':   0.50,  # median=0.726 pct<0.6=10.8% -> minimum
    'whisker(II)':  0.60,  # median=0.733 pct<0.6=8.1%  -> 0.60
    'whisker(III)': 0.50,  # median=0.742 pct<0.6=10.4% -> minimum
}


# Keypoints excluded due to respiratory apparatus permanently placed on the nose
EXCLUDED_KEYPOINTS = ['nose(tip)', 'nose(r)', 'nose(top)', 'nosebridge', 'nose(bottom)']

def filter_keypoints_by_likelihood(x, y, likelihood, bodyparts):
    """Set keypoint coordinates to NaN when FaceMap prediction confidence is below threshold.
    
    Thresholds are keypoint-specific (see LIKELIHOOD_THRESHOLDS) and calibrated on 47 sessions.
    Default threshold of 0.70 applied to any keypoint not in LIKELIHOOD_THRESHOLDS.
    x and y converted to float first to allow NaN assignment (int arrays cannot hold NaN).
    Returns filtered x, y arrays with unreliable frames set to NaN."""
    x = x.astype(float) #float in order to assign np.NaN
    y = y.astype(float)
    for k, bp in enumerate(bodyparts):
        threshold = LIKELIHOOD_THRESHOLDS.get(str(bp), 0.70) #0.70 is the confidence set on the added bodyparts that aren't found in the dictionnary = most common threshold in litterature on DeepLabCut
        low_conf_frames = np.where(likelihood[k] < threshold)[0]
        x[k, low_conf_frames] = np.nan 
        y[k, low_conf_frames] = np.nan 
    return x, y


def load_and_process_h5(file_path):
    """Load a FaceMap Pose h5 file and extract filtered keypoint coordinates.
    
    keypoints_data: 3D array (3, n_bodyparts, n_frames)
      - dim 0: x coordinates, y coordinates, likelihood scores
    
    Two filtering steps applied:
      1. Likelihood filter: frames below threshold set to NaN (see LIKELIHOOD_THRESHOLDS)
      2. Excluded keypoints: nose keypoints permanently set to NaN (masked by respiratory apparatus)
    
    Returns Dicorep: dict {bodypart_name: {'x': array, 'y': array, 'likelihood': array}}
    This format is used throughout the pipeline for all keypoint access."""
    keypoints_data = utils.load_keypoints(refine_pose.BODYPARTS, file_path)
    # keypoints_data: 3D array (3, n_bodyparts, n_frames) — dims: x, y, likelihood
    pose_x_coord = keypoints_data[0, :, :]
    pose_y_coord = keypoints_data[1, :, :]
    pose_likelihood = keypoints_data[2, :, :]
    # Filtration with likelihood
    pose_x_coord, pose_y_coord = filter_keypoints_by_likelihood(pose_x_coord, pose_y_coord, pose_likelihood, refine_pose.BODYPARTS)
    # Keypoints hidden by the respiratory apparatus are excluded
    if EXCLUDED_KEYPOINTS:
        bodyparts_list = [str(bp) for bp in refine_pose.BODYPARTS]
        for bp in EXCLUDED_KEYPOINTS:
            if bp in bodyparts_list:
                idx = bodyparts_list.index(bp)
                pose_x_coord[idx, :] = np.nan
                pose_y_coord[idx, :] = np.nan
                print(f'Keypoint excluded: {bp}')
    Dicorep={} #output dictionnary
    for i, bodypart in enumerate(refine_pose.BODYPARTS):
        Dicorep[str(bodypart)]={} #each dimension of a bodypart keypoint is stocked into a 1D dictionnary
        Dicorep[str(bodypart)]['x']=pose_x_coord[i,:]
        Dicorep[str(bodypart)]['y']=pose_y_coord[i,:]
        Dicorep[str(bodypart)]['likelihood']=pose_likelihood[i,:]
    return Dicorep


def compute_velocity(x, y, fps):
    """
    Compute absolute velocity (px/s) of a keypoint from x,y coordinate arrays.
    
    The velocity is calculated by comparison of the position between the frame i and i-1
    Velocity at frame i = sqrt((x[i]-x[i-1])² + (y[i]-y[i-1])²) * fps
    
    Frame 0 has no previous frame — offset [0] is prepended to x[:-1] and y[:-1]
    to keep array lengths equal, then frame 0 velocity is set to NaN (invalid).
    
    Returns speed array of same length as input, in pixels/second.
    Direction is ignored — only magnitude (absolute velocity) is computed..
    """
    offset = np.array([0]) #concatenating the table of frames by setting the frame 0 to 0 to have a table of velocities as long as the position table
    vx = x - np.concatenate((offset, x[:-1])) #the x-offset position = the i-1 position to calculate the velocity
    vy = y - np.concatenate((offset, y[:-1])) #Velocity = (i position) - (i-1 position)
    speed = np.sqrt(vx**2 + vy**2) * fps #calculated the absolute velocity because we don't look at the direction in this plot and convert pixels/frame into pixels/second
    speed[0] = np.nan #set to NaN the first frame which is wrong because compared to the created position of frame -1
    return speed


def control_image_1(Dico, name):
 """Generate the main temporal control figure (5 subplots).
 
 Subplots: pupil velocity, pupil area, whisker I/II/III velocity.
 X-axis: time in seconds, ticks every 500s.
 
 Three marker types at the bottom of each subplot:
   - Red * : frames set to NaN by likelihood/grooming filter
   - Purple * : frames set to NaN by velocity > 800px/s filter
   - Green . : attention markers (luminosity/white pixels anomaly, not filtered)
 
 Saved as *_FacemapPose_temporel.png alongside the pkl."""
 fig,axs=plt.subplots(5,1,figsize=(40,11)) #matplotlib figure
 frames = np.array(Dico['frames'])
 if 'fps' not in Dico:
     raise KeyError("FPS not found in Dico — make sure the video was processed with the current pipeline version")
 fps_plot = Dico['fps'] #forces to use the .avi to set the fps that allows to convert frames into seconds 
 print(f'FPS used for plotting: {fps_plot}') #Verification of the fps value that's detected : problem if the value does not match with what's expected (mostly fps will be = 30)
 
 # Liquid frames and grooming frames
 nan_velocity_frames_display = np.array(Dico.get('nan_velocity', []), dtype=int)  # velocity NaN frames (orange markers)
 nan_grooming_frames = np.array(Dico.get('nan_grooming_likelihood', []), dtype=int)
 nan_velocity_frames = np.array(Dico.get('nan_velocity', []), dtype=int)
 nan_attention_frames = np.array(Dico.get('nan_attention', []), dtype=int)
 legend_elements = [
     Line2D([0],[0],marker='*',color='w',markerfacecolor='red',markersize=8,label='NaN — likelihood/grooming'),
     Line2D([0],[0],marker='*',color='w',markerfacecolor='#8e44ad',markersize=8,label='NaN — velocity > 800px/s'),
     Line2D([0],[0],marker='.',color='w',markerfacecolor='green',markersize=8,label='Attention — luminosity/white pixels anomaly (not filtered)'),
 ]
 
 def plot_nan_markers(ax, nan_idx, frames, nan_velocity_frames_display, nan_grooming_frames, data, nan_velocity_frames=None, attention_frames=None):
     """Draw NaN frame markers at the bottom of a subplot.
    
    Three marker types placed 5% below the minimum data value:
      - Red * (grooming_mask): frames set to NaN by likelihood filter (low confidence keypoints)
      - Purple * (velocity_mask): frames set to NaN by velocity > 800px/s filter
      - Green . (attention_frames): luminosity/white pixels anomalies — informative only, not filtered
    
    clip_on=False allows markers to appear below the axis limits.
    marker_y computed from valid (non-NaN) data range for consistent positioning.
    
    Note: liquid_mask and grooming_mask naming is legacy — liquid_mask now refers to velocity frames,
    grooming_mask to likelihood frames."""
     if len(nan_idx) == 0: return 
     liquid_mask = np.isin(nan_idx, nan_velocity_frames_display)
     grooming_mask = ~liquid_mask
     velocity_mask = np.isin(nan_idx, nan_velocity_frames) if nan_velocity_frames is not None else np.zeros(len(nan_idx), dtype=bool)
     # Redefine grooming as not liquid and not velocity
     grooming_mask = ~liquid_mask & ~velocity_mask
     valid_data = data[~np.isnan(data)]
     if len(valid_data) > 0:
         y_min = np.min(valid_data)
         y_max = np.max(valid_data)
         y_range = y_max - y_min if y_max > y_min else 1.0
         marker_y = y_min - 0.05 * y_range
     else:
         marker_y = 0
     if np.sum(grooming_mask) > 0:
         ax.plot(frames[nan_idx[grooming_mask]], np.full(np.sum(grooming_mask), marker_y), 'r*', markersize=4, clip_on=False)
     if np.sum(velocity_mask) > 0:
         ax.plot(frames[nan_idx[velocity_mask]], np.full(np.sum(velocity_mask), marker_y),
                marker='*', color='#8e44ad', markersize=4, linestyle='None', clip_on=False)
     if np.sum(liquid_mask) > 0:
         ax.plot(frames[nan_idx[liquid_mask]], np.full(np.sum(liquid_mask), marker_y), 'r*', markersize=4, clip_on=False)
     # Attention markers — green dots for luminosity/white pixels anomalies
     if attention_frames is not None and len(attention_frames) > 0:
         attn_arr = np.array(attention_frames, dtype=int)
         attn_in_frames = attn_arr[attn_arr < len(frames)]
         ax.plot(frames[attn_in_frames], np.full(len(attn_in_frames), marker_y),
                marker='.', color='green', markersize=3, linestyle='None', clip_on=False, alpha=0.5)
 
    # Subplot 0 — pupil center absolute velocity
    # absolute_velocity is in px/frame — multiplied by fps to convert to px/s
 pup_vel = np.array(Dico['absolute_velocity'], dtype=float) * fps_plot #extract the datas from Facemap and multiplies by fps_plot to convert in seconds
 axs[0].plot(frames, pup_vel, 'b-') #velocity curve in blue
 nan_pup = np.where(np.isnan(pup_vel))[0] #finds the NaN frames = velocity filter + likelihood filter
 plot_nan_markers(axs[0], nan_pup, frames, nan_velocity_frames_display, nan_grooming_frames, pup_vel, nan_velocity_frames=nan_velocity_frames, attention_frames=nan_attention_frames) 
 n_pup = len(nan_pup) 
 pct_pup = round(100*n_pup/len(frames),1) #calculates the % of NaN frames
 axs[0].set_title(f'Absolute velocity of the center of the pupil — NaN: {n_pup} ({pct_pup}%)')
 axs[0].set_xlabel('t (seconds)')
 axs[0].set_ylabel('V(t) (pixels/second)')
 axs[0].legend(handles=legend_elements)
 axs[0].grid(False)
 
     # Subplot 1 — pupil area (px²)
     # Raw pixel area — converted to mm² in ComparaisonFaceMap.py using px_per_mm calibration
 pup_area = np.array(Dico['pupil_area'], dtype=float)
 axs[1].plot(frames, pup_area, 'b-')
 nan_area = np.where(np.isnan(pup_area))[0]
 plot_nan_markers(axs[1], nan_area, frames, nan_velocity_frames_display, nan_grooming_frames, pup_area, nan_velocity_frames=nan_velocity_frames, attention_frames=nan_attention_frames)
 n_area = len(nan_area)
 pct_area = round(100*n_area/len(frames),1)
 axs[1].set_title(f'Area of the pupil — NaN: {n_area} ({pct_area}%)')
 axs[1].set_xlabel('t (seconds)')
 axs[1].set_ylabel('A(t)')
 axs[1].legend(handles=legend_elements)
 axs[1].grid(False)
 
     # Subplots 2-4 — whisker I/II/III absolute velocity
     # Velocity computed from keypoint x,y coordinates via compute_velocity()
     # NaN frames include: likelihood filter + liquid/eye_closed from PostProcessingGUI
     # Whisker velocity kept for eye_closed segments (only set to NaN for liquid segments)

     # X ticks every 500s — consistent with ComparaisonFaceMap figures
 
    # Save as *_FacemapPose_temporel.png in the session folder
    # File then moved to VF1/ by the pipeline after all figures are generated
 
    # Subplot 2 — whisker(I)
 whx,why=np.array(Dico['whisker(I)']['x'],dtype=float),np.array(Dico['whisker(I)']['y'],dtype=float) #extract the data from Facemap
 absolute_whisker_I_velocity = compute_velocity(whx, why, fps_plot)
 axs[2].plot(frames, absolute_whisker_I_velocity, 'b-') 
 nan_wh = np.where(np.isnan(absolute_whisker_I_velocity))[0]
 plot_nan_markers(axs[2], nan_wh, frames, nan_velocity_frames_display, nan_grooming_frames, absolute_whisker_I_velocity, nan_velocity_frames=nan_velocity_frames, attention_frames=nan_attention_frames)
 n_wh = len(nan_wh)
 pct_wh = round(100*n_wh/len(frames),1)
 axs[2].set_title(f'Absolute velocity of the whisker(I) — NaN: {n_wh} ({pct_wh}%)')
 axs[2].set_xlabel('t (seconds)')
 axs[2].set_ylabel('V(t) (pixels/second)')
 axs[2].legend(handles=legend_elements)
 axs[2].grid(False)
 
     # Subplot 3 — whisker(II)
 wh2x,wh2y=np.array(Dico['whisker(II)']['x'],dtype=float),np.array(Dico['whisker(II)']['y'],dtype=float)
 absolute_whisker_II_velocity = compute_velocity(wh2x, wh2y, fps_plot)
 axs[3].plot(frames, absolute_whisker_II_velocity, 'b-')
 nan_wh2 = np.where(np.isnan(absolute_whisker_II_velocity))[0]
 plot_nan_markers(axs[3], nan_wh2, frames, nan_velocity_frames_display, nan_grooming_frames, absolute_whisker_II_velocity, nan_velocity_frames=nan_velocity_frames, attention_frames=nan_attention_frames)
 n_wh2 = len(nan_wh2)
 pct_wh2 = round(100*n_wh2/len(frames),1)
 axs[3].set_title(f'Absolute velocity of the whisker(II) — NaN: {n_wh2} ({pct_wh2}%)')
 axs[3].set_xlabel('t (seconds)')
 axs[3].set_ylabel('V(t) (pixels/second)')
 axs[3].legend(handles=legend_elements)
 axs[3].grid(False)
 
     # Subplot 4 — whisker(III)
 wh3x,wh3y=np.array(Dico['whisker(III)']['x'],dtype=float),np.array(Dico['whisker(III)']['y'],dtype=float)
 absolute_whisker_III_velocity = compute_velocity(wh3x, wh3y, fps_plot)
 axs[4].plot(frames, absolute_whisker_III_velocity, 'b-')
 nan_wh3 = np.where(np.isnan(absolute_whisker_III_velocity))[0]
 plot_nan_markers(axs[4], nan_wh3, frames, nan_velocity_frames_display, nan_grooming_frames, absolute_whisker_III_velocity, nan_velocity_frames=nan_velocity_frames, attention_frames=nan_attention_frames)
 n_wh3 = len(nan_wh3)
 pct_wh3 = round(100*n_wh3/len(frames),1)
 axs[4].set_title(f'Absolute velocity of the whisker(III) — NaN: {n_wh3} ({pct_wh3}%)')
 axs[4].set_xlabel('t (seconds)')
 axs[4].set_ylabel('V(t) (pixels/second)')
 axs[4].legend(handles=legend_elements)
 axs[4].grid(False)
 # X ticks every 500 seconds
 if len(frames) > 0:
     tick_vals = np.arange(0, frames[-1] + 500, 500)
     for ax in axs:
         ax.set_xticks(tick_vals)
         ax.set_xticklabels([str(int(t)) for t in tick_vals], fontsize=7)
 

 
 plt.tight_layout()
 output_file=str(name)+'_temporel.png'
 plt.savefig(output_file)
 plt.close()
 
 
def control_image_1_raw_data(Dico, name, h5_path):
    """
    Same layout as control_image_1 but showing raw unfiltered data from the h5 file.
    
    Key difference from control_image_1:
      - Whisker velocities computed from raw h5 keypoints (before likelihood filtering)
      - Pupil velocity and area still from Dico (no raw equivalent available from FaceMap)
      - No NaN markers drawn — filtered frame count shown in subplot title as '% filtered'
      - 6 subplots: pupil velocity, pupil area, whisker I/II/III velocity, nose tip velocity
    
    Purpose: visual quality control to assess how many frames were removed by filtering.
    Saved as *_FacemapPose_temporel_raw_data.png alongside the pkl.
    """
    raw_data = utils.load_keypoints(refine_pose.BODYPARTS, h5_path) #extract the keypoints directly from .h5 to avoid the filters
    raw_x = raw_data[0].astype(float)
    raw_y = raw_data[1].astype(float)
    bodyparts_list = [str(b) for b in refine_pose.BODYPARTS]

    frames = np.array(Dico['frames'])
    n_frames = len(frames)

    if 'fps' not in Dico:
        raise KeyError("FPS not found in Dico — make sure the video was processed with the current pipeline version")
    fps_plot = Dico['fps']

    fig, axs = plt.subplots(6, 1, figsize=(40, 13))

    # Subplot 0 — pupil velocity
    pup_vel = np.array(Dico['absolute_velocity'], dtype=float) * fps_plot
    nan_pup = np.isnan(pup_vel)
    pct_pup = round(100*np.sum(nan_pup)/n_frames, 1)
    axs[0].plot(frames, pup_vel, 'b-', linewidth=0.5)
    axs[0].set_title(f'Absolute velocity of the center of the pupil (raw) — filtered: {np.sum(nan_pup)} ({pct_pup}%)')
    axs[0].set_xlabel('t (seconds)')
    axs[0].set_ylabel('V(t) (pixels/second)')
    axs[0].grid(False)

    # Subplot 1 — pupil area
    pup_area = np.array(Dico['pupil_area'], dtype=float)
    nan_area = np.isnan(pup_area)
    pct_area = round(100*np.sum(nan_area)/n_frames, 1)
    axs[1].plot(frames, pup_area, 'b-', linewidth=0.5)
    axs[1].set_title(f'Area of the pupil (raw) — filtered: {np.sum(nan_area)} ({pct_area}%)')
    axs[1].set_xlabel('t (seconds)')
    axs[1].set_ylabel('A(t)')
    axs[1].grid(False)

    # Subplot 2 — whisker(I) velocity raw
    wh_idx = bodyparts_list.index('whisker(I)') if 'whisker(I)' in bodyparts_list else None
    if wh_idx is not None:
        abs_wh_vel_raw = compute_velocity(raw_x[wh_idx], raw_y[wh_idx], fps_plot)
        wh_filtered = np.isnan(np.array(Dico['whisker(I)']['x'], dtype=float))
        n_wh_filt = np.sum(wh_filtered)
        pct_wh = round(100*n_wh_filt/n_frames, 1)
        axs[2].plot(frames, abs_wh_vel_raw, 'b-', linewidth=0.5)
    axs[2].set_title(f'Absolute velocity of the whisker(I) (raw) — filtered: {n_wh_filt} ({pct_wh}%)')
    axs[2].set_xlabel('t (seconds)')
    axs[2].set_ylabel('V(t) (pixels/second)')
    axs[2].grid(False)

    # Subplot 3 — whisker(II) velocity raw
    wh2_idx = bodyparts_list.index('whisker(II)') if 'whisker(II)' in bodyparts_list else None
    if wh2_idx is not None:
        abs_wh2_vel_raw = compute_velocity(raw_x[wh2_idx], raw_y[wh2_idx], fps_plot)
        wh2_filtered = np.isnan(np.array(Dico['whisker(II)']['x'], dtype=float))
        n_wh2_filt = np.sum(wh2_filtered)
        pct_wh2 = round(100*n_wh2_filt/n_frames, 1)
        axs[3].plot(frames, abs_wh2_vel_raw, 'b-', linewidth=0.5)
    axs[3].set_title(f'Absolute velocity of the whisker(II) (raw) — filtered: {n_wh2_filt} ({pct_wh2}%)')
    axs[3].set_xlabel('t (seconds)')
    axs[3].set_ylabel('V(t) (pixels/second)')
    axs[3].grid(False)

    # Subplot 4 — whisker(III) velocity raw
    wh3_idx = bodyparts_list.index('whisker(III)') if 'whisker(III)' in bodyparts_list else None
    if wh3_idx is not None:
        abs_wh3_vel_raw = compute_velocity(raw_x[wh3_idx], raw_y[wh3_idx], fps_plot)
        wh3_filtered = np.isnan(np.array(Dico['whisker(III)']['x'], dtype=float))
        n_wh3_filt = np.sum(wh3_filtered)
        pct_wh3 = round(100*n_wh3_filt/n_frames, 1)
        axs[4].plot(frames, abs_wh3_vel_raw, 'b-', linewidth=0.5)
    axs[4].set_title(f'Absolute velocity of the whisker(III) (raw) — filtered: {n_wh3_filt} ({pct_wh3}%)')
    axs[4].set_xlabel('t (seconds)')
    axs[4].set_ylabel('V(t) (pixels/second)')
    axs[4].grid(False)
    # X ticks every 500 seconds
    if len(frames) > 0:
        tick_vals = np.arange(0, frames[-1] + 500, 500)
        for ax in axs:
            ax.set_xticks(tick_vals)
            ax.set_xticklabels([str(int(t)) for t in tick_vals], fontsize=7)

    # Subplot 5 — nose(tip) velocity raw
    nose_idx = bodyparts_list.index('nose(tip)') if 'nose(tip)' in bodyparts_list else None
    if nose_idx is not None:
        abs_nose_vel_raw = compute_velocity(raw_x[nose_idx], raw_y[nose_idx], fps_plot)
        nose_filtered = np.isnan(np.array(Dico['nose(tip)']['x'], dtype=float))
        n_nose_filt = np.sum(nose_filtered)
        pct_nose = round(100*n_nose_filt/n_frames, 1)
        axs[5].plot(frames, abs_nose_vel_raw, 'b-', linewidth=0.5)
    axs[5].set_title(f'Absolute nose tip velocity (raw) — filtered: {n_nose_filt} ({pct_nose}%)')
    axs[5].set_xlabel('t (seconds)')
    axs[5].set_ylabel('V(t) (pixels/second)')
    axs[5].grid(False)

    plt.tight_layout()
    output_file = str(name) + '_temporel_raw_data.png'
    plt.savefig(output_file)
    plt.close()
    print(f'Saved: {output_file}')



def control_image_1_HTML(Dico, name): #interactive version
 """Interactive Plotly version of control_image_1 — same 6 signals, zoomable and linked axes.
    
    Uses Plotly subplots instead of matplotlib for interactive exploration in a browser.
    All x-axes synchronized (matches='x') so zooming one subplot zooms all simultaneously.
    Shows filtered data (post-likelihood filter) — same as control_image_1, not raw.
    
    Saved as *_FacemapPose_temporel.html alongside the pkl.
    Open in any browser — no Python required to view."""
 if 'fps' not in Dico:
     raise KeyError("FPS not found in Dico — make sure the video was processed with the current pipeline version")
 fps_plot = Dico['fps']
 fig = make_subplots(rows=6, cols=1)

 # Row 1 — pupil velocity
 fig.add_trace(go.Scatter(x=Dico['frames'], y=np.array(Dico['absolute_velocity'], dtype=float) * fps_plot, mode='lines', name='Pupil Velocity'), row=1, col=1)
 fig.update_xaxes(title=dict(text="Time (s)", font=dict(size=20)), row=1, col=1)
 fig.update_yaxes(title=dict(text="V(t) (pixels/second)", font=dict(size=20)), row=1, col=1)

 # Row 2 — pupil area
 fig.add_trace(go.Scatter(x=Dico['frames'], y=Dico['pupil_area'], mode='lines', name='Pupil Area'), row=2, col=1)
 fig.update_xaxes(title=dict(text="Time (s)", font=dict(size=20)), row=2, col=1)
 fig.update_yaxes(title=dict(text="A(t)", font=dict(size=20)), row=2, col=1)

 # Row 3 — whisker(I) velocity
 whx, why = np.array(Dico['whisker(I)']['x'], dtype=float), np.array(Dico['whisker(I)']['y'], dtype=float)
 abs_wh1_vel = compute_velocity(whx, why, fps_plot)
 fig.add_trace(go.Scatter(x=Dico['frames'], y=abs_wh1_vel, mode='lines', name='Whisker I Velocity'), row=3, col=1)
 fig.update_xaxes(title=dict(text="Time (s)", font=dict(size=20)), row=3, col=1)
 fig.update_yaxes(title=dict(text="V(t) (pixels/second)", font=dict(size=20)), row=3, col=1)

 # Row 4 — whisker(II) velocity
 wh2x, wh2y = np.array(Dico['whisker(II)']['x'], dtype=float), np.array(Dico['whisker(II)']['y'], dtype=float)
 abs_wh2_vel = compute_velocity(wh2x, wh2y, fps_plot)
 fig.add_trace(go.Scatter(x=Dico['frames'], y=abs_wh2_vel, mode='lines', name='Whisker II Velocity'), row=4, col=1)
 fig.update_xaxes(title=dict(text="Time (s)", font=dict(size=20)), row=4, col=1)
 fig.update_yaxes(title=dict(text="V(t) (pixels/second)", font=dict(size=20)), row=4, col=1)

 # Row 5 — whisker(III) velocity
 wh3x, wh3y = np.array(Dico['whisker(III)']['x'], dtype=float), np.array(Dico['whisker(III)']['y'], dtype=float)
 abs_wh3_vel = compute_velocity(wh3x, wh3y, fps_plot)
 fig.add_trace(go.Scatter(x=Dico['frames'], y=abs_wh3_vel, mode='lines', name='Whisker III Velocity'), row=5, col=1)
 fig.update_xaxes(title=dict(text="Time (s)", font=dict(size=20)), row=5, col=1)
 fig.update_yaxes(title=dict(text="V(t) (pixels/second)", font=dict(size=20)), row=5, col=1)

 # Row 6 — nose(tip) velocity
 nox, noy = np.array(Dico['nose(tip)']['x'], dtype=float), np.array(Dico['nose(tip)']['y'], dtype=float)
 abs_nose_vel = compute_velocity(nox, noy, fps_plot)
 fig.add_trace(go.Scatter(x=Dico['frames'], y=abs_nose_vel, mode='lines', name='Nose Tip Velocity'), row=6, col=1)
 fig.update_xaxes(title=dict(text="Time (s)", font=dict(size=20)), row=6, col=1)
 fig.update_yaxes(title=dict(text="V(t) (pixels/second)", font=dict(size=20)), row=6, col=1)

 fig.update_layout(title='Control plots of data along time (seconds)', showlegend=True)
 fig.update_xaxes(matches='x') #synchronizes the x-axe of all plots to allow comparisons
 output_file = str(name) + '_temporel.html'
 fig.write_html(output_file)




def control_image_2(Dico,name,video_path): 
 """
 Generate the spatial control figure — 2x2 grid showing keypoint distributions.
    
    Panel layout:
      - Top left: distribution of absolute velocity (histogram)
      - Top right: distribution of pupil area (histogram)
      - Bottom left: pupil center of mass trajectory inside the ROI with eye keypoints
      - Bottom right: all keypoints overlaid on first video frame (colored by bodypart)
    
    Uses first frame of the video as background for the spatial panel.
    Saved as *_FacemapPose_distri_et_spatial.png alongside the pkl.
 """
    # Load first frame of video as background for spatial keypoint plot
    # Released immediately after reading to free RAM
    #Collecting the first image fo the video
 print(video_path)
 cap = cv2.VideoCapture(video_path)
 ret, frame = cap.read()
 if not ret:
    raise ValueError("Impossible to read the video or the video is empty.")
#Converting the image from BGR (OpenCV) to RGB (Matplotlib)
 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 cap.release() #releases the .avi to spare RAM 


#Creating a 2x2 grid of subplots
 fig,axs=plt.subplots(2,2,figsize=(20,16))

#First image : Ploting the distribution of the velocities (x and y)
# Top left — velocity distribution histogram (x and y components separately)
# x-axis limited to [-1, 1] px/frame — velocities outside this range are typically artifacts
# Overlapping histograms (alpha=0.75) allow visual comparison of x vs y movement patterns
# Asymmetry between x and y distributions can indicate systematic drift or recording artifacts
 axs[0, 0].hist(Dico['pupil_velocity[vx,vy]'][:,0], bins=100, edgecolor='black', label='x', alpha=0.75)
 axs[0, 0].hist(Dico['pupil_velocity[vx,vy]'][:,1], bins=100, edgecolor='black', label='y', alpha=0.75)
 axs[0, 0].set_xlim(-1,1) #centers around 0
 axs[0, 0].set_title('Distribution of the absolute velocity')
 axs[0, 0].set_xlabel('V(t)')
 axs[0, 0].set_ylabel('prevalence')
 axs[0, 0].legend()
 axs[0, 0].grid(False)


#Second image : Ploting the distribution of the pupil area
# Top right — pupil area distribution histogram (px²)
# 50 bins — good balance between resolution and noise for typical session lengths
# Bimodal distribution may indicate eye closures or liquid contamination events
# Spike at low values = eye closed frames, spike at high values = saturation artifacts
# Commented set_ylim — uncomment if outliers compress the main distribution
 axs[0, 1].hist(Dico['pupil_area'], bins=50, edgecolor='black')
 #axs[0, 1].set_ylim(-X,X) #if need to center  
 axs[0, 1].set_title('Distribution of the area of the pupil')
 axs[0, 1].set_xlabel('A(t)')
 axs[0, 1].set_ylabel('prevalence')
 axs[0, 1].legend()
 axs[0, 1].grid(False)

#Third image : the trajectory of the pupil into the ROI and between the keypoints
 # Bottom left — pupil center of mass trajectory inside the ROI
 # Blue ellipse = ROI bounding box drawn as ellipse for visual reference
 # Colored dots = mean position of each eye keypoint (nanmean ignores NaN frames)
 # Blue line = full trajectory of pupil center of mass over the session
 # center_coord is in [y, x] format (FaceMap convention) — axes inverted to match image coordinates
 # Large excursions from center cluster indicate liquid contamination or eye closure events
 center_coord=Dico['pupil_center_[x,y]']
 xmin=Dico['x_range_ROI'][0]
 xmax=Dico['x_range_ROI'][-1]
 ymin=Dico['y_range_ROI'][0]
 ymax=Dico['y_range_ROI'][-1]
 center=((xmin+xmax)/2,(ymin+ymax)/2)
 width=xmax-xmin
 height=ymax-ymin
 ellipse = patches.Ellipse(center, width, height, fill=False, edgecolor='blue') #creates the ellipse
 axs[1, 0].add_patch(ellipse)
 eye_back_meanpos=[np.nanmean(Dico['eye(back)']['x']),np.nanmean(Dico['eye(back)']['y'])] #compute mean keypoint position ignoring NaN frames
 eye_front_meanpos=[np.nanmean(Dico['eye(front)']['x']),np.nanmean(Dico['eye(front)']['y'])]
 eye_top_meanpos=[np.nanmean(Dico['eye(top)']['x']),np.nanmean(Dico['eye(top)']['y'])]
 eye_bottom_meanpos=[np.nanmean(Dico['eye(bottom)']['x']),np.nanmean(Dico['eye(bottom)']['y'])]
 axs[1, 0].scatter(eye_front_meanpos[0],eye_front_meanpos[1],label='eye(front)')
 axs[1, 0].scatter(eye_back_meanpos[0],eye_back_meanpos[1],label='eye(back)')
 axs[1, 0].scatter(eye_top_meanpos[0],eye_top_meanpos[1],label='eye(top)')
 axs[1, 0].scatter(eye_bottom_meanpos[0],eye_bottom_meanpos[1],label='eye(bottom)')
 axs[1, 0].plot(center_coord[:,1]+xmin,center_coord[:,0]+ymin) 
 axs[1, 0].set_title('Movements of the center of the pupil inside the ROI, surrounded by the keypoints)')
 axs[1, 0].set_ylabel('y')
 axs[1, 0].set_xlabel('x')
 axs[1, 0].invert_yaxis() #invert the x et y axes because Facemap gives the coordinates of the center of the pupli as [y,x]
 axs[1, 0].legend(bbox_to_anchor=(1.005, 1), loc='upper left', borderaxespad=0.) #position of legend
 axs[1, 0].grid(False)

#Fourth image : trajectory of the keypoints over the image of the mouse
 # Bottom right — all keypoint positions over all frames overlaid on first video frame
 # s=1: small point size to show density rather than individual frames
 # markerscale=5 in legend to make markers visible despite s=1 in scatter
 # extent=[0, width, height, 0]: maps image to pixel coordinates (y-axis not inverted here because imshow already handles image orientation)
 # Dense clusters = stable keypoint detection, scattered points = noisy/unreliable tracking
 # Nose keypoints (excluded by EXCLUDED_KEYPOINTS) will show all frames as NaN = no visible points
 axs[1, 1].scatter(Dico['eye(front)']['x'],Dico['eye(front)']['y'],s=1,label='eye(front)')
 axs[1, 1].scatter(Dico['eye(back)']['x'],Dico['eye(back)']['y'],s=1,label='eye(back)')
 axs[1, 1].scatter(Dico['eye(top)']['x'],Dico['eye(top)']['y'],s=1,label='eye(top)')
 axs[1, 1].scatter(Dico['eye(bottom)']['x'],Dico['eye(bottom)']['y'],s=1,label='eye(bottom)')
 axs[1, 1].scatter(Dico['whisker(I)']['x'],Dico['whisker(I)']['y'],s=1,label='whisker(I)')
 axs[1, 1].scatter(Dico['whisker(II)']['x'],Dico['whisker(II)']['y'],s=1,label='whisker(II)')
 axs[1, 1].scatter(Dico['whisker(III)']['x'],Dico['whisker(III)']['y'],s=1,label='whisker(III)')
 axs[1, 1].scatter(Dico['nose(bottom)']['x'],Dico['nose(bottom)']['y'],s=1,label='nose(bottom)')
 axs[1, 1].scatter(Dico['nose(r)']['x'],Dico['nose(r)']['y'],s=1,label='nose(r)')
 axs[1, 1].scatter(Dico['nose(tip)']['x'],Dico['nose(tip)']['y'],s=1,label='nose(tip)')
 axs[1, 1].scatter(Dico['nosebridge']['x'],Dico['nosebridge']['y'],s=1,label='nosebridge')
 axs[1, 1].scatter(Dico['mouth']['x'],Dico['mouth']['y'],s=1,label='mouth')
 axs[1, 1].scatter(Dico['lowerlip']['x'],Dico['lowerlip']['y'],s=1,label='lowerlip')
 axs[1, 1].imshow(frame_rgb, extent=[0, frame.shape[1], frame.shape[0], 0]) #draws all keypoints on the first frame 
 axs[1, 1].legend(scatterpoints=1, markerscale=5,bbox_to_anchor=(1.005, 1), loc='upper left', borderaxespad=0.)
 axs[1, 1].set_title('Detected keypoints over frames, background first image of the video')
 axs[1, 1].set_xlabel('x')
 axs[1, 1].set_ylabel('y')
 axs[1, 1].grid(False)


# Save spatial distribution figure
# tight_layout() prevents overlap between subplots and legends
 plt.tight_layout()
 output_file=str(name)+'_distri_et_spatial.png'
 plt.savefig(output_file)
 plt.close() #free RAM — important for batch processing of many sessions
 
 # Commented plt.show() — uncomment for interactive inspection of a single session

# ── IMPORTANT: FaceMap pose model context ─────────────────────────────────────
# FaceMap uses a neural network trained on labeled images (similar to DeepLabCut).
# The model predicts ALL keypoint positions simultaneously using global image context.
# This means keypoints are NOT independent — the predicted position of nose(tip) is influenced by and influences nosebridge, nose(bottom), whiskers, etc.
# The model learned typical spatial relations between bodyparts during training.
# Consequence: if one keypoint is wrong, neighboring keypoints may also be affected.

def control_image_2_raw_data(Dico, name, video_path, h5_path):
    """
    Same 2x2 layout as control_image_2 but distinguishing valid vs filtered keypoints.
    
    Key difference from control_image_2:
      - Bottom right panel shows raw h5 keypoints in two colors:
          * Colored points: frames kept after likelihood filtering (valid)
          * Red points (alpha=0.3): frames removed by likelihood filter (NaN in Dico)
      - Bottom left panel: eye keypoint means computed from raw data (includes outliers)
      - Top panels: same as control_image_2 (velocity and area distributions)
    
    Purpose: visual quality control to assess likelihood filter performance per keypoint.
    Large red clusters indicate a keypoint with poor detection confidence.
    Saved as *_FacemapPose_raw_data_frames.png alongside the pkl.
    """
    # Raw h5 data loaded to show unfiltered positions
    # valid/nan split based on Dico (post-filter) applied to raw coordinates

    # np.mean on raw data for bottom-left panel — unlike control_image_2 which uses nanmean
    # This means outlier frames contribute to the mean position shown

    # Red legend entry added manually since scatter with color='red' shares the same label as filtered frames across all keypoints
    print(video_path)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Impossible to read the video.")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cap.release()

    # Extract raw data from .h5 to avoid filters
    raw_data = utils.load_keypoints(refine_pose.BODYPARTS, h5_path)
    raw_x = raw_data[0].astype(float)  # (n_keypoints, n_frames)
    raw_y = raw_data[1].astype(float)

    fig, axs = plt.subplots(2, 2, figsize=(20, 16))

    # Distribution of pupil velocity
    axs[0, 0].hist(Dico['pupil_velocity[vx,vy]'][:,0], bins=100, edgecolor='black', label='x', alpha=0.75)
    axs[0, 0].hist(Dico['pupil_velocity[vx,vy]'][:,1], bins=100, edgecolor='black', label='y', alpha=0.75)
    axs[0, 0].set_xlim(-1, 1)
    axs[0, 0].set_title('Distribution of the absolute velocity')
    axs[0, 0].set_xlabel('V(t)')
    axs[0, 0].set_ylabel('prevalence')
    axs[0, 0].legend()
    axs[0, 0].grid(False)

    # Distribution pupil area
    axs[0, 1].hist(Dico['pupil_area'], bins=50, edgecolor='black')
    axs[0, 1].set_title('Distribution of the area of the pupil')
    axs[0, 1].set_xlabel('A(t)')
    axs[0, 1].set_ylabel('prevalence')
    axs[0, 1].legend()
    axs[0, 1].grid(False)

    # Trajectory of the pupil
    center_coord = Dico['pupil_center_[x,y]']
    xmin = Dico['x_range_ROI'][0]
    xmax = Dico['x_range_ROI'][-1]
    ymin = Dico['y_range_ROI'][0]
    ymax = Dico['y_range_ROI'][-1]
    center = ((xmin+xmax)/2, (ymin+ymax)/2)
    width = xmax - xmin
    height = ymax - ymin
    ellipse = patches.Ellipse(center, width, height, fill=False, edgecolor='blue')
    axs[1, 0].add_patch(ellipse)
    bodyparts_list = [str(b) for b in refine_pose.BODYPARTS]
    for bp in ['eye(front)', 'eye(back)', 'eye(top)', 'eye(bottom)']:
        idx = bodyparts_list.index(bp)
        axs[1, 0].scatter(np.mean(raw_x[idx]), np.mean(raw_y[idx]), label=bp) #mean calculated on raw data = it'll take the outliers
    axs[1, 0].plot(center_coord[:,1]+xmin, center_coord[:,0]+ymin)
    axs[1, 0].set_title('Movements of the center of the pupil inside the ROI')
    axs[1, 0].set_ylabel('y')
    axs[1, 0].set_xlabel('x')
    axs[1, 0].invert_yaxis()
    axs[1, 0].legend(bbox_to_anchor=(1.005, 1), loc='upper left', borderaxespad=0.)
    axs[1, 0].grid(False)

    # Keypoints on the image s=1
    bodyparts = list(refine_pose.BODYPARTS)
    colors = plt.cm.tab20(np.linspace(0, 1, len(bodyparts)))

    for idx, bp in enumerate(bodyparts):
        if str(bp) not in Dico:
            continue
        x_filt = np.array(Dico[str(bp)]['x'], dtype=float)

        # Valid frames = non-NaN post filters
        valid = ~np.isnan(x_filt)
        x_valid = raw_x[idx][valid]
        y_valid = raw_y[idx][valid]
        if np.sum(valid) > 0:
            axs[1, 1].scatter(x_valid, y_valid, s=1, color=colors[idx], label=str(bp))

        # NaN frames = filtered by likelihood in red 
        nan_mask = np.isnan(x_filt)
        x_nan = raw_x[idx][nan_mask]
        y_nan = raw_y[idx][nan_mask]
        if np.sum(nan_mask) > 0:
            axs[1, 1].scatter(x_nan, y_nan, s=1, color='red', alpha=0.3)

    axs[1, 1].imshow(frame_rgb, extent=[0, frame.shape[1], frame.shape[0], 0])
    
    # Adds the red markers in the legend
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                               markersize=1, label='Filtered (NaN)')]
    handles, labels = axs[1, 1].get_legend_handles_labels()
    axs[1, 1].legend(handles + legend_elements, labels + ['Filtered (NaN)'],
                     scatterpoints=1, markerscale=5,
                     bbox_to_anchor=(1.005, 1), loc='upper left', borderaxespad=0.)
    axs[1, 1].set_title('Valid keypoints (colored) vs Filtered frames (red)')
    axs[1, 1].set_xlabel('x')
    axs[1, 1].set_ylabel('y')
    axs[1, 1].grid(False)

    plt.tight_layout()
    output_file = str(name) + '_raw_data_frames.png'
    plt.savefig(output_file)
    plt.close()
    print(f'Saved: {output_file}')


def control_image_2_HTML(Dico,name,video_path):
 """
    Interactive Plotly version of control_image_2 — same 2x2 layout, zoomable in browser.
    
    Panel layout (same as control_image_2 but panels rearranged):
      - Top left: all keypoints overlaid on first video frame (colored by bodypart, marker size=2)
      - Top right: pupil area histogram
      - Bottom left: pupil center trajectory inside ROI with ellipse and eye keypoints
      - Bottom right: velocity distribution (x and y components)
    
    Key differences from control_image_2:
      - Ellipse drawn manually with parametric equations (Plotly has no native Ellipse patch)
      - Image added as background layer via fig.update_layout(images=[...])
      - y-axis reversed via autorange='reversed' (FaceMap [y,x] convention)
    
    Comment 'Directly comment the ones you don't want to show' — useful for debugging single keypoints.
    Saved as *_FacemapPose_distri_et_spatial.html alongside the pkl.
 """
#Collecting the first image fo the video
 print(video_path)
 cap = cv2.VideoCapture(video_path)
 ret, frame = cap.read()
 if not ret:
    raise ValueError("Impossible to read the video or the video is empty.")
    
#Converting the image from BGR (OpenCV) to RGB (Matplotlib)
 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 cap.release()
 image = Image.fromarray(frame_rgb)

#Creating a 2x2 grid of subplots
 fig = make_subplots(rows=2, cols=2)
 fig.update_layout(colorway=px.colors.qualitative.Light24)


#Fourth image : trajectory of the keypoints over the image of the mouse
 #Adding scatter traces for each keypoint
#Directly comment the ones you don't want to show
 fig.add_trace(go.Scatter(x=Dico['eye(front)']['x'], y=Dico['eye(front)']['y'], mode='markers', name='eye(front)', marker=dict(size=2)), row=1, col=1)
 fig.add_trace(go.Scatter(x=Dico['eye(back)']['x'], y=Dico['eye(back)']['y'], mode='markers', name='eye(back)', marker=dict(size=2)), row=1, col=1)
 fig.add_trace(go.Scatter(x=Dico['eye(top)']['x'], y=Dico['eye(top)']['y'], mode='markers', name='eye(top)', marker=dict(size=2)), row=1, col=1)
 fig.add_trace(go.Scatter(x=Dico['eye(bottom)']['x'], y=Dico['eye(bottom)']['y'], mode='markers', name='eye(bottom)', marker=dict(size=2)), row=1, col=1)
 fig.add_trace(go.Scatter(x=Dico['whisker(I)']['x'], y=Dico['whisker(I)']['y'], mode='markers', name='whisker(I)', marker=dict(size=2)), row=1, col=1)
 fig.add_trace(go.Scatter(x=Dico['whisker(II)']['x'], y=Dico['whisker(II)']['y'], mode='markers', name='whisker(II)', marker=dict(size=2)), row=1, col=1)
 fig.add_trace(go.Scatter(x=Dico['whisker(III)']['x'], y=Dico['whisker(III)']['y'], mode='markers', name='whisker(III)', marker=dict(size=2)), row=1, col=1)
 fig.add_trace(go.Scatter(x=Dico['nose(bottom)']['x'], y=Dico['nose(bottom)']['y'], mode='markers', name='nose(bottom)', marker=dict(size=2)), row=1, col=1)
 fig.add_trace(go.Scatter(x=Dico['nose(r)']['x'], y=Dico['nose(r)']['y'], mode='markers', name='nose(r)', marker=dict(size=2)), row=1, col=1)
 fig.add_trace(go.Scatter(x=Dico['nose(tip)']['x'], y=Dico['nose(tip)']['y'], mode='markers', name='nose(tip)', marker=dict(size=2)), row=1, col=1)
 fig.add_trace(go.Scatter(x=Dico['nosebridge']['x'], y=Dico['nosebridge']['y'], mode='markers', name='nosebridge', marker=dict(size=2)), row=1, col=1)
 fig.add_trace(go.Scatter(x=Dico['mouth']['x'], y=Dico['mouth']['y'], mode='markers', name='mouth', marker=dict(size=2)), row=1, col=1)
 fig.add_trace(go.Scatter(x=Dico['lowerlip']['x'], y=Dico['lowerlip']['y'], mode='markers', name='lowerlip', marker=dict(size=2)), row=1, col=1)
 
 
 # Add the image as a background : s=1
 fig.update_layout(
        images=[dict(
            source=image,
            xref="x",
            yref="y",
            x=0,
            y=frame_rgb.shape[0],
            sizex=frame_rgb.shape[1],
            sizey=frame_rgb.shape[0],
            sizing="stretch", #stretch image to fit axes
            xanchor="left",
            yanchor="bottom",
            layer="below")]) #place image behind keypoints
 
    # Set the axis ranges to match the image dimensions
    #fig.update_xaxes(range=[0, frame_rgb.shape[1]])
 fig.update_xaxes(range=[0,frame_rgb.shape[1]],row=1, col=1)
 fig.update_yaxes(range=[frame_rgb.shape[0],0],row=1, col=1)
 fig.update_xaxes(title=dict(text="x",font=dict(size=20)), row=1, col=1)
 fig.update_yaxes(title=dict(text="y",font=dict(size=20)), row=1, col=1)
 
    # Update layout for the figure
 fig.update_layout(title_text='Controls of the detection', legend=dict(itemsizing='constant', itemwidth=30))

#Second image
 fig.add_trace(go.Histogram(x=Dico['pupil_area'],name='Pupil Area',nbinsx=100,
    marker=dict(color='blue', line=dict(color='black', width=1)),opacity=0.75),row=1,col=2)
 fig.update_layout(
    bargap=0.1, # gap between bars of adjacent location coordinates
    bargroupgap=0.1, # gap between bars of the same location coordinate
    showlegend=True)
 fig.update_xaxes(title=dict(text="t (seconds)",font=dict(size=20)), row=1, col=2)
 fig.update_yaxes(title=dict(text="Area of the pupil(t)",font=dict(size=20)), row=1, col=2)
  
#Third image : the trajectory of the pupil into the ROI and between the keypoints
 center_coord=Dico['pupil_center_[x,y]']
 xmin=Dico['x_range_ROI'][0]
 xmax=Dico['x_range_ROI'][-1]
 ymin=Dico['y_range_ROI'][0]
 ymax=Dico['y_range_ROI'][-1]
 center=((xmin+xmax)/2,(ymin+ymax)/2)
 width=xmax-xmin
 height=ymax-ymin
 eye_back_meanpos=[np.nanmean(Dico['eye(back)']['x']),np.nanmean(Dico['eye(back)']['y'])]
 eye_front_meanpos=[np.nanmean(Dico['eye(front)']['x']),np.nanmean(Dico['eye(front)']['y'])]
 eye_top_meanpos=[np.nanmean(Dico['eye(top)']['x']),np.nanmean(Dico['eye(top)']['y'])]
 eye_bottom_meanpos=[np.nanmean(Dico['eye(bottom)']['x']),np.nanmean(Dico['eye(bottom)']['y'])]
 theta = np.linspace(0, 2 * np.pi, 100) #draws manually the ellipse
 x_ellipse = center[0] + width / 2 * np.cos(theta)
 y_ellipse = center[1] + height / 2 * np.sin(theta)
 fig.add_trace(go.Scatter(x=x_ellipse, y=y_ellipse, mode='lines', line=dict(color='blue'), showlegend=False), row=2, col=1)
 
 # Add scatter points for eye positions
 fig.add_trace(go.Scatter(x=[eye_front_meanpos[0]], y=[eye_front_meanpos[1]], mode='markers', name='eye(front)'), row=2, col=1)
 fig.add_trace(go.Scatter(x=[eye_back_meanpos[0]], y=[eye_back_meanpos[1]], mode='markers', name='eye(back)'), row=2, col=1)
 fig.add_trace(go.Scatter(x=[eye_top_meanpos[0]], y=[eye_top_meanpos[1]], mode='markers', name='eye(top)'), row=2, col=1)
 fig.add_trace(go.Scatter(x=[eye_bottom_meanpos[0]], y=[eye_bottom_meanpos[1]], mode='markers', name='eye(bottom)'), row=2, col=1)
 
 # Add trajectory line
 fig.add_trace(go.Scatter(x=center_coord[:, 1] + xmin, y=center_coord[:, 0] + ymin, mode='lines', name='Pupil Trajectory'), row=2, col=1)
 
 # Update layout for the subplot
 fig.update_yaxes(autorange="reversed", row=2, col=1) #invert y-axis because FaceMap returns [y, x] coordinates 
 fig.update_xaxes(title=dict(text="x",font=dict(size=20)), row=2, col=1)
 fig.update_yaxes(title=dict(text="y",font=dict(size=20)), row=2, col=1)
 
 # Set the background color to light black for the subplot at row=2, col=1
 fig.update_layout(
    plot_bgcolor='rgba(50, 50, 50, 0.3)',  # This sets the background color for the entire figure
    xaxis2=dict(showgrid=True, gridcolor='rgba(200, 200, 200, 0.5)'),
    yaxis2=dict(showgrid=True, gridcolor='rgba(200, 200, 200, 0.5)')
 )

 #Ploting the distribution of the velocities (x and y)

 fig.add_trace(go.Histogram(x=Dico['pupil_velocity[vx,vy]'][:, 0],name='x Velocity',nbinsx=100,
    marker=dict(color='blue', line=dict(color='black', width=1)),opacity=0.75),row=2,col=2)

 fig.add_trace(go.Histogram(x=Dico['pupil_velocity[vx,vy]'][:, 1],name='y Velocity', nbinsx=100,
    marker=dict(color='green', line=dict(color='black', width=1)),opacity=0.75),row=2,col=2)


 fig.update_layout(
    bargap=0.1, # gap between bars of adjacent location coordinates
    bargroupgap=0.1, # gap between bars of the same location coordinate
    showlegend=True)
 fig.update_xaxes(title=dict(text="t (seconds)",font=dict(size=20)), row=2, col=2)
 fig.update_yaxes(title=dict(text="Velocity of the pupil(t)",font=dict(size=20)), row=2, col=2)

#Save the entire figure as a HTML file
 output_file=str(name)+'_distri_et_spatial.html'
 fig.write_html(output_file)




def control_image_outliers(Dico, name, h5_path):
    """
    Detect and display outlier frames as a summary table.
    
    Outliers detected on RAW h5 data (before likelihood filter) to show what the filter removes.
    Outlier criterion: frame-to-frame velocity exceeds a threshold (computed per keypoint).
    
    Output: PNG table showing for each keypoint:
      - Number of outlier frames detected
      - % of total frames
      - Timestamps of first few outlier frames
    
    Useful for verifying that the likelihood filter correctly removes high-velocity artifacts and for identifying keypoints with systematic tracking issues.
    Saved as *_FacemapPose_outliers_table.png alongside the pkl.
    """

    raw_data = utils.load_keypoints(refine_pose.BODYPARTS, h5_path)
    raw_x = raw_data[0].astype(float)
    raw_y = raw_data[1].astype(float)
    bodyparts_list = [str(b) for b in refine_pose.BODYPARTS]

    if 'fps' not in Dico:
        raise KeyError("FPS not found in Dico")
    fps = Dico['fps']

    def frame_to_hms(frame, fps):
        """
        Convert a frame index to a human-readable timestamp string (HH:MM:SS).
        Used in the outliers table to display when outlier frames occur in the recording.
        Example: frame 5400 at 30fps → '00h03m00s'
        """
        total_s = frame / fps
        h = int(total_s // 3600)
        m = int((total_s % 3600) // 60)
        s = int(total_s % 60)
        return f"{h:02d}h{m:02d}m{s:02d}s"

    def find_whisker_outliers(vel, label, fps, threshold_px_s=2000): 
        # This threshold is to AJUST in order to find the frames where the mouse scratchs itself or tries to by moving a lot 
        """
        Find frames where whisker velocity exceeds a fixed threshold (px/s).
    
        Threshold of 2000 px/s detects sustained high-velocity movements (grooming, scratching)
        that would not be caught by the derivative threshold (which detects sudden spikes).
    
        Spacing of 30s minimum between selected outliers avoids reporting the same grooming bout multiple times — only the peak velocity frame per bout is reported.
    
        Returns list of rows [keypoint_label, time_s, time_hms, velocity] for the outlier table.
        """ 
        vel = np.array(vel, dtype=float)
        valid_mask = ~np.isnan(vel)
        if not np.any(valid_mask):
            return []
        outlier_idx = np.where((vel > threshold_px_s) & valid_mask)[0]
        if len(outlier_idx) == 0:
            return []
        # Sort by velocity descending, apply spacing
        sorted_idx = outlier_idx[np.argsort(vel[outlier_idx])[::-1]]
        min_spacing = int(30 * fps)
        selected = []
        for idx in sorted_idx:
            if all(abs(int(idx) - int(s)) >= min_spacing for s in selected):
                selected.append(idx)
        selected = sorted(selected)
        rows = []
        for idx in selected:
            rows.append([label, f"{idx/fps:.2f}s", frame_to_hms(idx, fps), f"{vel[idx]:.2f}"])
        return rows

    def find_outliers(signal, label, fps):
        """
        Find the 30 worst frames per signal based on frame-to-frame derivative (sudden changes).
    
        Derivative criterion detects abrupt changes (sudden spikes) in the signal.
        Complementary to find_whisker_outliers() which detects sustained high velocity.
    
        30s minimum spacing between selected frames avoids reporting consecutive frames of the same event — each grooming bout contributes at most one entry.
        Limit of 30 outliers per signal keeps the table readable.
    
        Returns list of rows [signal_label, time_s, time_hms, derivative_value] for outlier table.
        """
        signal = np.array(signal, dtype=float)
        deriv = np.abs(np.diff(signal, prepend=signal[0]))
        deriv[0] = np.nan
        valid = deriv[~np.isnan(deriv)]
        if len(valid) == 0:
            return []
        # Sort all frames by derivative value descending
        all_idx = np.argsort(deriv)[::-1]
        min_spacing = int(30 * fps)  # 30 seconds minimum between outliers because the scrachting lasts approx 30 seconds
        selected = []
        for idx in all_idx:
            if np.isnan(deriv[idx]):
                continue
            if all(abs(int(idx) - int(s)) >= min_spacing for s in selected):
                selected.append(idx)
            if len(selected) >= 30:
                break
        selected = sorted(selected)
        rows = []
        for idx in selected:
            rows.append([label, f"{idx/fps:.2f}s", frame_to_hms(idx, fps), f"{deriv[idx]:.2f}"])
        return rows

    all_rows = [] # Accumulate outlier rows across all signals

    # Pupil velocity — from filtered Dico (velocity NaN frames already removed)
    # Using Dico instead of raw proc.npy avoids false outliers from liquid contamination frames
    # To use truly raw data: reload proc.npy via load_and_process_npy() and pass absolute_velocity like this : raw_npy = load_and_process_npy(file_path_npy)
    
    pup_vel = np.array(Dico['absolute_velocity'], dtype=float) #Outliers from de dico WITHOUT the liquid frame because those movement are for sure not changes in the pupil that could be misinterpreted because the liquid is everywhere on the eye when it's filtered
    all_rows += find_outliers(pup_vel, 'Pupil velocity', fps)

    # Pupil area (raw) - same rationale as pupil velocity (filtered Dico)

    pup_area = np.array(Dico['pupil_area'], dtype=float) #same than the pupil velocity : Dico['pupil_center_[x,y]']
    # To use truly raw data, pass file_path_npy as argument and reload _proc.npy
    all_rows += find_outliers(pup_area, 'Pupil area', fps)

    # Whisker I/II/III — velocity computed from raw h5 keypoints (no likelihood filter applied)
    # Uses find_whisker_outliers() with fixed threshold (2000 px/s) instead of derivative
    # threshold_px_s=2000: adjust if whisker movements in your recordings are systematically faster/slower
   
    for wh_name, wh_str in [('Whisker I', 'whisker(I)'), ('Whisker II', 'whisker(II)'), ('Whisker III', 'whisker(III)')]:
        if wh_str in bodyparts_list:
            idx = bodyparts_list.index(wh_str)
            vel = compute_velocity(raw_x[idx], raw_y[idx], fps)  # px/s using real fps
            all_rows += find_whisker_outliers(vel, wh_name + ' velocity', fps, threshold_px_s=2000)

     # Nose tip excluded — respiratory apparatus permanently placed on nose causes systematic artifacts

    print(f'Total outliers found: {len(all_rows)}') #allow to compare with the outliers that we can see with bare eyes


    # Build outlier table figure
    # Figure height adapts to number of outliers (0.3 per row + header)
    # ax.axis('off') hides axes — only the table is shown
    col_labels = ['Signal', 'Time (s)', 'Time (h:m:s)', 'Derivative']
    n_rows = max(len(all_rows), 1) #minimum 1 row if no outliers are found

    fig, ax = plt.subplots(figsize=(16, max(4, n_rows * 0.3 + 1.5))) #the size of the figure adapts with the numbers of outliers 
    ax.axis('off')

    if len(all_rows) == 0:
        all_rows = [['No outliers detected', '', '', '', '']]

    table = ax.table(
        cellText=all_rows,
        colLabels=col_labels,
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False) # disable auto font size — use fixed 8pt
    table.set_fontsize(8) # writing size at 8 
    table.auto_set_column_width(col=list(range(len(col_labels))))  # adjust column widths to content

    #  # Dark header row
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Alternating row colors per signal type for readability
    # Each signal has a unique color — rows from the same signal share the same background
    colors = {'Pupil velocity': '#d6eaf8', 'Pupil area': '#d5f5e3',
              'Whisker I velocity': '#fdebd0', 'Whisker II velocity': '#f9ebea',
              'Whisker III velocity': '#f4ecf7', 'Nose tip velocity': '#fdfefe'}
    for i, row in enumerate(all_rows):
        color = colors.get(row[0], '#ffffff')
        for j in range(len(col_labels)):
            table[i+1, j].set_facecolor(color)
    
    # Title includes session name for identification when viewing saved PNG
    ax.set_title(f'Outliers table (raw data) — 99.95 percentile | {name.split("/")[-1]}', 
                 fontsize=10, pad=10)

    output_file = str(name) + '_outliers_table.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_file}')


#Extracting core function


def extract(file_path_npy,file_path_h5,file_path_video):
      """Core extraction function — builds the complete Dico from npy, h5 and video files.
    
        Assembles all signals into a single dictionary (Dicotemp) that is saved as the final pkl.
    
        Data sources:
            - proc.npy: pupil area, center of mass, velocity, SVD motion components, ROI coordinates
            - h5: keypoint coordinates (x, y, likelihood) for all bodyparts — filtered by likelihood threshold
            - video: fps and first frame (for spatial figures)
    
        Velocity computation:
            - pupil_velocity[vx,vy]: frame-to-frame displacement (px/frame) — prepend [0,0] offset for frame 0
            - absolute_velocity: sqrt(vx² + vy²) — frame 0 set to NaN (no previous frame)
    
        Dicotemp.update(Dicoh5): merges h5 keypoints into Dico — overwrites any existing keys with same name.
        FPS read from video file (not hardcoded) to handle different recording setups.
      """
      Dicotemp={}
      NPY=load_and_process_npy(file_path_npy) #initialize from the _proc.npy data - pupil area, center of mass, SVD, ROI
      Dicotemp['pupil_area']=NPY[0]
      com=NPY[1] #com = center of mass
      #print(com) - debug line
      Dicotemp['pupil_center_[x,y]']=com
      offset=np.array([[0,0]]) # Velocity: offset by one frame (prepend [0,0]) then set frame 0 to NaN
      #same logic, we create a -1 frame with x(-1)=y(-1)=0 and then set to NaN the velocities : vx(0)=vy(0)=NaN
      compop=com[:-1]
      comdelayed=np.concatenate((offset,compop))
      Dicotemp['pupil_velocity[vx,vy]']=com-comdelayed   #velocity normalized by 1 (delta t = 1)
      Dicotemp['absolute_velocity']=np.sqrt(Dicotemp['pupil_velocity[vx,vy]'][:, 0]**2 + Dicotemp['pupil_velocity[vx,vy]'][:, 1]**2)
      Dicotemp['absolute_velocity'][0] = np.nan
      Dicotemp['movSVD']=NPY[2]
      Dicotemp['x_range_ROI']=NPY[3]
      Dicotemp['y_range_ROI']=NPY[4]
      Dicoh5=load_and_process_h5(file_path_h5)  # filtered keypoints from h5
      Dicotemp.update(Dicoh5) #updates the data into filtered datas
      
      # First frame loaded in the GUI + calculating the FPS for the frame to seconds conversion
      # FPS read from video metadata (not hardcoded) to support different recording setups
      # Prefixed with _ to avoid name conflicts with outer scope variables
      # Raises ValueError if video is unreadable or FPS is invalid (0 or negative)
      # fps stored in Dico so all downstream functions use the same value
      _cap = _cv2.VideoCapture(file_path_video)
      _ret, _frame = _cap.read()
      _fps = _cap.get(_cv2.CAP_PROP_FPS)
      _cap.release()
      if not _ret:
          raise ValueError(f"Impossible to read the first frame of {file_path_video} — corrupted video or incorrect pathway.")
      if _fps <= 0:
          raise ValueError(f"FPS non detected for {file_path_video} — Verify the video file.")
      fps = _fps
      Dicotemp['fps'] = fps
      print(f'FPS detected: {fps}')


  # ── Calibration — respiratory apparatus ───────────────────────────────────
      # Loads px/mm calibration from JSON files created by AnalyseFaceMap.py batch pre-pass.
      # Multiple JSON files supported (one per terminal) — all merged into _cal_data dict.
      # Key = absolute video path, value = {px_per_mm, dist_px, diam_mm}.
      # If calibration not found: px_per_mm_calibration = None → ComparaisonFaceMap.py will use eye keypoints fallback to estimate px/mm
      # if True / if False: pass — legacy structure from refactoring, harmless.

      # Commented GUI block below — uncomment ONLY when testing a single video directly (not needed for overnight batch processing via AnalyseFaceMap.py).

      import json as _json
      # Search all calibration JSON files in /tmp/ (one per batch terminal)
      import glob as _glob
      _cal_files = sorted(_glob.glob('/tmp/facemap_calibrations_*.json'))
      _cal_loaded = False
      _cal_data = {}
      for _cal_file in _cal_files:
          try:
              with open(_cal_file, 'r') as _f:
                  _cal_data.update(_json.load(_f))
          except:
              pass
      if True:
          try:
              if file_path_video in _cal_data:
                  _cal = _cal_data[file_path_video]
                  Dicotemp['px_per_mm_calibration'] = _cal['px_per_mm']
                  Dicotemp['calibration_dist_px'] = _cal['dist_px']
                  Dicotemp['calibration_diam_mm'] = _cal['diam_mm']
                  print(f'\n=== Calibration (from pre-pass) ===')
                  print(f'   ✅ {_cal["dist_px"]:.1f} px = {_cal["diam_mm"]} mm → {_cal["px_per_mm"]:.2f} px/mm')
                  _cal_loaded = True
          except Exception as _e:
              print(f'   ⚠️  Could not load calibration file: {_e}')
      if False: pass
      if not _cal_loaded:
          print('   ⚠️  No calibration found — eye keypoints fallback will be used in ComparaisonFaceMap.')
          Dicotemp['px_per_mm_calibration'] = None
          Dicotemp['calibration_dist_px'] = None
          Dicotemp['calibration_diam_mm'] = None
      
      
      # ── Likelihood NaN frames ────────────────────────────────────
      # Builds a boolean mask of frames flagged by the likelihood filter.
      # EXCLUDED_KEYPOINTS (nose keypoints) skipped — they are 100% NaN by design (not grooming).
      # nan_grooming_likelihood stored as frame indices list in Dico for figure marker display.
      # nan_grooming_likelihood = frames filtered by likelihood threshold only (no velocity filter)
      
      # Current filter: proxy using whisker keypoints only 
      # Actual VF1 pkl are all processed using this filter
      nan_grooming_likelihood = np.zeros(len(Dicotemp['pupil_area']), dtype=bool)
      for kp in ['nose(tip)', 'whisker(I)', 'whisker(II)', 'whisker(III)']:
          if kp in Dicotemp and kp not in EXCLUDED_KEYPOINTS:
              nan_grooming_likelihood |= np.isnan(np.array(Dicotemp[kp]['x'], dtype=float))
              
              
      # ── Experimental: full keypoint scan — TO TEST before activating ──────────
      # Tests showed this catches up to +50% more frames vs proxy (mouth, paw, eye keypoints)
      # but needs validation: are these extra frames truly problematic or false positives?
      # Use AnalyseFacemap on all videos and then post-process them using the following code to test if the likelihood filter on all keypoints helps or if it's too agressive 
      # I tested here : /media/nas8-2/ProjectCardioSense/K1690/2025-03-04_09-48-02/1690_250304_Injection_Saline/SLEEP-Mouse-1690-04032025-Sleep_01/Analyse pupille 
      # It doesn't seem to help 
      
      # Uncomment below and comment above to test:
      # nan_grooming_likelihood = np.zeros(len(Dicotemp['pupil_area']), dtype=bool)
      # for kp in Dicotemp:
      #     if kp not in EXCLUDED_KEYPOINTS and isinstance(Dicotemp[kp], dict) and 'x' in Dicotemp[kp]:
      #         nan_grooming_likelihood |= np.isnan(np.array(Dicotemp[kp]['x'], dtype=float))
      print(f'Likelihood filtered frames: {np.sum(nan_grooming_likelihood)} ({100*np.mean(nan_grooming_likelihood):.1f}%)')


      # ── Liquid frames threshold (ROI luminosity) ───────────────────────────────
     # Detects frames where liquid covers the eye by measuring mean luminosity inside the pupil ROI.
     # Only the pupil ROI region is analyzed — liquid appears there first and most clearly.
     # xmin/xmax/ymin/ymax: pixel boundaries of the pupil ROI extracted from proc.npy.
     
      xmin_roi = int(Dicotemp['x_range_ROI'][0]) #uses only the coordinate in the pupil ROI in each frame because the liquid does not appear elsewhere
      xmax_roi = int(Dicotemp['x_range_ROI'][-1])
      ymin_roi = int(Dicotemp['y_range_ROI'][0])
      ymax_roi = int(Dicotemp['y_range_ROI'][-1])
 
      # Corneal reflector masking:
      # If a reflector ROI was placed in FaceMap, its pixel coordinates are loaded from proc.npy.
      # The reflector zone is excluded from luminosity calculation — it is always bright
      # and would otherwise bias the mean luminosity upward, causing false liquid detections.
      # reflector_yrange/xrange = None if no reflector was placed → full ROI used for luminosity.
      
      raw_npy = np.load(file_path_npy, allow_pickle=True)
      raw_npy = raw_npy.item() if isinstance(raw_npy, np.ndarray) and raw_npy.shape == () else raw_npy #extracting the coordinates of the corneal reflector
      reflector_yrange = None #initalisation on None = if no reflector is found, calculates normally the luminosity
      reflector_xrange = None
      if 'rois' in raw_npy and len(raw_npy['rois']) > 0:
          roi0 = raw_npy['rois'][0] #extracts the number 0 ROI from FaceMap = pupil
          if 'reflector' in roi0 and len(roi0['reflector']) > 0: #Verifies the presence and the content of the reflector list 
              reflector_yrange = roi0['reflector'][0]['yrange'] #Extracts the coordinate of the reflector in the pupile ROI : the reflector is immobile
              reflector_xrange = roi0['reflector'][0]['xrange']
              print(f'Corneal reflector detected — masking it from luminosity calculation') #Manual verification possible

      def compute_roi_luminosity(roi_patch, reflector_yrange, reflector_xrange):
          """
          Compute mean luminosity of the pupil ROI patch, masking the corneal reflector if present. 
          roi_patch: 2D grayscale array cropped to the pupil ROI bounding box.
          Reflector masking: sets reflector pixels to NaN before computing nanmean.
          valid_y/valid_x clipped to patch size — protects against IndexError when reflector is positioned at the edge of the ROI and partially outside it.
          Returns NaN if all pixels are masked.
        """
          patch = roi_patch.astype(float)
          if reflector_yrange is not None and reflector_xrange is not None:
              h, w = patch.shape
              valid_y = reflector_yrange[reflector_yrange < h] #allows to protect from crashing when the reflector goes out from the pupile ROI because it's on the edge
              valid_x = reflector_xrange[reflector_xrange < w]
              if len(valid_y) > 0 and len(valid_x) > 0:
                  patch[np.ix_(valid_y, valid_x)] = np.nan #a rectangular mask is laid on the coordinates of the reflector and all the luminosity points are set to NaN (clipped to patch size to avoid IndexError)
          return np.nanmean(patch)
      
        # Open video for frame-by-frame luminosity extraction
        # lum_per_frame: array of mean ROI luminosity per frame — used for liquid detection threshold
      _cap2 = cv2.VideoCapture(file_path_video)
      total_frames = len(Dicotemp['pupil_area'])
      lum_per_frame = np.full(total_frames, np.nan)

     # Sample 250 frames evenly spaced across the video to estimate baseline luminosity
      # 250 frames is enough for a robust MAD estimate without loading the full video
      # np.linspace ensures even coverage — total_frames-1 avoids out-of-bounds indexing
      sample_idx = np.linspace(0, total_frames-1, min(250, total_frames), dtype=int) #Echantillonne avec 1 frame tous les 500 et total_frames évite de sortir des cadres de la vidéo quand elle est courte
      lum_sample = []
      for idx in sample_idx:
          _cap2.set(cv2.CAP_PROP_POS_FRAMES, idx)
          _ret2, _frame2 = _cap2.read()
          if _ret2:
              gray2 = cv2.cvtColor(_frame2, cv2.COLOR_BGR2GRAY)
              roi_patch = gray2[ymin_roi:ymax_roi, xmin_roi:xmax_roi]
              lum_sample.append(compute_roi_luminosity(roi_patch, reflector_yrange, reflector_xrange))
      _cap2.release()
      
      # MAD-based threshold: robust to outliers (liquid frames in the sample)
      # lum_threshold = median + 5 * MAD * 1.4826
      # 1.4826 = consistency factor converting MAD to equivalent standard deviation (normal distribution)
      # +5 std: conservative threshold calibrated on videos without liquid (K1690/K1711/K1712)
      lum_sample = np.array(lum_sample)
      lum_median = np.median(lum_sample)
      lum_mad = np.median(np.abs(lum_sample - lum_median))
      lum_threshold = lum_median + 5 * lum_mad * 1.4826 #+5 standard deviation, calibrated on videos without liquid
      Dicotemp['lum_threshold'] = lum_threshold # stored in Dico for traceability


      # ── Biconditional liquid detection ────────────────────────────────────────
      # Two complementary criteria to detect liquid contamination frames:
      #   Criterion 1: mean ROI luminosity > lum_threshold (bright ROI = liquid present)
      #   Criterion 2: % pixels above saturation threshold > pct_white_threshold (white pixels = liquid)
      # OR logic: a frame is marked as liquid if EITHER criterion is triggered
      # Using two criteria reduces false negatives — liquid can appear bright without saturating pixels
      # and vice versa depending on lighting conditions and liquid opacity.

      # Dynamic thresholds: both computed from 250 sampled frames using MAD (same sample_idx as luminosity sampling above — video not reopened for efficiency)
      # PCT_WHITE_N_MAD = 5: — increase to be less sensitive, decrease for more sensitivity
      # 1.4826 consistency factor: converts MAD to equivalent standard deviation

      # saturation_threshold: pixel value above which a pixel is considered 'white'
      # read from FaceMap ROI settings (user-defined in FaceMap GUI)
      # roi0_total_px: total pixels in ROI — used to normalize white pixel count to percentage

      # Frame-by-frame loop reads video sequentially (faster than random access)
      # lum_per_frame and pct_white_per_frame stored in Dico for PostProcessingGUI display
      # crit_lum will be dilated below — crit_white applied immediately (no dilation needed)

      # Get saturation threshold from ROI
      raw_npy2 = load_and_process_npy(file_path_npy)  # already loaded but needed for saturation
      saturation_threshold = roi0.get('saturation', 255.0) # default 255 = no saturation filter
      roi0_total_px = (ymax_roi - ymin_roi) * (xmax_roi - xmin_roi) # total pixels in pupil ROI

      # Compute dynamic % white pixels threshold from sample frames (reopen video)
      _cap_white = cv2.VideoCapture(file_path_video)
      pct_white_sample = []
      for idx in sample_idx:
          _cap_white.set(cv2.CAP_PROP_POS_FRAMES, idx)
          _ret_w, _frame_w = _cap_white.read()
          if _ret_w:
              gray_w = cv2.cvtColor(_frame_w, cv2.COLOR_BGR2GRAY)
              roi_patch_w = gray_w[ymin_roi:ymax_roi, xmin_roi:xmax_roi]
              n_white = np.sum(roi_patch_w > saturation_threshold)
              pct_white_sample.append(100 * n_white / roi0_total_px if roi0_total_px > 0 else 0)
      _cap_white.release()
      pct_white_sample = np.array(pct_white_sample)
      pct_white_median = np.median(pct_white_sample)
      pct_white_mad = np.median(np.abs(pct_white_sample - pct_white_median))
      PCT_WHITE_N_MAD = 5  # number of MAD above median to trigger liquid detection
      pct_white_threshold = pct_white_median + PCT_WHITE_N_MAD * pct_white_mad * 1.4826
      Dicotemp['pct_white_threshold'] = pct_white_threshold
      Dicotemp['pct_white_median'] = pct_white_median
      print(f'% white pixels threshold: {pct_white_threshold:.1f}% (median={pct_white_median:.1f}%)')

      _cap3 = cv2.VideoCapture(file_path_video)
      nan_liquid = np.zeros(total_frames, dtype=bool)
      pct_white_per_frame = np.full(total_frames, np.nan)
      i = 0
      while i < total_frames:
          _ret3, _frame3 = _cap3.read()
          if not _ret3:
              break
          gray3 = cv2.cvtColor(_frame3, cv2.COLOR_BGR2GRAY)
          roi_patch = gray3[ymin_roi:ymax_roi, xmin_roi:xmax_roi]
          lum_per_frame[i] = compute_roi_luminosity(roi_patch, reflector_yrange, reflector_xrange)
          n_white = np.sum(roi_patch > saturation_threshold)
          pct_white_per_frame[i] = 100 * n_white / roi0_total_px if roi0_total_px > 0 else 0
          # Frame is liquid if ANY criterion is true (OR logic)
          crit_lum = lum_per_frame[i] > lum_threshold
          crit_white = pct_white_per_frame[i] > pct_white_threshold
          # crit_white applied directly (no dilation)
          # crit_lum will be dilated separately below
          # crit_vel (velocity) applied after loop on full array
          if crit_white:
              nan_liquid[i] = True
          if crit_lum:
              nan_liquid[i] = True  # also set here, will be re-merged after dilation
      Dicotemp['roi0_total_px'] = roi0_total_px
      Dicotemp['lum_per_frame'] = lum_per_frame

      # Criterion 3: pupil center velocity > 800 px/s — applied on full array after loop
      #Active NaN filter: velocity > 800 px/s ONLY
      # Luminosity and white pixels → attention markers (informative, NOT filtered)
      VELOCITY_THRESHOLD_PX_S = 800  # 
      vel_arr = np.array(Dicotemp['absolute_velocity'], dtype=float) * fps
      nan_liquid_vel = ~np.isnan(vel_arr) & (vel_arr > VELOCITY_THRESHOLD_PX_S)
      Dicotemp['nan_velocity'] = np.where(nan_liquid_vel)[0].tolist()
      nan_liquid = nan_liquid_vel  # velocity is the only active NaN filter
      print(f'Velocity > {VELOCITY_THRESHOLD_PX_S}px/s: {np.sum(nan_liquid_vel)} frames ({100*np.mean(nan_liquid_vel):.2f}%)')
      
      # ── Temporal dilation — luminosity criterion only ────────────────────
      # Luminosity dilation logic (attention markers only):
      # uniform_filter1d: sliding window mean over LIQUID_WINDOW_S seconds
      # If > LIQUID_PCT_TRIGGER% of frames in window are above lum_threshold → window flagged
      # Purpose: catch gradual liquid buildup that doesn't trigger frame-by-frame threshold
      # crit_white (>99% white pixels) applied without dilation
     
      nan_liquid_lum_only = (lum_per_frame > lum_threshold)
      LIQUID_WINDOW_S = 120   # window size in seconds
      LIQUID_PCT_TRIGGER = 20  # % of liquid frames to trigger the window
      window_frames = int(LIQUID_WINDOW_S * fps)
      density = uniform_filter1d(nan_liquid_lum_only.astype(float), size=window_frames)
      nan_liquid_lum_dilated = density > (LIQUID_PCT_TRIGGER / 100)
      nan_liquid_white = pct_white_per_frame > pct_white_threshold
      
      # Luminosity and white pixels → attention markers only (not filtered)
      # nan_attention = luminosity dilation OR white pixels anomaly stored in Dico as frame indices list for figure marker display (green dots)
      # NOT applied as NaN filter — user reviews these frames in PostProcessingGUI

      nan_attention = nan_liquid_lum_dilated | nan_liquid_white
      Dicotemp['nan_attention'] = np.where(nan_attention)[0].tolist()
      n_attention = np.sum(nan_attention)
      print(f'Attention markers (luminosity/white pixels): {n_attention} frames ({100*np.mean(nan_attention):.1f}%)')
     
      
     # Final active filter: nan_liquid = nan_liquid_vel only
     # Luminosity/white pixels downgraded to attention markers after calibration showed
     # the velocity filter alone was sufficient for clean data
     # Attention frames = luminosity OR white pixels anomalies — stored but NOT filtered

      Dicotemp['nan_liquid'] = np.where(nan_liquid)[0].tolist()
      Dicotemp['lum_window_s'] = LIQUID_WINDOW_S
      Dicotemp['lum_pct_trigger'] = LIQUID_PCT_TRIGGER
      print(f'Seuil luminosité ROI: {lum_threshold:.1f} | frames liquide: {np.sum(nan_liquid)} ({100*np.mean(nan_liquid):.1f}%)')

      # ── Apply NaN filter to all signals ───────────────────────────────────────
      # Sets liquid frames (velocity > 800px/s) to NaN in ALL signals simultaneously
      # pupil_area and absolute_velocity: 1D arrays — direct NaN assignment
      # Keypoints (all bodyparts): iterate over all dict keys with 'x' field
      #   nan_liquid[:len(x)] — safety clip in case array lengths differ slightly
      #   likelihood NOT set to NaN here — kept as-is for traceability
      # After this block, Dico contains clean data ready for figure generation and pkl export
      pup_area = np.array(Dicotemp['pupil_area'], dtype=float)
      pup_area[nan_liquid] = np.nan
      Dicotemp['pupil_area'] = pup_area

      abs_vel = np.array(Dicotemp['absolute_velocity'], dtype=float)
      abs_vel[nan_liquid] = np.nan
      Dicotemp['absolute_velocity'] = abs_vel

      for bp in list(Dicotemp.keys()):
          if isinstance(Dicotemp[bp], dict) and 'x' in Dicotemp[bp]:
              x = np.array(Dicotemp[bp]['x'], dtype=float)
              y = np.array(Dicotemp[bp]['y'], dtype=float)
              x[nan_liquid[:len(x)]] = np.nan
              y[nan_liquid[:len(y)]] = np.nan
              Dicotemp[bp]['x'] = x
              Dicotemp[bp]['y'] = y
              
              
 # ── Frame timestamps and figure generation ────────────────────────────────
      # frames: list of timestamps in seconds for each frame (index / fps)
      # Used as x-axis for all temporal figures in control_image_1 and ComparaisonFaceMap
      Dicotemp['frames']=[i/fps for i in range(len(Dicotemp['pupil_area']))]
      
      
      # nametemp: base path for output files — derived from h5 path, extension removed
      # All figure files will be saved as nametemp + '_temporel.png', '_distri_et_spatial.png', etc.
      
    #print(Dicotemp) = debug
      nametemp=str(file_path_h5)
    #print(nametemp) = debug
      nametemp=nametemp[:-3]
    #print(nametemp) = debug
      session_dir = os.path.dirname(file_path_h5)  # directory containing the h5 file
      control_image_1(Dicotemp,nametemp)
      control_image_outliers(Dicotemp,nametemp,file_path_h5)
      control_image_1_raw_data(Dicotemp,nametemp,file_path_h5)
      control_image_1_HTML(Dicotemp,nametemp)
      control_image_2(Dicotemp,nametemp,file_path_video)
      control_image_2_HTML(Dicotemp,nametemp,file_path_video)
      control_image_2_raw_data(Dicotemp,nametemp,file_path_video,file_path_h5)
    #print(file_path_video) = debug
    #souris_seule(Dicotemp,nametemp)
    
    # Commented debug prints and souris_seule() — legacy single-mouse figure function




   # ── NaN statistics summary ────────────────────────────────────────────────
      # Printed to terminal after each session for quick quality control
      # Shows NaN count and % for pupil_area and all keypoints
      # First 10 NaN frame indices printed for manual inspection
      # nan_stats stored in Dico for downstream access (ComparaisonFaceMap, PostProcessingGUI)
      # Format: {signal: {count, pct, frames}} — frames = full list of NaN frame indices
      # '=' * 40 separator for readability in terminal when processing multiple sessions
      
      total_frames = len(Dicotemp['frames'])
      print(f'\n=== NaN stats for {nametemp} ===')
      print(f'Total frames: {total_frames}')
      
      # Pupil
      pup_nan = int(np.sum(np.isnan(np.array(Dicotemp['pupil_area'], dtype=float))))
      pup_nan_frames = np.where(np.isnan(np.array(Dicotemp['pupil_area'], dtype=float)))[0]
      print(f'pupil_area           NaN: {pup_nan} ({round(100*pup_nan/total_frames,1)}%)')
      if pup_nan > 0: print('  frames: ' + str(pup_nan_frames[:10].tolist()) + ('...' if pup_nan>10 else ''))
      
      # Keypoints
      kp_list = ['eye(front)','eye(back)','eye(top)','eye(bottom)',
                 'whisker(I)','whisker(II)','whisker(III)',
                 'nose(bottom)','nose(r)','nose(tip)','nosebridge',
                 'mouth','lowerlip','paw']
      Dicotemp['nan_stats'] = {'pupil_area': {'count': pup_nan, 'pct': round(100*pup_nan/total_frames,1), 'frames': pup_nan_frames.tolist()}}
      for bp in kp_list:
          if bp in Dicotemp:
              x = np.array(Dicotemp[bp]['x'], dtype=float)
              n = int(np.sum(np.isnan(x)))
              nan_f = np.where(np.isnan(x))[0]
              print(f'{bp.ljust(20)} NaN: {n} ({round(100*n/total_frames,1)}%)')
              if n > 0: print('  frames: ' + str(nan_f[:10].tolist()) + ('...' if n>10 else ''))
              Dicotemp['nan_stats'][bp] = {'count': n, 'pct': round(100*n/total_frames,1), 'frames': nan_f.tolist()}
      print('=' * 40)
      
 # ── Save pkl ───────────────────────────────────────────────────────────────
      # protocol=4: compatible with Python 3.4+ — supports large arrays (>4GB)
      # Saved first in session_dir, then moved to VF1/ below
      import pickle
      with open(str(nametemp)+'.pkl', 'wb') as file: pickle.dump(Dicotemp, file, protocol=4)
 
    
 # ── Move output files to VF1/ folder ──────────────────────────────────────
      # Two possible destination structures:
      #   1. 'Analyse pupille/VF1/' exists → old mouse structure (K1690/K1711/K1712)
      #   2. No 'Analyse pupille/' → new structure → create VF1/ directly in session_dir
      # Existing VF1/ renamed to old_version_X/ to preserve previous analyses
      # Files moved: all files starting with base_name EXCEPT .h5, .avi, .mp4, .npy, _metadata.pkl
      # moved counter printed for verification — should match number of figures + pkl
      analyse_pupille_dir = os.path.join(session_dir, 'Analyse pupille')
      # Determine final destination
      if os.path.isdir(analyse_pupille_dir):
          dst_vf1 = os.path.join(analyse_pupille_dir, 'VF1')
          # Rename existing VF1 if present
          if os.path.exists(dst_vf1):
              x = 1
              while os.path.exists(os.path.join(analyse_pupille_dir, f'old_version_{x}')):
                  x += 1
              old_name = os.path.join(analyse_pupille_dir, f'old_version_{x}')
              os.rename(dst_vf1, old_name)
              print(f'⚠️  Existing VF1 renamed to old_version_{x}/')
          # Create VF1 directly in Analyse pupille
          os.makedirs(dst_vf1, exist_ok=True)
          target_dir = dst_vf1
      else:
          # No Analyse pupille — create VF1 in session dir
          target_dir = os.path.join(session_dir, 'VF1')
          os.makedirs(target_dir, exist_ok=True)
          print(f'⚠️  Please create a folder named "Analyse pupille" and put VF1 in it')

      # Move files directly to final destination
      base_name = os.path.basename(nametemp)
      moved = 0
      EXCLUDED_EXTENSIONS = {'.h5', '.avi', '.mp4', '.npy'}
      EXCLUDED_SUFFIXES = {'_metadata.pkl'}
      for fname in os.listdir(session_dir):
          fpath = os.path.join(session_dir, fname)
          if not os.path.isfile(fpath):
              continue
          if not fname.startswith(base_name):
              continue
          ext = os.path.splitext(fname)[1].lower()
          if ext in EXCLUDED_EXTENSIONS:
              continue
          if any(fname.endswith(s) for s in EXCLUDED_SUFFIXES):
              continue
          os.rename(fpath, os.path.join(target_dir, fname))
          moved += 1
      print(f'Moved {moved} files to {target_dir}')

# ── Main loop ─────────────────────────────────────────────────────────────────
# try/except per video — if one session fails, analysis continues to the next
# Allows overnight batch processing without manual intervention
# Error message printed with video path for post-run diagnosis

for i in range(N):
  try:
      extract(npy_filepaths[i],h5_filepaths[i],video_filepaths[i])
  except Exception as e:
      print(f"\n❌ Error on video {video_filepaths[i]}: {e}")
      print("Passing to the next video...\n")
      continue #Continue the analysis even if there's an error on 1 file => Allows to run the analysis overnight


# ── Commented debug/legacy code ───────────────────────────────────────────────
# print(extract()) — debug call, replaced by the main loop above
# with open('output_dictionary_new.py'...) — legacy export to .py file, replaced by pkl

# Previously used to filter frames as NaN — now replaced by velocity filter only:
#   - Velocity > 800px/s → ACTIVE NaN filter
#   - Luminosity MAD threshold → downgraded to attention marker only
#   - % white pixels > 99% → downgraded to attention marker only
#   - Temporal dilation (120s, 20%) → still applied on luminosity for attention markers
#
# To reactivate luminosity/white pixels as NaN filters:
#   nan_liquid = nan_liquid_lum_dilated | nan_liquid_white | nan_liquid_vel
#
# Current calibration values (TO ADJUST if recording conditions change):
#   lum_threshold   = lum_median + 5 * lum_mad * 1.4826
#   pct_white_threshold = pct_white_median + 5 * pct_white_mad * 1.4826
#   LIQUID_WINDOW_S = 120s  (dilation window)
#   LIQUID_PCT_TRIGGER = 20%  (% liquid frames to trigger dilation)
#   VELOCITY_THRESHOLD_PX_S = 800 px/s

#Lines that are suppress because of the converting from .py to .pkl :
    #array=np.array
    #np.set_printoptions(threshold=np.inf) = Setting NumPy print options to display the full array (no ellipses)
    





# ── TEST FUNCTION: souris_seule() ─────────────────────────────────────────────


    #This was a test where only the mouse and the keypoints were plotted; I keep it here in case it is the olny thing you would like to keep (just uncomment the correspondng line i the extract function)
    # def souris_seule(Dico, name):
    #     video_path = 'WIN_20250626_14_41_12_Pro3.mp4'  # Replace this with the path to your video
    #     cap = cv2.VideoCapture(video_path)
    #     ret, frame = cap.read()
    #     if not ret:
    #         raise ValueError("Impossible de lire la vidéo ou la vidéo est vide.")
    #     # Convert the image from BGR (OpenCV) to RGB
    #     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     image = Image.fromarray(frame_rgb)
    # #    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    #     h=frame_rgb.shape[1]
    #     # Create a Plotly figure
    #     fig = go.Figure(layout=go.Layout(colorway=px.colors.qualitative.Light24))
    #     #fig.update_layout(yaxis=dict(autorange='reversed'))
    #     # Add scatter traces for each keypoint
    #     fig.add_trace(go.Scatter(x=Dico['eye(front)']['x'], y=Dico['eye(front)']['y'], mode='markers', name='eye(front)', marker=dict(size=2)))
    #     fig.add_trace(go.Scatter(x=Dico['eye(back)']['x'], y=Dico['eye(back)']['y'], mode='markers', name='eye(back)', marker=dict(size=2)))
    #     fig.add_trace(go.Scatter(x=Dico['eye(top)']['x'], y=Dico['eye(top)']['y'], mode='markers', name='eye(top)', marker=dict(size=2)))
    #     fig.add_trace(go.Scatter(x=Dico['eye(bottom)']['x'], y=Dico['eye(bottom)']['y'], mode='markers', name='eye(bottom)', marker=dict(size=2)))
    #     fig.add_trace(go.Scatter(x=Dico['whisker(I)']['x'], y=Dico['whisker(I)']['y'], mode='markers', name='whisker(I)', marker=dict(size=2)))
    #     fig.add_trace(go.Scatter(x=Dico['whisker(II)']['x'], y=Dico['whisker(II)']['y'], mode='markers', name='whisker(II)', marker=dict(size=2)))
    #     fig.add_trace(go.Scatter(x=Dico['whisker(III)']['x'], y=Dico['whisker(III)']['y'], mode='markers', name='whisker(III)', marker=dict(size=2)))
    #     fig.add_trace(go.Scatter(x=Dico['nose(bottom)']['x'], y=Dico['nose(bottom)']['y'], mode='markers', name='nose(bottom)', marker=dict(size=2)))
    #     fig.add_trace(go.Scatter(x=Dico['nose(r)']['x'], y=Dico['nose(r)']['y'], mode='markers', name='nose(r)', marker=dict(size=2)))
    #     fig.add_trace(go.Scatter(x=Dico['nose(tip)']['x'], y=Dico['nose(tip)']['y'], mode='markers', name='nose(tip)', marker=dict(size=2)))
    #     fig.add_trace(go.Scatter(x=Dico['nosebridge']['x'], y=Dico['nosebridge']['y'], mode='markers', name='nosebridge', marker=dict(size=2)))
    #     fig.add_trace(go.Scatter(x=Dico['mouth']['x'], y=Dico['mouth']['y'], mode='markers', name='mouth', marker=dict(size=2)))
    #     fig.add_trace(go.Scatter(x=Dico['lowerlip']['x'], y=Dico['lowerlip']['y'], mode='markers', name='lowerlip', marker=dict(size=2)))
    #     # Add the image as a background
    #     fig.update_layout(
    #         images=[dict(
    #             source=image,
    #             xref="x",
    #             yref="y",
    #             x=0,
    #             y=frame_rgb.shape[0],
    #             sizex=frame_rgb.shape[1],
    #             sizey=frame_rgb.shape[0],
    #             sizing="stretch",
    #             xanchor="left",
    #             yanchor="bottom",
    #             layer="below")])
    #     # Set the axis ranges to match the image dimensions
    #     #fig.update_xaxes(range=[0, frame_rgb.shape[1]])
    #     fig.update_xaxes(range=[0,frame_rgb.shape[1]])
    #     fig.update_yaxes(range=[frame_rgb.shape[0],0])
    #     # Update layout for the figure
    #     fig.update_layout(title_text='Detected keypoints over the mouse face image', legend=dict(itemsizing='constant', itemwidth=30))
    # # Save the entire figure as a HTML file
    #     output_file=str(name)+'_souris_seule.html'
    #     fig.write_html(output_file)