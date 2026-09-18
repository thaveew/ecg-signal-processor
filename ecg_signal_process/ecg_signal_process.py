"""
ECG QRS Detection and Arrhythmia Classification Pipeline
=========================================================
 
Pipeline stages:
    1. Signal cleaning       -> bandpass_filter, notch_filter
    2. QRS detection         -> pan_tompkins, adaptive_threshold
    3. Peak refinement       -> refine_peak_locations
    4. Feature extraction    -> compute_qrs_width, build_dataset
    5. Beat classification   -> DecisionTreeClassifier (Normal vs Abnormal)
 
Evaluation uses an INTER-PATIENT split (DS1 / DS2), not a random 80/20 split.
See the note above build_dataset() for why this matters for ECG data specifically.
"""

import wfdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

fs = 360  #  MIT BIH Arrhythmia Database sampling rate (Hz) shared by every record used here

# 1. SIGNAL CLEANING


def bandpass_filter(data, lowcut, highcut, fs=360, order=4):
    """
    Zero-phase Butterworth bandpass filter.
 
    Used here as a general-purpose ECG "cleaning" filter (e.g. 0.5-40 Hz) to remove
    baseline wander (very low frequency) and high-frequency noise, WITHOUT touching
    the QRS-relevant 5-15 Hz band that pan_tompkins() needs internally

    data: input signal 
    lowcut: lower cutoff frequency (Hz)
    highcut: upper cutoff frequency (Hz)
    fs: sampling frequency (Hz)
    order: filter order (higher = sharper rolloff)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    # Filter along samples axis (rows) for each channel
    filtered = filtfilt(b, a, data, axis=0)
    return filtered[:]

def notch_filter(data, notch_freq=60 , fs = 360, quality_factor=30):
    """
    Narrowband notch filter to remove powerline interference (60 Hz in the US,
    50 Hz elsewhere). 

    data: input signal 
    notch_freq: frequency to remove (Hz), e.g. 60 for US power line hum
    fs: sampling frequency (Hz)
    quality_factor: controls how narrow the notch is (higher = narrower)
    """
    nyquist = 0.5 * fs
    freq = notch_freq / nyquist  # normalize, same idea as bandpass

    b, a = iirnotch(freq, quality_factor)
    filtered = filtfilt(b, a, data)
    return filtered


# 2. QRS DETECTION (Pan-Tompkins)


def pan_tompkins(ecg_signal, fs=360):
    """
    Pan-Tompkins QRS detector.
    Pipeline: bandpass(5-15Hz) -> derivative -> squaring -> moving-window
    integration -> adaptive thresholding.
 
    Returns: approximate indices of detected QRS complexes (in the INTEGRATED
    signal's timeline). Pass these into refine_peak_locations() to get the
    true R-peak location in the raw/filtered ECG.
    """
    
    # ---- Step 1: Bandpass filter (5-15 Hz) ----
    # Removes baseline wander, muscle noise, and powerline interference
    # while preserving the QRS complex's dominant frequency content

    nyquist = 0.5 * fs
    low = 5 / nyquist
    high = 15 / nyquist
    b, a = butter(2, [low, high], btype='band')
    filtered = filtfilt(b, a, ecg_signal)
    
    # ---- Step 2: Derivative ----
    # Highlights the steep slope of the QRS complex
    # Standard 5-point derivative: y[n] = (1/8)(-x[n-2] -2x[n-1] +2x[n+1] +x[n+2])
    derivative_kernel = np.array([-1, -2, 0, 2, 1]) * (1/8)
    derivative = np.convolve(filtered, derivative_kernel, mode='same')
    
    # ---- Step 3: Squaring ----
    # Makes all values positive and emphasizes higher frequencies (QRS)
    squared = derivative ** 2
    
    # ---- Step 4: Moving window integration ----
    # Smooths the squared signal, producing a waveform whose width
    # correlates with QRS complex duration
    window_size = int(0.150 * fs)  # ~150ms window
    integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
    
    # ---- Step 5: Adaptive thresholding to find peaks ----
    peaks = adaptive_threshold(integrated)
    
    return peaks


def adaptive_threshold(integrated_signal, fs=360):
    """
    Adaptive peak detector for the Pan-Tompkins integrated signal.
 
    Key design points (each fixes a specific failure mode found during development):
      - Region tracking: once the signal crosses threshold, we track the TRUE
        maximum across the whole above-threshold region, instead of grabbing
        the first point that crosses (which could be a small ripple before the
        real R-peak).
      - Grace period: tolerates brief (~30ms) dips below threshold within one
        QRS region, so signal noise doesn't fracture one beat into two regions.
      - Search-back: if the gap since the last beat is much longer than the
        recent average RR interval, re-scan that gap with a HALVED threshold.
        This recovers real-but-smaller beats (seen with some abnormal/PVC
        beats) that get missed because SPKI/threshold spiked after a preceding
        unusually tall beat.
    """
    min_distance = int(0.2 * fs)  # 300bpm ceiling -> min plausible gap between real beats
    grace_samples = int(0.03 * fs)  # ~30ms grace period

    # Initial rough estimates of "typical signal peak" and "typical noise level"
    SPKI = np.max(integrated_signal[:2*fs]) * 0.25
    NPKI = np.mean(integrated_signal[:2*fs]) * 0.5
    threshold1 = NPKI + 0.25 * (SPKI - NPKI)
    
    peaks = []
    last_peak_idx = -min_distance # negative init so the very first real beat isn't blocked
    rr_history = []  # track recent RR intervals for search-back comparison
    
    i, n = 1, len(integrated_signal)
   
    while i < n - 1:
        if integrated_signal[i] > threshold1 and (i - last_peak_idx) > min_distance:
            # Walk forward through the whole above-threshold region, tracking its true max

            region_peak_idx = i
            region_peak_val = integrated_signal[i]
            j = i
            below_count = 0

            while j < n - 1:    
                
                if integrated_signal[j] > threshold1:
                    below_count = 0
                    if integrated_signal[j] > region_peak_val:
                        region_peak_val = integrated_signal[j]
                        region_peak_idx = j
                else:
                    #checking if the signal is below threshold for more than grace_samples                  
                    below_count += 1
                    if below_count > grace_samples:
                        break
                j += 1
            
            # ---- SEARCH-BACK: check if the gap before this beat is abnormally long ----
            if len(rr_history) >= 2:
                avg_rr = np.mean(rr_history[-8:]) # Get the avg of last 8 peaks
                current_rr = region_peak_idx - last_peak_idx
                
                if current_rr > 1.66 * avg_rr:
                    # Search the gap using a relaxed threshold (half of current)
                    search_threshold = 0.5 * threshold1
                    search_start = last_peak_idx + min_distance #possible region with a peak
                    search_end = region_peak_idx - min_distance #possible region with a peak
                    
                    if search_end > search_start:
                        search_segment = integrated_signal[search_start:search_end]
                        candidate_offset = np.argmax(search_segment) #Index of the maximum value in the search segment
                        candidate_val = search_segment[candidate_offset] # Value of the maximum in the search segment
                        
                        if candidate_val > search_threshold:
                            missed_idx = search_start + candidate_offset
                            peaks.append(missed_idx) #Adding that peak into the peak index list
                            rr_history.append(missed_idx - last_peak_idx)
                            last_peak_idx = missed_idx
                            SPKI = 0.125 * candidate_val + 0.875 * SPKI
                            threshold1 = NPKI + 0.25 * (SPKI - NPKI)
            
            # ---- Commit the originally-found peak ----
            peaks.append(region_peak_idx)
            rr_history.append(region_peak_idx - last_peak_idx)
            last_peak_idx = region_peak_idx
            SPKI = 0.125 * region_peak_val + 0.875 * SPKI
            threshold1 = NPKI + 0.25 * (SPKI - NPKI)
            
            i = j # skip past the region we just processed
        else:
            if (integrated_signal[i] > integrated_signal[i-1] and 
                integrated_signal[i] > integrated_signal[i+1]):
                NPKI = 0.125 * integrated_signal[i] + 0.875 * NPKI
                threshold1 = NPKI + 0.25 * (SPKI - NPKI)
            i += 1

    return np.array(peaks)

# 3. PEAK REFINEMENT

def refine_peak_locations(raw_signal, candidate_peaks, fs=360, search_window_ms=50):
    """
    Given approximate peak locations from the integrated signal,
    find the true R-peak location in the raw ECG signal.

    """
    search_radius = int((search_window_ms / 1000) * fs)
    refined_peaks = []
    
    for peak_idx in candidate_peaks:
        start = max(0, peak_idx - search_radius) # check whether starting point goes below 0        
        end = min(len(raw_signal), peak_idx + search_radius) # check whether ending point goes beyond the signal length 
        
        # Find the true local maximum within this small window
        local_window = raw_signal[start:end]
        true_peak_offset = np.argmax(np.abs(local_window))   # or np.argmax(np.abs(local_window)) if R can be negative
        
        true_peak_idx = start + true_peak_offset
        refined_peaks.append(true_peak_idx)
    
    return np.array(refined_peaks)

# 4. FEATURE EXTRACTION

def compute_qrs_width(signal, peak_idx, fs=360, window_ms=200, threshold_fraction=0.5):
    """
    Estimate QRS complex duration (seconds) by finding where the signal crosses
    half the peak's amplitude on either side of the peak.
 
    This is the single most discriminative feature for abnormal beats: PVCs and
    other ventricular beats are characteristically WIDE (often >120ms) compared
    to a normal QRS (~80-100ms), because they bypass the normal fast conduction
    pathway. RR interval and raw amplitude alone cannot capture this.
    """
    window_samples = int((window_ms / 1000) * fs)
    start_bound = max(0, peak_idx - window_samples)
    end_bound = min(len(signal), peak_idx + window_samples)
    threshold = threshold_fraction * abs(signal[peak_idx])
 
    onset_idx = start_bound
    for k in range(peak_idx, start_bound, -1):
        if abs(signal[k]) < threshold:
            onset_idx = k
            break
 
    offset_idx = end_bound
    for k in range(peak_idx, end_bound):
        if abs(signal[k]) < threshold:
            offset_idx = k
            break
 
    return (offset_idx - onset_idx) / fs


BEAT_SYMBOLS = ('N', 'L', 'R', 'A', 'V', 'F', '/')

def to_binary_label(symbol):
    """ Return 0 for normal beats ('N') and 1 for abnormal beats (all others). """
    return 0 if symbol == 'N' else 1

def process_record(rec_name):

    record = wfdb.rdrecord(rec_name, pn_dir ='mitdb')
    annotation = wfdb.rdann(rec_name, 'atr', pn_dir ='mitdb')
    fs = 360 #Sample frequency
 
    # Clean the signal with a bandpass filter and notch filter
    raw_signal = record.p_signal[:, 0]
    filtered = bandpass_filter(raw_signal, lowcut=0.5, highcut=40, fs=fs)
    filtered = notch_filter(filtered, fs=fs)
 
    # Detecting the actual peaks 
    candidate_peaks = pan_tompkins(filtered, fs=360)
    my_peaks = refine_peak_locations(filtered, candidate_peaks, fs=fs)
 
    # Masking the real annoations with targeted peaks
    beat_mask = np.isin(annotation.symbol, BEAT_SYMBOLS)
    true_peaks_annotation = annotation.sample[beat_mask]
    true_symbols = np.array(annotation.symbol)[beat_mask]
 
    match_tolerance = int(0.05 * fs)  # 50ms tolerance for matching detected peaks to true annotations
    X, y, matched_peak_indices = [], [], []
 
   
    for i in range(1, len(my_peaks) - 1):
         # Finding the nearest real annotaion
        idx_closest = np.argmin(np.abs(true_peaks_annotation - my_peaks[i]))
        if abs(true_peaks_annotation[idx_closest] - my_peaks[i]) > match_tolerance:
            continue
 
        # Calculating the features for each peak
        symbol = true_symbols[idx_closest]
        rr_prev = (my_peaks[i] - my_peaks[i - 1]) / fs
        rr_next = (my_peaks[i + 1] - my_peaks[i]) / fs
        amplitude = filtered[my_peaks[i]]
        qrs_width = compute_qrs_width(filtered, my_peaks[i], fs=fs)
 
        X.append([rr_prev, rr_next, amplitude, qrs_width])
        y.append(to_binary_label(symbol))
        matched_peak_indices.append(my_peaks[i])
 
    #Returning a dictionary
    return {
        "record_name": rec_name,
        "filtered_signal": filtered,
        "all_peaks": my_peaks,                       # every detected peak (for plotting)
        "matched_peaks": np.array(matched_peak_indices),  # peaks that got a valid feature/label
        "X": np.array(X),
        "y": np.array(y),
    }
  
def build_dataset(record_list):
    """Combine process_record() output across many records into one training set."""
    X_all, y_all = [], []
    for rec_name in record_list:
        result = process_record(rec_name)
        if len(result["X"]) == 0:
            continue
        X_all.append(result["X"])
        y_all.append(result["y"])
    return np.vstack(X_all), np.concatenate(y_all)
 

# 5. HRV / DERIVED METRICS (unchanged logic, kept for the visualization script)

def heart_rate_from_r_peaks(r_peaks, fs=360):
    """
    Calculate heart rate from R-peak indices.
    
    r_peaks: array of R-peak indices
    fs: sampling frequency (Hz)
    
    Returns: heart rate in beats per minute (BPM)
    """
    rr_intervals = np.diff(r_peaks) / fs  # in seconds
    heart_rate = 60 / rr_intervals  # convert to BPM
    return heart_rate

def heart_rate_variability(r_peaks, fs=360):
    """
    Calculate heart rate variability (HRV) metrics from R-peak indices.
    
    r_peaks: array of R-peak indices
    fs: sampling frequency (Hz)
    
    Returns: HRV metrics (e.g., SDNN, RMSSD)
    """
    rr_intervals = np.diff(r_peaks) / fs  # in seconds
    sdnn = np.std(rr_intervals)  # Standard deviation of NN intervals
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))  # Root mean square of successive differences
    return sdnn, rmssd

def poincare_plot(r_peaks, fs=360):
    """
    Generate a Poincare plot from R-peak indices.
    
    r_peaks: array of R-peak indices
    fs: sampling frequency (Hz)
    
    Returns: x and y coordinates for the Poincare plot
    """
    rr_intervals = np.diff(r_peaks) / fs  # in seconds
    x = rr_intervals[:-1]
    y = rr_intervals[1:]
    return x, y

# 6. TRAIN / TEST SPLIT INTER-PATIENT (DS1 / DS2), NOT RANDOM 80/20

Train_records = [
    '101', '106', '108', '109', '112', '114', '115', '116', '118', '119','122', '124', '201', '203', '205', '207', '208', '209', '215', '220', '223', '230',]
 
Test_records = [
    '100', '103', '105', '111', '113', '117', '121', '123', '200', '202', '210', '212', '213', '214', '219', '221', '222', '228', '231', '232','233', '234',]


# ============================================================
# 5. GUI
# ============================================================
 
class ECGRecordViewer(tk.Tk):
    """
    Main window layout:
      - Left column: one button per record.
      - Top-right: filtered signal + detected peaks (always shown once a
        record is selected).
      - Middle-right: that record's confusion matrix, annotated with
        accuracy/precision/recall/F1 (same numbers classification_report
        would print, just for this one record).
      - Bottom-right: a switchable panel - "Heart Rate", "HRV", or
        "Poincare Plot" buttons choose what's drawn here, for whichever
        record is currently selected.
    """
 
    def __init__(self, clf, record_list):
        super().__init__()
        self.clf = clf
        self.record_list = record_list
        self.current_result = None      # process_record() output for the selected record
        self.current_record_name = None
        self.bottom_view = "heart_rate"  # which plot the bottom panel shows
 
        self.title("ECG Per-Record Viewer")
        self.geometry("1200x850")
 
        # --- Left column: one button per record ---
        record_frame = ttk.Frame(self)
        record_frame.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
 
        ttk.Label(record_frame, text="Records", font=("Segoe UI", 11, "bold")).pack(pady=(0, 8))
        for rec_name in self.record_list:
            ttk.Button(
                record_frame, text=rec_name, width=10,
                command=lambda r=rec_name: self.show_record(r)
            ).pack(pady=2)
 
        # --- Right side: view-selector buttons on top, plot canvas below ---
        right_frame = ttk.Frame(self)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=8, pady=8)
 
        view_button_frame = ttk.Frame(right_frame)
        view_button_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
 
        ttk.Label(view_button_frame, text="Bottom panel:").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(view_button_frame, text="Heart Rate",
                   command=lambda: self.set_bottom_view("heart_rate")).pack(side=tk.LEFT, padx=2)
        ttk.Button(view_button_frame, text="HRV",
                   command=lambda: self.set_bottom_view("hrv")).pack(side=tk.LEFT, padx=2)
        ttk.Button(view_button_frame, text="Poincare Plot",
                   command=lambda: self.set_bottom_view("poincare")).pack(side=tk.LEFT, padx=2)
 
        self.figure = Figure(figsize=(9, 9), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
 
        self.status_var = tk.StringVar(value="Select a record to view its results.")
        ttk.Label(self, textvariable=self.status_var).pack(side=tk.BOTTOM, pady=4)
 
    # ---- Called when a record button is clicked ----
    def show_record(self, rec_name):
        self.status_var.set(f"Processing record {rec_name}...")
        self.update_idletasks()  # redraw the status text before the slower processing call runs
 
        self.current_result = process_record(rec_name)
        self.current_record_name = rec_name
        self.render()
 
        n_peaks = len(self.current_result["all_peaks"])
        n_matched = len(self.current_result["X"])
        self.status_var.set(f"Showing record {rec_name} "
                             f"({n_peaks} peaks detected, {n_matched} beats matched to annotations).")
 
    # ---- Called when a bottom-panel view button is clicked ----
    def set_bottom_view(self, view_name):
        self.bottom_view = view_name
        if self.current_result is not None:
            self.render()
 
    # ---- Draws all panels for whichever record/view is currently selected ----
    def render(self):
        if self.current_result is None:
            return
 
        result = self.current_result
        rec_name = self.current_record_name
        filtered = result["filtered_signal"]
        all_peaks = result["all_peaks"]
        all_peaks_1 = all_peaks[:100]
        X, y_true = result["X"], result["y"]
 
        self.figure.clear()
 
        # ---- Panel 1: filtered signal with detected peaks (first 3000 samples only) ----
        display_range = 30000
        signal_slice = filtered[:display_range]

        # Keep only peaks that actually fall within this displayed window
        peaks_in_range = all_peaks[all_peaks < display_range]

        ax_signal = self.figure.add_subplot(3, 1, 1)
        ax_signal.plot(signal_slice, color='blue', linewidth=0.8)
        ax_signal.plot(peaks_in_range, filtered[peaks_in_range], 'ro', markersize=3)    
        ax_signal.set_title(f"Record {rec_name} - Filtered Signal & Detected Peaks")
        ax_signal.set_xlabel("Sample Index")
        ax_signal.set_ylabel("Voltage (mV)")
 
        # ---- Panel 2: confusion matrix + accuracy/precision/recall/F1 ----
        ax_cm = self.figure.add_subplot(3, 1, 2)
        if len(X) > 0:
            y_pred = self.clf.predict(X)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
 
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=[0, 1], zero_division=0
            )
 
            ax_cm.imshow(cm, cmap='Blues')
            for r in range(2):
                for c in range(2):
                    ax_cm.text(c, r, str(cm[r, c]), ha='center', va='center', fontsize=12)
            ax_cm.set_xticks([0, 1]); ax_cm.set_xticklabels(['Normal', 'Abnormal'])
            ax_cm.set_yticks([0, 1]); ax_cm.set_yticklabels(['Normal', 'Abnormal'])
            ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("Actual")
            ax_cm.set_title(f"Confusion Matrix - Record {rec_name}  (Accuracy: {accuracy:.2f})")
 
            # Metrics text block, same numbers classification_report() would print,
            # placed beside the matrix rather than as a separate console printout.
            metrics_text = (
                f"{'':10s}{'Precision':>10s}{'Recall':>10s}{'F1':>8s}{'Support':>9s}\n"
                f"{'Normal':10s}{precision[0]:>10.2f}{recall[0]:>10.2f}{f1[0]:>8.2f}{support[0]:>9d}\n"
                f"{'Abnormal':10s}{precision[1]:>10.2f}{recall[1]:>10.2f}{f1[1]:>8.2f}{support[1]:>9d}"
            )
            self.figure.text(0.62, 0.53, metrics_text, family='monospace', fontsize=9,
                              verticalalignment='center')
        else:
            ax_cm.text(0.5, 0.5, "No matched beats for this record", ha='center', va='center')
            ax_cm.axis('off')
 
        # ---- Panel 3: switchable bottom view (Heart Rate / HRV / Poincare) ----
        ax_bottom = self.figure.add_subplot(3, 1, 3)
        if len(all_peaks) > 2:
            if self.bottom_view == "heart_rate":
                hr = heart_rate_from_r_peaks(all_peaks)
                ax_bottom.plot(all_peaks[1:], hr, color='green')
                ax_bottom.set_title(f"Heart Rate - Record {rec_name}")
                ax_bottom.set_xlabel("Sample Index")
                ax_bottom.set_ylabel("BPM")
 
            elif self.bottom_view == "hrv":
                sdnn, rmssd = heart_rate_variability(all_peaks)
                ax_bottom.bar(['SDNN', 'RMSSD'], [sdnn, rmssd], color=['steelblue', 'darkorange'])
                ax_bottom.set_title(f"HRV - Record {rec_name}")
                ax_bottom.set_ylabel("Seconds")
 
            elif self.bottom_view == "poincare":
                x_rr, y_rr = poincare_plot(all_peaks)
                ax_bottom.scatter(x_rr, y_rr, color='purple', alpha=0.5, s=10)
                ax_bottom.set_title(f"Poincare Plot - Record {rec_name}")
                ax_bottom.set_xlabel("RR(n) (s)")
                ax_bottom.set_ylabel("RR(n+1) (s)")
        else:
            ax_bottom.text(0.5, 0.5, "Not enough peaks for this view", ha='center', va='center')
            ax_bottom.axis('off')
 
        self.figure.tight_layout()
        self.canvas.draw()

# 8. MAIN build dataset, train, evaluate


if __name__ == "__main__":
    

    # Build the dataset for training and testing
    print("Building training set (DS1)...")
    X_train, y_train = build_dataset(Train_records)
    print("Building test set (DS2)...")
    X_test, y_test = build_dataset(Test_records)
 
    # Train a Decision Tree Classifier  
    clf = DecisionTreeClassifier(max_depth=3, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
 
    print("Classifier trained. Launching GUI...")
 
    #calling the GUI to visualize the results
    app = ECGRecordViewer(clf, Test_records)
    app.mainloop()

