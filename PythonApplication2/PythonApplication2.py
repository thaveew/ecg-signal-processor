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
from numpy import ma
import wfdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

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


def build_dataset(record_list):
    fs= 360  # Sampling frequency (Hz)

    X, y = [], []
    match_tolerance = int(0.05 * fs)  # 50ms tolerance for matching detected peaks to true annotations

    for rec_name in record_list:
        record = wfdb.rdrecord(rec_name)
        annotation = wfdb.rdann(rec_name, 'atr')       
        fs = 360

        # --- Clean the signal (wide band: keeps 5-15Hz intact for pan_tompkins) ---
        raw_signal = record.p_signal[:, 0]
        filtered = bandpass_filter(raw_signal, lowcut=0.5, highcut=40)
        filtered = notch_filter(filtered, fs=fs)

        # --- Detect QRS complexes ---
        candidate_peaks = pan_tompkins(filtered)
        my_peaks_index = refine_peak_locations(filtered, candidate_peaks)

        # --- Align detected peaks with true annotations ---
        beat_mask = np.isin(annotation.symbol, BEAT_SYMBOLS)
        true_peaks_annotation = annotation.sample[beat_mask]
        true_symbols = np.array(annotation.symbol)[beat_mask]


        for i in range (1, len(my_peaks_index) - 1):  
            
            idx_closest_index = np.argmin(np.abs(true_peaks_annotation - my_peaks_index[i])) #index where the peak is closest to the true annotation
            if abs(true_peaks_annotation[idx_closest_index] - my_peaks_index[i]) > match_tolerance:
                continue # no confident matching annotation for this detected peak

            symbol = true_symbols[idx_closest_index]
           
            rr_prev = (my_peaks_index[i] - my_peaks_index[i-1]) / fs
            rr_next = (my_peaks_index[i+1] - my_peaks_index[i]) / fs
            amplitude = filtered[my_peaks_index[i]]
            qrs_width = compute_qrs_width(filtered, my_peaks_index[i], fs=360)


            X.append([rr_prev, rr_next, amplitude, qrs_width])
            y.append(to_binary_label(symbol))
   
    return np.array(X), np.array(y)

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

# 7. VISUALIZATION (single-record exploratory plots)

def plot_record_overview(record_name):
    """
    Plot an overview of the ECG record, including:
    - Filtered signals for both channels
    - Detected R-peaks
    - Heart rate over time
    - Heart rate variability metrics
    - Poincare plot
    record_name: name of the record to plot (e.g., '234')
    """
    record = wfdb.rdrecord(record_name)
    ecg_signals_0 = record.p_signal[:, 0]
    ecg_signals_1 = record.p_signal[:, 1]

    filtered_signal_1 = notch_filter(bandpass_filter(ecg_signals_0, lowcut=0.5, highcut=40))
    filtered_signal_2 = notch_filter(bandpass_filter(ecg_signals_1, lowcut=0.5, highcut=40))

    r_peaks_1 = refine_peak_locations(filtered_signal_1, pan_tompkins(filtered_signal_1))
    r_peaks_2 = refine_peak_locations(filtered_signal_2, pan_tompkins(filtered_signal_2))

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    for ax, sig, peaks, color, ch in (( axes[0], filtered_signal_1, r_peaks_1, 'blue', record.sig_name[0]), (axes[1], filtered_signal_2, r_peaks_2, 'red', record.sig_name[1])):

        ax.plot(sig, color=color, label=f'Filtered {ch}')
        ax.plot(peaks, sig[peaks], 'o', color='black', markersize=4, label='R-peaks')
        ax.set_title(f"R-peaks Detection {ch}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Voltage (mV)")
        ax.legend()
        ax.grid(True)
    plt.tight_layout()

    fig_rate, axes_rate_0 = plt.subplots(2, 2, figsize=(12, 8))
    fig_rate, axes_rate_1 = plt.subplots(2, 2, figsize=(12, 8))


    for ax, peaks, color, ch in ((axes_rate_0, r_peaks_1, 'blue', record.sig_name[0]), (axes_rate_1, r_peaks_2, 'red', record.sig_name[1])):
        hr = heart_rate_from_r_peaks(peaks)
        hr_times = peaks[1:]  # Heart rate corresponds to intervals between peaks
        ax[0,0].plot(hr_times, hr, color=color, label=f'Heart Rate {ch}')
        ax[0,0].set_title(f"Heart Rate {ch}")
        ax[0,0].set_xlabel("Sample Index")
        ax[0,0].set_ylabel("BPM")
        ax[0,0].legend()
        ax[0,0].grid(True) 
        
        sdnn, rmssd = heart_rate_variability(peaks)
        ax[0,1].bar(['SDNN', 'RMSSD'], [sdnn, rmssd], color=color)
        

        x, y = poincare_plot(peaks)
        ax[1,0].scatter(x, y, color=color, alpha=0.5)
        ax[1,0].set_title(f"Poincare Plot {ch}")
        ax[1,0].set_xlabel("RR(n) (s)")
        ax[1,0].set_ylabel("RR(n+1) (s)")
        ax[1,0].grid(True)

    plt.tight_layout()
    plt.show()


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
 
    # Evaluate the classifier on the test set
    y_pred = clf.predict(X_test)

    # Print classification report and confusion matrix
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal'], zero_division=0), flush=True)
    print(confusion_matrix(y_test, y_pred), flush=True)



