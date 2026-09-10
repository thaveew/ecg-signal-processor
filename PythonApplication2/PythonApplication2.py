import wfdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

fs = 360  # sampling rate in Hz

# Load record 121
record = wfdb.rdrecord('234')
ecg_signals_0 = record.p_signal[:, 0]
ecg_signals_1 = record.p_signal[:, 1]

def bandpass_filter(data, lowcut, highcut, fs=360, order=4):
    """
    data: input signal (1D array)
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
    data: input signal (1D array)
    notch_freq: frequency to remove (Hz), e.g. 60 for US power line hum
    fs: sampling frequency (Hz)
    quality_factor: controls how narrow the notch is (higher = narrower)
    """
    nyquist = 0.5 * fs
    freq = notch_freq / nyquist  # normalize, same idea as bandpass

    b, a = iirnotch(freq, quality_factor)
    filtered = filtfilt(b, a, data)
    return filtered


def pan_tompkins(ecg_signal, fs=360):
    """
    Pan-Tompkins QRS detection algorithm.
    
    ecg_signal: 1D array of raw ECG data
    fs: sampling frequency (Hz)
    
    Returns: indices of detected R-peaks
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
    min_distance = int(0.2 * fs)
    
    SPKI = np.max(integrated_signal[:2*fs]) * 0.25
    NPKI = np.mean(integrated_signal[:2*fs]) * 0.5
    threshold1 = NPKI + 0.25 * (SPKI - NPKI)
    
    peaks = []
    last_peak_idx = -min_distance
    rr_history = []  # track recent RR intervals for search-back comparison
    
    i = 1
    n = len(integrated_signal)
    grace_samples = int(0.03 * fs)

    while i < n - 1:
        if integrated_signal[i] > threshold1 and (i - last_peak_idx) > min_distance:
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
                    below_count += 1
                    if below_count > grace_samples:
                        break
                j += 1
            
            # ---- SEARCH-BACK: check if the gap before this beat is abnormally long ----
            if len(rr_history) >= 2:
                avg_rr = np.mean(rr_history[-8:])
                current_rr = region_peak_idx - last_peak_idx
                
                if current_rr > 1.66 * avg_rr:
                    # Search the gap using a relaxed threshold (half of current)
                    search_threshold = 0.5 * threshold1
                    search_start = last_peak_idx + min_distance
                    search_end = region_peak_idx - min_distance
                    
                    if search_end > search_start:
                        search_segment = integrated_signal[search_start:search_end]
                        candidate_offset = np.argmax(search_segment)
                        candidate_val = search_segment[candidate_offset]
                        
                        if candidate_val > search_threshold:
                            missed_idx = search_start + candidate_offset
                            peaks.append(missed_idx)
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
            
            i = j
        else:
            if (integrated_signal[i] > integrated_signal[i-1] and 
                integrated_signal[i] > integrated_signal[i+1]):
                NPKI = 0.125 * integrated_signal[i] + 0.875 * NPKI
                threshold1 = NPKI + 0.25 * (SPKI - NPKI)
            i += 1

    return np.array(peaks)

def refine_peak_locations(raw_signal, candidate_peaks, fs=360, search_window_ms=100):
    """
    Given approximate peak locations from the integrated signal,
    find the true R-peak location in the raw ECG signal.
    """
    search_radius = int((search_window_ms / 1000) * fs / 2)
    refined_peaks = []
    
    for peak_idx in candidate_peaks:
        start = max(0, peak_idx - search_radius)
        end = min(len(raw_signal), peak_idx + search_radius)
        
        # Find the true local maximum within this small window
        local_window = raw_signal[start:end]
        true_peak_offset = np.argmax(np.abs(local_window))   # or np.argmax(np.abs(local_window)) if R can be negative
        
        true_peak_idx = start + true_peak_offset
        refined_peaks.append(true_peak_idx)
    
    return np.array(refined_peaks)

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

# Create a figure with 4 subplots vertically stacked
fig, axes = plt.subplots(2, 1)

#Create a figure with 2 subplots vertically stacked
fig, axes_rate = plt.subplots(2, 1, figsize=(12, 8))

#create a figure with 2 subplots vertically stacked
fig, axes_hrv = plt.subplots(2, 1, figsize=(12, 8))

#create a figure with 2 subplots vertically stacked
fig, axes_poincare = plt.subplots(2, 1, figsize=(12, 8))    

# Plot Channel 0 (MLII) on the top graph
filtered_signal_1 = bandpass_filter(ecg_signals_0, lowcut=0.5, highcut=40)
filtered_signal_1 = notch_filter(filtered_signal_1)

# Plot Channel 1 (V1) on the bottom graph 
filtered_signal_2 = bandpass_filter(ecg_signals_1, lowcut=0.5, highcut=40)
filtered_signal_2 = notch_filter(filtered_signal_2)


# Apply Pan-Tompkins algorithm to detect R-peaks in Channel 0
r_peaks = pan_tompkins(filtered_signal_1)
true_peaks_1 = refine_peak_locations(filtered_signal_1, r_peaks)  
axes[0].plot(filtered_signal_1, color='blue', label='Filtered Signal_1')
axes[0].plot(true_peaks_1, filtered_signal_1[true_peaks_1], 'bo', label='R-peaks')
axes[0].set_title("R-peaks Detection")
axes[0].set_xlabel("Sample Index")
axes[0].set_ylabel("Voltage (mV)")
axes[0].legend()
axes[0].grid(True)


# Apply pan-Tompkins algorithm to detect R-peaks in Channel 1   
r_peaks_2 = pan_tompkins(filtered_signal_2)
true_peaks_2 = refine_peak_locations(filtered_signal_2, r_peaks_2)
axes[1].plot(filtered_signal_2, color='red', label='Filtered Signal_2')
axes[1].plot(true_peaks_2, filtered_signal_2[true_peaks_2], 'ro', label='R-peaks')
axes[1].set_title("R-peaks Detection")
axes[1].set_xlabel("Sample Index")
axes[1].set_ylabel("Voltage (mV)")
axes[1].legend()
axes[1].grid(True)

#plot heart rate for channel 0
hr_times_0 = true_peaks_1[1:]           
heart_rate_0 = heart_rate_from_r_peaks(true_peaks_1)
axes_rate[0].plot(hr_times_0,heart_rate_0, color='blue', label='Heart Rate Channel 0')
axes_rate[0].set_title("Heart Rate Channel 0")
axes_rate[0].set_xlabel("Sample Index")
axes_rate[0].set_ylabel("BPM")
axes_rate[0].legend()
axes_rate[0].grid(True)

#plot heart rate for channel 1
hr_times_1 = true_peaks_2[1:]
heart_rate_1 = heart_rate_from_r_peaks(true_peaks_2)
axes_rate[1].plot(hr_times_1,heart_rate_1, color='red', label='Heart Rate Channel 1')
axes_rate[1].set_title("Heart Rate Channel 1")
axes_rate[1].set_xlabel("Sample Index")
axes_rate[1].set_ylabel("BPM")
axes_rate[1].legend()
axes_rate[1].grid(True)

#plot heart rate variability for channel 0
sdnn_0, rmssd_0 = heart_rate_variability(true_peaks_1)
axes_hrv[0].bar(['SDNN', 'RMSSD'], [sdnn_0, rmssd_0], color=['blue', 'cyan'])

#plot heart rate variability for channel 1
sdnn_1, rmssd_1 = heart_rate_variability(true_peaks_2)
axes_hrv[1].bar(['SDNN', 'RMSSD'], [sdnn_1, rmssd_1], color=['red', 'orange'])

#plot poincare plot for channel 0
x_0, y_0 = poincare_plot(true_peaks_1)
axes_poincare[0].scatter(x_0, y_0, color='blue', alpha=0.5)
axes_poincare[0].set_title("Poincare Plot Channel 0")
axes_poincare[0].set_xlabel("RR(n) (s)")
axes_poincare[0].set_ylabel("RR(n+1) (s)")
axes_poincare[0].grid(True)

#plot poincare plot for channel 1
x_1, y_1 = poincare_plot(true_peaks_2)
axes_poincare[1].scatter(x_1, y_1, color='red', alpha=0.5)
axes_poincare[1].set_title("Poincare Plot Channel 1")
axes_poincare[1].set_xlabel("RR(n) (s)")
axes_poincare[1].set_ylabel("RR(n+1) (s)")
axes_poincare[1].grid(True)


plt.tight_layout()
plt.show()

def to_binary_label(symbol):
    return 0 if symbol == 'N' else 1


def build_dataset(record_list):
    X, y = [], []

    for rec_name in record_list:
        record = wfdb.rdrecord(rec_name)
        annotation = wfdb.rdann(rec_name, 'atr')

        raw_signal = record.p_signal[:, 0]
        fs = 360

        filtered = bandpass_filter(raw_signal, lowcut=0.5, highcut=40)
        filtered = notch_filter(filtered, fs=fs)
        candidate_peaks = pan_tompkins(filtered)
        my_peaks_time = refine_peak_locations(filtered, candidate_peaks)

        true_peaks_annotation = annotation.sample
        true_symbols = annotation.symbol


        for i in range (1, len(my_peaks_time) - 1):
            
            idx_closest_index = np.argmin(np.abs(true_peaks_annotation - my_peaks_time[i]))
            if abs(true_peaks_annotation[idx_closest_index] - my_peaks_time[i]) > int(0.05*fs):
                continue
            symbol = true_symbols[idx_closest_index]
           
            if symbol not in ('N', 'L', 'R', 'A', 'V', 'F', '/'):
                continue
           
            rr_prev = (my_peaks_time[i] - my_peaks_time[i-1]) / fs
            rr_next = (my_peaks_time[i+1] - my_peaks_time[i]) / fs
            amplitude = filtered[my_peaks_time[i]]

            X.append([rr_prev, rr_next, amplitude])
            y.append(to_binary_label(symbol))
   
    return np.array(X), np.array(y)

    

# Build the dataset for training and testing
X_train, y_train = build_dataset(['100', '101', '103', '104', '105','106','107','108', '109', '111', '112', '114', '115', '116', '117', '118', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '207', '208', '209', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230', '231', '232','233', '234'])
X_test, y_test = build_dataset(['210'])

# Train a Decision Tree Classifier  
clf = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
clf.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = clf.predict(X_test)

# Print classification report and confusion matrix
print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal'], zero_division=0), flush=True)
print(confusion_matrix(y_test, y_pred), flush=True)

