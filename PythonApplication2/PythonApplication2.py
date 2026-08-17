import wfdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

fs = 360  # sampling rate in Hz

# Load record 100
record = wfdb.rdrecord('100')
ecg_signals = record.p_signal

def bandpass_filter(index, data, lowcut, highcut, fs=360, order=4):
    """
    data: input signal (2D array)
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
    if index == 1:
        return filtered[:1000, 0]  # channel 0
    else:
        return filtered[:1000, 1]  # channel 1

def notch_filter(data, notch_freq=60 , fs = 360, quality_factor=30):
    """
    data: input signal (2D array)
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
    
    i = 1
    n = len(integrated_signal)
    grace_samples = int(0.03 * fs)  # tolerate up to ~30ms below threshold before truly exiting

    while i < n - 1:
       
        if integrated_signal[i] > threshold1 and (i - last_peak_idx) > min_distance:
            # We've entered a region above threshold,find its TRUE max
            # by walking forward until the signal drops back below threshold
            region_start = i
            region_peak_idx = i
            region_peak_val = integrated_signal[i]
            
            j = i
            below_count = 0  # count how many samples we've been below threshold
            while j < n - 1:
                if integrated_signal[j] > threshold1:
                    below_count = 0  # reset grace counter,we're back above threshold
                    if integrated_signal[j] > region_peak_val:
                        region_peak_val = integrated_signal[j]
                        region_peak_idx = j
                else:
                    below_count += 1
                    if below_count > grace_samples:
                        break  # sustained drop,region genuinely ended
                j += 1
            
            # Commit to the TRUE max of this above threshold region
            peaks.append(region_peak_idx)
            last_peak_idx = region_peak_idx
            SPKI = 0.125 * region_peak_val + 0.875 * SPKI
            threshold1 = NPKI + 0.25 * (SPKI - NPKI)
            
            i = j  # skip ahead past this region, avoid re-detecting inside it
        else:
            # Track sub-threshold local maxima to update NPKI (noise estimate)
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
        true_peak_offset = np.argmax(local_window)  # or np.argmax(np.abs(local_window)) if R can be negative
        
        true_peak_idx = start + true_peak_offset
        refined_peaks.append(true_peak_idx)
    
    return np.array(refined_peaks)

# Create a figure with 4 subplots vertically stacked
fig, axes = plt.subplots(4, 1)

# Plot Channel 0 (MLII) on the top graph
filtered_signal_1 = bandpass_filter(1,ecg_signals, lowcut=20, highcut=80)
filtered_signal_1 = notch_filter(filtered_signal_1)
axes[0].plot(filtered_signal_1, color='blue')
axes[0].set_title(f"Channel 0: {record.sig_name[0]}")
axes[0].set_ylabel("Voltage (mV)")
axes[0].grid(True)

# Plot Channel 1 (V1) on the bottom graph 
filtered_signal_2 = bandpass_filter(2,ecg_signals, lowcut=20, highcut=80)
filtered_signal_2 = notch_filter(filtered_signal_2)
axes[1].plot(filtered_signal_2, color='red')
axes[1].set_title(f"Channel 1: {record.sig_name[1]}")
axes[1].set_xlabel("Sample Index")
axes[1].set_ylabel("Voltage (mV)")
axes[1].grid(True)

# Apply Pan-Tompkins algorithm to detect R-peaks in Channel 0
r_peaks = pan_tompkins(filtered_signal_1)
true_peaks_1 = refine_peak_locations(filtered_signal_1, r_peaks)  
axes[2].plot(filtered_signal_1, color='blue', label='Filtered Signal_1')
axes[2].plot(true_peaks_1, filtered_signal_1[true_peaks_1], 'ro', label='R-peaks')
axes[2].set_title("R-peaks Detection")
axes[2].set_xlabel("Sample Index")
axes[2].set_ylabel("Voltage (mV)")
axes[2].legend()
axes[2].grid(True)

# Apply pan-Tompkins algorithm to detect R-peaks in Channel 1   
r_peaks_2 = pan_tompkins(filtered_signal_2)
true_peaks_2 = refine_peak_locations(filtered_signal_2, r_peaks_2)
axes[3].plot(filtered_signal_2, color='red', label='Filtered Signal_2')
axes[3].plot(true_peaks_2, filtered_signal_2[true_peaks_2], 'ro', label='R-peaks')
axes[3].set_title("R-peaks Detection")
axes[3].set_xlabel("Sample Index")
axes[3].set_ylabel("Voltage (mV)")
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()