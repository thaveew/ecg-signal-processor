import wfdb
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

fs = 360 # sampling rate in Hz

# Load record 121
record = wfdb.rdrecord(234)
annotation = wfdb.rdann(234, 'atr')

true_peaks_annotation = annotation.sample
true_symbols = annotation.symbol

#count the number of normal beats and abnormal beats        
for i in range(len(true_symbols)):
    if true_symbols[i] == 'N':
        count_normal= count_normal + 1
    elif (true_symbols[i] == 'V' or true_symbols[i] == 'A' or true_symbols[i] == 'L' or true_symbols[i] == 'R' or true_symbols[i] == 'F' or true_symbols[i] == '/'):
            count_abnormal= count_abnormal + 1  

print("Number of normal beats: ", count_normal) 
print("Number of abnormal beats: ", count_abnormal)
