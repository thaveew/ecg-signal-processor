import wfdb
import matplotlib.pyplot as plt

# Load record 100
record = wfdb.rdrecord('100')
ecg_signals = record.p_signal

# Create a figure with 2 subplots vertically stacked
fig, axes = plt.subplots(2, 1)

# Plot Channel 0 (MLII) on the top graph
axes[0].plot(ecg_signals[:1000, 0], color='blue')
axes[0].set_title(f"Channel 0: {record.sig_name[0]}")
axes[0].set_ylabel("Voltage (mV)")
axes[0].grid(True)

# Plot Channel 1 (V1) on the bottom graph
axes[1].plot(ecg_signals[:1000, 1], color='red')
axes[1].set_title(f"Channel 1: {record.sig_name[1]}")
axes[1].set_xlabel("Sample Index")
axes[1].set_ylabel("Voltage (mV)")
axes[1].grid(True)

plt.tight_layout()
plt.show()