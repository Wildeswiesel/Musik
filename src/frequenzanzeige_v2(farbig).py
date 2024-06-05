import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

def analyze_audio(file_path):
    # Load the audio file
    y, sr = librosa.load(file_path)

    # Calculate the amplitude envelope
    hop_length = 512
    frame_length = 1024
    amplitude_envelope = np.array([
        max(y[i:i + frame_length])
        for i in range(0, len(y), hop_length)
    ])

    # Define thresholds for quiet, medium, and loud segments
    quiet_threshold = 0.2
    loud_threshold = 0.6

    # Initialize plot
    plt.figure(figsize=(14, 5))

    # Plot the waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.5)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Mark quiet, medium, and loud segments
    time = np.arange(len(amplitude_envelope)) * hop_length / sr
    for i, amp in enumerate(amplitude_envelope):
        if amp < quiet_threshold:
            plt.axvspan(time[i] - hop_length / sr / 2, time[i] + hop_length / sr / 2, color='red', alpha=0.5)
        elif amp > loud_threshold:
            plt.axvspan(time[i] - hop_length / sr / 2, time[i] + hop_length / sr / 2, color='green', alpha=0.5)
        else:
            plt.axvspan(time[i] - hop_length / sr / 2, time[i] + hop_length / sr / 2, color='purple', alpha=0.5)

    # Calculate and display audio metadata
    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    avg_rms = np.mean(rms)
    max_rms = np.max(rms)

    plt.subplot(2, 1, 2)
    plt.axis('off')
    metadata = f"""
    Metadata:
    - Sample rate: {sr} Hz
    - Duration: {duration:.2f} seconds
    - Average RMS: {avg_rms:.2f}
    - Max RMS: {max_rms:.2f}
    """
    plt.text(0.1, 0.5, metadata, fontsize=12)

    plt.tight_layout()
    plt.show()

# Example usage
file_path = 'data/Zeit/Zeit.mp3'  # Replace with your audio file path
analyze_audio(file_path)
