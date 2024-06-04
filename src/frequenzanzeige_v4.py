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
    librosa.display.waveshow(y, sr=sr, alpha=0.8)
    plt.title('Wellenform')
    plt.xlabel('Zeit (s)')
    plt.ylabel('Amplitude')

    # Get time array for the waveform
    time = np.linspace(0, len(y) / sr, len(y))

    # Color the waveform based on amplitude envelope
    for i in range(len(amplitude_envelope)):
        start_sample = i * hop_length
        end_sample = start_sample + hop_length
        if end_sample > len(y):
            end_sample = len(y)

        if amplitude_envelope[i] < quiet_threshold:
            plt.plot(time[start_sample:end_sample], y[start_sample:end_sample], color='green', linewidth=0.5)
        elif amplitude_envelope[i] > loud_threshold:
            plt.plot(time[start_sample:end_sample], y[start_sample:end_sample], color='red', linewidth=0.5)
        else:
            plt.plot(time[start_sample:end_sample], y[start_sample:end_sample], color='purple', linewidth=0.5)

    # Calculate and display audio metadata
    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    avg_rms = np.mean(rms)
    max_rms = np.max(rms)

    # Display metadata
    metadata_text = f"""
    Metadata:
    - Sample rate: {sr} Hz
    - Duration: {duration:.2f} seconds
    - Average RMS: {avg_rms:.2f}
    - Max RMS: {max_rms:.2f}
    """
    plt.figtext(0.15, 0.02, metadata_text, fontsize=12)

    plt.tight_layout()
    plt.show()

# Example usage
file_path = 'data/Zeit/Zeit.mp3'  # Replace with your audio file path
analyze_audio(file_path)
