import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from mutagen.easyid3 import EasyID3  # For MP3 files with ID3 tags
from mutagen.flac import FLAC  # For FLAC files with metadata

def get_audio_metadata(file_path):
    metadata = {
        "Sample rate": None,
        "Duration": None,
        "Average RMS": None,
        "Max RMS": None,
        "Title": None,
        "Artist": None,
        "Album": None,
        "Year": None,
    }

    # Extract metadata using mutagen
    if file_path.endswith('.mp3'):
        audio = EasyID3(file_path)
    elif file_path.endswith('.flac'):
        audio = FLAC(file_path)
    else:
        audio = {}

    metadata["Title"] = audio.get("title", ["Unknown"])[0]
    metadata["Artist"] = audio.get("artist", ["Unknown"])[0]
    metadata["Album"] = audio.get("album", ["Unknown"])[0]
    metadata["Year"] = audio.get("date", ["Unknown"])[0]

    return metadata

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

    # Get additional metadata
    metadata = get_audio_metadata(file_path)

    # Initialize plot
    plt.figure(figsize=(14, 7))

    # Plot the waveform
    plt.subplot(2, 1, 1)
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

    # Calculate audio statistics
    duration = librosa.get_duration(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    avg_rms = np.mean(rms)
    max_rms = np.max(rms)

    # Update metadata with calculated values
    metadata["Sample rate"] = sr
    metadata["Duration"] = f"{duration:.2f} seconds"
    metadata["Average RMS"] = f"{avg_rms:.2f}"
    metadata["Max RMS"] = f"{max_rms:.2f}"

    # Display metadata
    metadata_text = "\n".join([f"{key}: {value}" for key, value in metadata.items()])
    plt.subplot(2, 1, 2)
    plt.axis('off')
    plt.text(0.1, 0.5, metadata_text, fontsize=12, verticalalignment='top')

    plt.tight_layout()
    plt.show()

# Example usage
file_path = 'data/Zeit/Zeit.mp3'  # Replace with your audio file path
analyze_audio(file_path)
