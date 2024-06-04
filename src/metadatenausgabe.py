import numpy as np
from pydub import AudioSegment
import matplotlib.pyplot as plt

# Funktion zum Extrahieren von Audiodaten und Identifizieren lauter und leiser Stellen
def analyze_audio(file_path, threshold=-20.0):
    # Audiodatei laden
    audio = AudioSegment.from_file(file_path)
    
    # Audiodaten in dBFS (Decibels relative to full scale) umwandeln
    samples = np.array(audio.get_array_of_samples())
    samples = samples / np.power(2, 15)  # Normierung auf [-1, 1]
    samples = 20 * np.log10(np.abs(samples))  # Umwandlung in dBFS
    
    # Fensterweise Analyse (z.B. 100 ms Fenster)
    window_size = int(audio.frame_rate / 10)
    loudness = [np.mean(samples[i:i + window_size]) for i in range(0, len(samples), window_size)]
    
    # Identifikation lauter und leiser Stellen
    loud_sections = [i for i, l in enumerate(loudness) if l > threshold]
    quiet_sections = [i for i, l in enumerate(loudness) if l <= threshold]
    
    # Umwandlung in Zeit (in Sekunden)
    loud_times = [i * 0.1 for i in loud_sections]
    quiet_times = [i * 0.1 for i in quiet_sections]
    
    return loudness, loud_times, quiet_times

# Funktion zur Visualisierung
def plot_loudness(loudness, loud_times, quiet_times):
    times = np.arange(0, len(loudness) * 0.1, 0.1)
    
    plt.figure(figsize=(15, 5))
    plt.plot(times, loudness, label="Loudness (dBFS)")
    plt.scatter(loud_times, [loudness[int(t * 10)] for t in loud_times], color='red', label="Loud Sections")
    plt.scatter(quiet_times, [loudness[int(t * 10)] for t in quiet_times], color='blue', label="Quiet Sections")
    
    plt.xlabel("Time (s)")
    plt.ylabel("Loudness (dBFS)")
    plt.title("Loud and Quiet Sections in Audio")
    plt.legend()
    plt.show()

# Hauptprogramm
if __name__ == "__main__":
    file_path = "E:/Zeit/Zeit.mp3"  # Pfad zur Audiodatei
    loudness, loud_times, quiet_times = analyze_audio(file_path)
    plot_loudness(loudness, loud_times, quiet_times)
