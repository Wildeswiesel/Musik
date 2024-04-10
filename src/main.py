import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def analyze_audio(file_path):
    # Audiofile laden
    y, sr = librosa.load(file_path)
    # Wellenform mit plt.plot darstellen
    plt.figure(figsize=(12, 4))
    plt.plot(np.linspace(0, len(y) / sr, num=len(y)), y)
    plt.title('Wellenform')
    plt.xlabel('Zeit (s)')
    plt.ylabel('Amplitude')
    plt.show()

    # Spektrogramm darstellen
    plt.figure(figsize=(12, 4))
    S = librosa.feature.melspectrogram(y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spektrogramm')
    plt.show()

    # Tempo berechnen
    tempo, _ = librosa.beat.beat_track(y, sr=sr)
    print(f"Tempo: {tempo} Schläge pro Minute")

if __name__ == "__main__":
    # Pfad zur Audiodatei - Ersetze diesen mit dem Pfad zu deiner Audiodatei
    audio_file_path = 'src/Hey Brother.mp3'
    analyze_audio(audio_file_path)
