import librosa
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Global variables for the GUI components
datei_eingabe = None
ergebnisse_text = None

def setup_gui():
    """
    Erstellt und konfiguriert das Hauptfenster der Benutzeroberfläche.
    """
    global datei_eingabe
    global ergebnisse_text

    fenster = tk.Tk()
    fenster.title("Musik-Analyse")

    # Eingabefeld für die Audiodatei
    eingabe_frame = tk.Frame(fenster)
    eingabe_frame.pack()

    datei_eingabe = tk.Entry(eingabe_frame, width=50)
    datei_eingabe.pack(side=tk.LEFT)

    # Auswahl der Audiodatei
    def datei_auswaehlen():
        datei_pfad = filedialog.askopenfilename(title="Audiodatei auswählen", filetypes=[("Audio Files", "*.wav *.mp3 *.flac")])
        datei_eingabe.delete(0, tk.END)
        datei_eingabe.insert(0, datei_pfad)

    datei_auswahl_button = tk.Button(eingabe_frame, text="...", command=datei_auswaehlen)
    datei_auswahl_button.pack(side=tk.LEFT)

    # Analyse-Button
    analyse_button = tk.Button(fenster, text="Analysieren", command=analyse_musik)
    analyse_button.pack(pady=10)

    # Ergebnisanzeige
    ergebnisse_frame = tk.Frame(fenster)
    ergebnisse_frame.pack()

    ergebnisse_text = tk.Text(ergebnisse_frame, width=80, height=20)
    ergebnisse_text.pack()

    fenster.mainloop()

def analyse_musik():
    """
    Analysiert die ausgewählte Audiodatei.
    """
    global datei_eingabe
    global ergebnisse_text

    datei_pfad = datei_eingabe.get()

    if not datei_pfad:
        return

    try:
        audio, sr = librosa.load(datei_pfad)
    except Exception as e:
        ergebnisse_text.delete(1.0, tk.END)
        ergebnisse_text.insert(tk.END, f"Fehler beim Laden der Datei: {e}")
        return

    # Analyse der Audiodatei
    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    segmente = np.zeros_like(audio)

    for i, onset_time in enumerate(onset_times):
        if i == len(onset_times) - 1:
            dauer = audio.shape[0] / sr - onset_time
        else:
            dauer = onset_times[i + 1] - onset_time

        start_frame = int(onset_time * sr)
        end_frame = int((onset_time + dauer) * sr)

        segment = audio[start_frame:end_frame]
        energie = librosa.feature.rms(y=segment).flatten()
        mittelwert_energie = np.mean(energie)

        if mittelwert_energie > np.median(energie):
            segmente[start_frame:end_frame] = 1  # Energiegeladen
        else:
            segmente[start_frame:end_frame] = 2  # Ruhig

    ergebnisse = {
        "segmente": segmente,
        "onset_zeiten": onset_times,
    }

    # Ergebnisse in der GUI anzeigen
    ergebnisse_text.delete(1.0, tk.END)
    ergebnisse_text.insert(tk.END, f"Energieniveau-Segmente:\n{segmente}\n\n")
    ergebnisse_text.insert(tk.END, f"Onset-Zeiten:\n{onset_times}")

if __name__ == "__main__":
    setup_gui()
