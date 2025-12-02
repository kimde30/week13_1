import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from sklearn.cluster import KMeans
import matplotlib.patches as patches

# chord‑extractor 사용
from chord_extractor.extractors import Chordino

# ------------------------------------------------------------
# 🎵 KEY DETECTION 함수
# ------------------------------------------------------------
def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    keys = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    minor_keys = [k + "m" for k in keys]

    major_template = np.array([1,0.1,0.8,0.1,1,1,0.1,1,0.1,0.8,0.1,0.8])
    minor_template = np.array([1,0.1,0.8,1,0.1,1,1,0.1,1,0.1,0.8,0.1])

    major_corr = [np.corrcoef(np.roll(major_template,i), chroma_mean)[0,1] for i in range(12)]
    minor_corr = [np.corrcoef(np.roll(minor_template,i), chroma_mean)[0,1] for i in range(12)]

    best_major = keys[np.argmax(major_corr)]
    best_minor = minor_keys[np.argmax(minor_corr)]

    return best_major if max(major_corr) >= max(minor_corr) else best_minor

# ------------------------------------------------------------
# 🎬 구조 분석 + 마디 단위 분할
# ------------------------------------------------------------
def analyze_structure_measures(y, sr, bpm=None, n_sections=5):
    if bpm is None:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo) if tempo > 0 else 120
    measure_duration = 60 / bpm * 4  # assuming 4/4

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc = librosa.util.normalize(mfcc)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    if len(beats) < n_sections:
        return None, measure_duration

    beat_features = librosa.util.sync(mfcc, beats, aggregate=np.mean).T
    n_clusters = min(n_sections, len(beat_features))

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(beat_features)

    times = librosa.frames_to_time(beats, sr=sr)
    min_len = min(len(times), len(labels))

    section_labels = ["A","B","C","D","E"]
    results = []
    for i in range(min_len - 1):
        start_sec = times[i]
        end_sec = times[i+1]
        start_measure = int(start_sec / measure_duration) + 1
        end_measure = int(end_sec / measure_duration) + 1
        results.append((section_labels[labels[i]], start_measure, end_measure))

    return results, measure_duration

# ------------------------------------------------------------
# 🎹 코드 진행 (Chord) 추출 함수
# ------------------------------------------------------------
def extract_chords(file_path):
    chordino = Chordino(roll_on=1)
    chords = chordino.extract(file_path)
    # chords: list of ChordChange(timestamp, chord)
    # return as-is
    return chords

# ------------------------------------------------------------
# 🎨 마디 기반 구조 + 코드 진행 시각화
# ------------------------------------------------------------
def plot_structure_and_chords(sections, chords, measure_duration):
    # 먼저 structure timeline by measures
    max_measure = max(end for _, _, end in sections)
    measures_per_line = 4
    n_lines = int(np.ceil(max_measure / measures_per_line))
    colors = plt.cm.tab20.colors

    fig, axs = plt.subplots(n_lines, 1, figsize=(14, 2 * n_lines))

    if n_lines == 1:
        axs = [axs]

    for line_idx, ax in enumerate(axs):
        start_m = line_idx * measures_per_line + 1
        end_m = (line_idx + 1) * measures_per_line

        ax.set_xlim(start_m, end_m)
        ax.set_ylim(0, 2)  # 0-1: structure, 1-2: chords
        ax.set_yticks([])
        ax.set_xlabel("Measures (마디)")
        ax.set_title(f"Measures {start_m}–{end_m}")

        # 구간 구조 막대
        for i, (part, s, e) in enumerate(sections):
            if e < start_m or s > end_m:
                continue
            s_clip = max(s, start_m)
            e_clip = min(e, end_m)
            width = e_clip - s_clip + 0.001
            ax.add_patch(
                patches.Rectangle(
                    (s_clip, 1.1),
                    width,
                    0.8,
                    color=colors[i % len(colors)],
                    alpha=0.6
                )
            )
            ax.text((s_clip + e_clip)/2, 1.5, part,
                    ha='center', va='center',
                    fontsize=9, color='white')

        # 코드 진행 시각화
        for ch in chords:
            t = ch.timestamp  # seconds
            chord_name = ch.chord
            # 초 → 마디
            m = int(t / measure_duration) + 1
            if m < start_m or m > end_m:
                continue
            ax.text(m, 0.3, chord_name,
                    ha='center', va='center',
                    fontsize=8, color='black', rotation=90)

    plt.tight_layout()
    st.pyplot(fig)

# ------------------------------------------------------------
# 🧪 Streamlit App 시작
# ------------------------------------------------------------
st.set_page_config(page_title="Music Analyzer + Chord Extraction", layout="centered")
st.markdown("<h1 style='text-align:center; color:#6C5AFF;'>🎶 Music Analyzer + Chord Extraction</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("MP3 파일 업로드", type=["mp3", "wav"])

if uploaded_file is not None:
    with open("tmp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())
    audio_path = "tmp_audio.wav"

    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo) if tempo > 0 else None

    key = detect_key(y, sr)

    st.write("**Detected Key:**", key)
    st.write("**Estimated BPM:**", round(bpm) if bpm else "Unknown")

    st.audio(audio_path)

    # 구조 분석
    sections, measure_duration = analyze_structure_measures(y, sr, bpm=bpm, n_sections=6)

    if sections:
        # 코드 추출
        chords = extract_chords(audio_path)
        st.write("🎹 코드 진행 (자동 감지)")

        plot_structure_and_chords(sections, chords, measure_duration)
    else:
        st.warning("구조 분석 실패 — 비트 부족")
