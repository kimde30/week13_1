import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import matplotlib.patches as patches
from sklearn.cluster import KMeans
import random

# ------------------------------
# Key Detection
# ------------------------------
def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    keys = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    minor_keys = [k+"m" for k in keys]
    major_template = np.array([1,0.1,0.8,0.1,1,1,0.1,1,0.1,0.8,0.1,0.8])
    minor_template = np.array([1,0.1,0.8,1,0.1,1,1,0.1,1,0.1,0.8,0.1])
    major_corr = [np.corrcoef(np.roll(major_template,i), chroma_mean)[0,1] for i in range(12)]
    minor_corr = [np.corrcoef(np.roll(minor_template,i), chroma_mean)[0,1] for i in range(12)]
    best_major = keys[np.argmax(major_corr)]
    best_minor = minor_keys[np.argmax(minor_corr)]
    return best_major if max(major_corr) >= max(minor_corr) else best_minor

# ------------------------------
# Structure Analysis (Measure-based)
# ------------------------------
def analyze_structure_measures(y, sr, bpm=None, n_sections=5):
    if bpm is None:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo) if tempo > 0 else 120
    measure_duration = 60 / bpm * 4
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

# ------------------------------
# Plot Song Structure with Measures & simple chord
# ------------------------------
def plot_structure_and_chords(sections, measure_duration):
    measures_per_line = 4
    max_measure = max(end for _,_,end in sections)
    n_lines = int(np.ceil(max_measure / measures_per_line))
    colors = plt.cm.tab20.colors

    fig, axs = plt.subplots(n_lines, 1, figsize=(14, 2*n_lines))
    if n_lines == 1:
        axs = [axs]

    for line_idx, ax in enumerate(axs):
        start_m = line_idx * measures_per_line + 1
        end_m = (line_idx+1) * measures_per_line
        ax.set_xlim(start_m, end_m)
        ax.set_ylim(0, 2)
        ax.set_yticks([])
        ax.set_xlabel("Measures (마디)", fontsize=11)
        ax.set_title(f"Measures {start_m}–{end_m}", fontsize=12, fontweight='bold', color="#6C63FF")

        # 구조 막대
        for i, (part, s, e) in enumerate(sections):
            if e < start_m or s > end_m:
                continue
            s_clip = max(s, start_m)
            e_clip = min(e, end_m)
            width = e_clip - s_clip + 0.001
            ax.add_patch(patches.Rectangle(
                (s_clip,1.1), width,0.8,
                color=colors[i%len(colors)],
                alpha=0.7,
                edgecolor="black"
            ))
            ax.text((s_clip+e_clip)/2,1.5,part,ha='center',va='center',fontsize=10,color='white',fontweight='bold')

        # 코드 진행 (마디별 랜덤)
        for m in range(start_m,end_m+1):
            chord = random.choice(["C","Dm","Em","F","G","Am"])
            ax.text(m,0.3,chord,ha='center',va='center',fontsize=9,color='#FF5733',rotation=90,fontweight='bold')

    plt.tight_layout()
    st.pyplot(fig)

# ------------------------------
# Streamlit App
# ------------------------------
st.set_page_config(page_title="🎶 Music Analyzer Lite", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background-color: #F0F2FF; }
    .stFileUploader > label { font-weight: bold; color: #6C63FF; }
    </style>
    """, unsafe_allow_html=True
)

st.markdown("<h1 style='text-align:center; color:#6C63FF;'>🎶 Music Analyzer Lite</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>MP3/WAV 파일을 업로드하면 자동으로 BPM, Key, 구조 분석을 수행합니다.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("파일 업로드", type=["mp3","wav"])

if uploaded_file is not None:
    with open("tmp_audio.wav","wb") as f:
        f.write(uploaded_file.read())
    audio_path = "tmp_audio.wav"
    y,sr = librosa.load(audio_path,sr=None,mono=True)
    duration = librosa.get_duration(y=y,sr=sr)
    tempo,_ = librosa.beat.beat_track(y=y,sr=sr)
    bpm = float(tempo) if tempo>0 else None
    key = detect_key(y,sr)

    # 결과 박스
    st.markdown("<div style='background-color:#E0E0FF;padding:10px;border-radius:10px'>", unsafe_allow_html=True)
    st.markdown(f"**🎵 Detected Key:** {key}")
    st.markdown(f"**⏱ Estimated BPM:** {round(bpm) if bpm else 'Unknown'}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.audio(audio_path)

    # 구조 분석
    sections, measure_duration = analyze_structure_measures(y,sr,bpm=bpm,n_sections=5)
    if sections:
        plot_structure_and_chords(sections, measure_duration)
    else:
        st.warning("구조 분석 실패 — 비트 부족")
