import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from sklearn.cluster import KMeans
import matplotlib.patches as patches

# ------------------------------------------------------------
# 🍀 Streamlit GUI 설정
# ------------------------------------------------------------
st.set_page_config(
    page_title="🎶 Music Analyzer Advanced",
    page_icon="🎧",
    layout="centered"
)

st.markdown(
    """
    <h1 style='text-align: center; color:#6C63FF;'>🎶 Music Analyzer Advanced</h1>
    <p style='text-align: center; color:#555; font-size:17px;'>
        MP3 파일을 업로드하면 BPM, Key, Waveform, Spectrogram, 자동 마디 기반 구조 분석 및 시각화를 제공합니다.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ------------------------------------------------------------
# 🎵 MP3 업로드
# ------------------------------------------------------------
uploaded_file = st.file_uploader("MP3 파일 업로드", type=["mp3"])

# ------------------------------------------------------------
# 🎼 Key Detection Function
# ------------------------------------------------------------
def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    keys = ["C", "C#", "D", "D#", "E", "F",
            "F#", "G", "G#", "A", "A#", "B"]
    minor_keys = [k + "m" for k in keys]

    major_template = np.array([1, 0.1, 0.8, 0.1, 1, 1, 0.1, 1, 0.1, 0.8, 0.1, 0.8])
    minor_template = np.array([1, 0.1, 0.8, 1, 0.1, 1, 1, 0.1, 1, 0.1, 0.8, 0.1])

    major_corr = [np.corrcoef(np.roll(major_template, i), chroma_mean)[0, 1] for i in range(12)]
    minor_corr = [np.corrcoef(np.roll(minor_template, i), chroma_mean)[0, 1] for i in range(12)]

    best_major = keys[np.argmax(major_corr)]
    best_minor = minor_keys[np.argmax(minor_corr)]

    return best_major if max(major_corr) >= max(minor_corr) else best_minor

# ------------------------------------------------------------
# 🎬 자동 구조 분석 (마디 단위)
# ------------------------------------------------------------
def analyze_structure_measures(y, sr, n_sections=4, bpm=None):
    """
    자동 마디 감지 + 구조 분석
    - y: 오디오 신호
    - sr: 샘플링 레이트
    - bpm: 기본 템포 (없으면 자동 추출)
    - n_sections: 자동 섹션 개수
    """
    if bpm is None:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo) if tempo > 0 else 120

    # 4/4 기준 1마디 길이
    measure_duration = 60 / bpm * 4

    # MFCC 특징
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc = librosa.util.normalize(mfcc)

    # Beat Tracking
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    if len(beats) < n_sections:
        return None, measure_duration

    # Beat Sync MFCC
    beat_features = librosa.util.sync(mfcc, beats, aggregate=np.mean).T
    n_clusters = min(n_sections, len(beat_features))

    # KMeans 클러스터링
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(beat_features)

    # Beat → Time → Measure 변환
    times = librosa.frames_to_time(beats, sr=sr)
    min_len = min(len(times), len(labels))

    section_labels = ["A", "B", "C", "D", "E"]
    results = []

    for i in range(min_len - 1):
        start_sec = times[i]
        end_sec = times[i + 1]
        start_measure = int(start_sec / measure_duration) + 1
        end_measure = int(end_sec / measure_duration) + 1
        results.append((section_labels[labels[i]], start_measure, end_measure))

    return results, measure_duration

# ------------------------------------------------------------
# 🎨 마디 기반 구조 시각화
# ------------------------------------------------------------
def plot_song_structure_measures(sections):
    fig, ax = plt.subplots(figsize=(14, 2))
    colors = plt.cm.tab20.colors
    y = 0.5
    for i, (name, start, end) in enumerate(sections):
        ax.add_patch(
            patches.Rectangle(
                (start, y - 0.3),
                end - start,
                0.6,
                color=colors[i % len(colors)],
                alpha=0.9
            )
        )
        ax.text(
            (start + end) / 2,
            y,
            name,
            ha='center',
            va='center',
            color='white',
            fontsize=10,
            fontweight='bold'
        )
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max([end for _, _, end in sections]) + 1)
    ax.set_xlabel("Measures (마디)")
    ax.set_yticks([])
    ax.set_title("Song Structure by Measures")
    plt.tight_layout()
    st.pyplot(fig)

# ------------------------------------------------------------
# 🎚 분석 실행
# ------------------------------------------------------------
if uploaded_file is not None:
    st.success("파일 업로드 완료! 분석 시작 🔍")

    y, sr = librosa.load(uploaded_file, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    bpm = float(tempo) if tempo > 0 else None

    if bpm:
        measures = round(duration / (60 / bpm))
    else:
        measures = "계산 불가"

    key_result = detect_key(y, sr)

    st.markdown("## 📌 분석 결과")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**🎵 Key:** {key_result}")
    with col2:
        st.write(f"**⏱ BPM:** {round(bpm) if bpm else '추출 실패'}")

    st.write(f"**📏 Measures (마디 수):** {measures}")
    st.markdown("---")

    # 🌊 Waveform
    st.markdown("## 🌊 Waveform")
    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    st.pyplot(fig)

    # 🔥 Spectrogram
    st.markdown("## 🔥 Spectrogram")
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax2)
    fig2.colorbar(img, ax=ax2, format="%+2.f dB")
    ax2.set_title("Mel Spectrogram")
    st.pyplot(fig2)

    # ------------------------------------------------------------
    # 🎬 자동 구조 분석 + 마디 기반 시각화
    # ------------------------------------------------------------
    sections, measure_duration = analyze_structure_measures(y, sr, n_sections=5, bpm=bpm)
    if sections is None:
        st.warning("비트가 충분하지 않아 구조 분석 불가")
    else:
        plot_song_structure_measures(sections)

    # ------------------------------------------------------------
    # 🧩 AiR/코드 분석 연동 구조 (예시)
    # ------------------------------------------------------------
    st.markdown("## 🎹 코드 진행 / AiR 분석 (예시)")
    st.info("추후 AiR 분석 라이브러리 또는 코드 진행 추출 기능 연동 가능")
