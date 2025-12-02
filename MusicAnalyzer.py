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
    page_title="🎶 Music Analyzer",
    page_icon="🎧",
    layout="centered"
)

st.markdown(
    """
    <h1 style='text-align: center; color:#6C63FF;'>🎶 Music Analyzer</h1>
    <p style='text-align: center; color:#555; font-size:17px;'>
        MP3 파일을 업로드하면 BPM, Key, 스펙트럼, 곡 구조 등을 자동 분석해줍니다!
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
# 🎬 곡 구조 분석 함수
# ------------------------------------------------------------
def analyze_structure(y, sr, n_sections=4):
    hop = 1024
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc = librosa.util.normalize(mfcc)

    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)

    if len(beats) < n_sections:
        return None

    beat_features = librosa.util.sync(mfcc, beats, aggregate=np.mean)
    beat_features = beat_features.T

    n_clusters = min(n_sections, len(beat_features))

    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    labels = kmeans.fit_predict(beat_features)

    times = librosa.frames_to_time(beats, sr=sr)
    min_len = min(len(times), len(labels))

    section_labels = ["A", "B", "C", "D", "E"]
    results = []

    for i in range(min_len - 1):
        results.append((section_labels[labels[i]], times[i], times[i + 1]))

    return results


# ------------------------------------------------------------
# 🎚 분석 실행
# ------------------------------------------------------------
if uploaded_file is not None:
    st.success("파일 업로드 완료! 분석 시작합니다 🔍")

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
    # 🎬 구조 분석
    # ------------------------------------------------------------
    st.markdown("## 🎬 Song Structure (A/B/C Parts)")

    sections = analyze_structure(y, sr)

    if sections is None:
        st.warning("비트가 충분히 감지되지 않아 구조 분석이 불가능합니다.")
    else:
        for part, start, end in sections:
            st.write(f"**{part}** : {start:5.1f}초 → {end:5.1f}초")

        # ------------------------------------------------------------
        # 🎨 구조 시각화 그래프 추가
        # ------------------------------------------------------------
        st.markdown("## 🎨 Song Structure Timeline")

        fig3, ax3 = plt.subplots(figsize=(12, 2))

        colors = {
            "A": "#6C63FF",
            "B": "#FF6584",
            "C": "#4CD3C2",
            "D": "#FFA500",
            "E": "#00A8FF"
        }

        for part, start, end in sections:
            width = end - start
            ax3.add_patch(
                patches.Rectangle(
                    (start, 0),
                    width,
                    1,
                    color=colors.get(part, "gray"),
                    alpha=0.7
                )
            )
            ax3.text(start + width/2, 0.5, part, ha='center', va='center', fontsize=12, color='white')

        ax3.set_xlim(0, duration)
        ax3.set_ylim(0, 1)
        ax3.set_yticks([])
        ax3.set_xlabel("Time (sec)")
        ax3.set_title("Song Structure Timeline")

        st.pyplot(fig3)
