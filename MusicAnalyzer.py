import streamlit as st
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        MP3 파일을 업로드하면 BPM, Key, 스펙트럼 등을 자동 분석해줍니다!
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
# 🎼 Key Detection Function (정확도 강화 버전)
# ------------------------------------------------------------
def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    keys = [
        "C", "C#", "D", "D#", "E", "F",
        "F#", "G", "G#", "A", "A#", "B"
    ]
    minor_keys = [k + "m" for k in keys]

    # major/minor 템플릿 비교
    major_template = np.array(
        [1, 0.1, 0.8, 0.1, 1, 1, 0.1, 1, 0.1, 0.8, 0.1, 0.8]
    )
    minor_template = np.array(
        [1, 0.1, 0.8, 1, 0.1, 1, 1, 0.1, 1, 0.1, 0.8, 0.1]
    )

    major_corr = [np.corrcoef(np.roll(major_template, i), chroma_mean)[0, 1] for i in range(12)]
    minor_corr = [np.corrcoef(np.roll(minor_template, i), chroma_mean)[0, 1] for i in range(12)]

    best_major = keys[np.argmax(major_corr)]
    best_minor = minor_keys[np.argmax(minor_corr)]

    return best_major if max(major_corr) >= max(minor_corr) else best_minor

# ------------------------------------------------------------
# 🎚 분석 실행
# ------------------------------------------------------------
if uploaded_file is not None:
    st.success("파일 업로드 완료! 분석 시작합니다 🔍")

    # ------------------------------------------------------------
    # 🔊 오디오 로드
    # ------------------------------------------------------------
    y, sr = librosa.load(uploaded_file, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # ------------------------------------------------------------
    # 🎧 BPM 분석
    # ------------------------------------------------------------
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(tempo) if tempo > 0 else None
    except:
        bpm = None

    # measure 계산 (bpm이 있을 때만)
    if bpm:
        measures = round(duration / (60 / bpm))
    else:
        measures = "계산 불가"

    # ------------------------------------------------------------
    # 🎼 Key 분석
    # ------------------------------------------------------------
    key_result = detect_key(y, sr)

    # ------------------------------------------------------------
    # 📊 출력
    # ------------------------------------------------------------
    st.markdown("## 📌 분석 결과")

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**🎵 Key:** {key_result}")

    with col2:
        st.write(f"**⏱ BPM:** {round(bpm) if bpm else '추출 실패'}")

    st.write(f"**📏 Measures (마디 수):** {measures}")

    st.markdown("---")

    # ------------------------------------------------------------
    # 📈 Waveform Plot
    # ------------------------------------------------------------
    st.markdown("## 🌊 Waveform")

    fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    st.pyplot(fig)

    # ------------------------------------------------------------
    # 🔥 Spectrogram
    # ------------------------------------------------------------
    st.markdown("## 🔥 Spectrogram")

    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax2)
    fig2.colorbar(img, ax=ax2, format="%+2.f dB")
    ax2.set_title("Mel Spectrogram")
    st.pyplot(fig2)

