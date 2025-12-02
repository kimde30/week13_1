import streamlit as st
import numpy as np
import librosa
import soundfile as sf

# -----------------------------------------
# 🎵 고급 Key Detection (Krumhansl-Schmuckler Algorithm 기반)
# -----------------------------------------

MAJOR_PROFILES = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88
])

MINOR_PROFILES = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 3.98, 2.69, 3.34, 3.17
])

KEYS = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]

def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_vector = np.mean(chroma, axis=1)

    major_corr = np.zeros(12)
    minor_corr = np.zeros(12)

    for i in range(12):
        major_corr[i] = np.corrcoef(chroma_vector, np.roll(MAJOR_PROFILES, i))[0, 1]
        minor_corr[i] = np.corrcoef(chroma_vector, np.roll(MINOR_PROFILES, i))[0, 1]

    best_major = np.argmax(major_corr)
    best_minor = np.argmax(minor_corr)

    if major_corr[best_major] >= minor_corr[best_minor]:
        return f"{KEYS[best_major]} Major"
    else:
        return f"{KEYS[best_minor]} Minor"

# -----------------------------------------
# 🎵 BPM 안정 추출
# -----------------------------------------
def detect_bpm(y, sr):
    try:
        bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
        if bpm <= 0 or np.isnan(bpm) or np.isinf(bpm):
            return None
        return float(bpm)
    except:
        return None

# -----------------------------------------
# 🎵 마디 계산 (오류 0% 안전 버전)
# -----------------------------------------
def safe_measures(bpm, duration):
    if bpm is None or duration is None:
        return None
    if bpm <= 0 or duration <= 0:
        return None
    try:
        return round(duration / (60 / bpm))
    except:
        return None

# -----------------------------------------
# 🌈 Streamlit UI
# -----------------------------------------
st.set_page_config(page_title="Music Analyzer", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #6C63FF;'>
        🎵 Music Analyzer (MP3) 
    </h1>
    <p style='text-align: center; color: #999;'>
        BPM · Key(Major/Minor) · Length · Measures 분석
    </p>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("🎧 MP3 파일을 업로드하세요", type=["mp3", "wav", "flac"])

if uploaded_file:
    st.success("파일 업로드 완료! 분석 중입니다...")

    # Read file
    y, sr = librosa.load(uploaded_file, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # BPM
    bpm = detect_bpm(y, sr)

    # KEY
    key = detect_key(y, sr)

    # Measures
    measures = safe_measures(bpm, duration)

    # -----------------------------------------
    # 출력 UI
    # -----------------------------------------
    st.subheader("📊 분석 결과")

    st.write(f"**⏱ BPM:** {round(bpm) if bpm else '추출 실패'}")
    st.write(f"**🎼 Key (Major/Minor):** {key}")
    st.write(f"**⏳ 전체 길이:** {round(duration, 2)} 초")
    st.write(f"**📏 전체 마디 수:** {measures if measures else '계산 불가'}")

    st.markdown("---")
    st.audio(uploaded_file, format="audio/mp3")

