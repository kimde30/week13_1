import streamlit as st
import librosa
import numpy as np
from mutagen.mp3 import MP3
from mutagen.id3 import ID3
import tempfile


# ---------------------- Key Detection (Improved) ------------------------

MAJOR_PROFILE = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88
])

MINOR_PROFILE = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 3.98, 2.69, 3.34, 3.17
])


def detect_key_advanced(y, sr):
    try:
        # 1) Harmonic component only
        y_harmonic = librosa.effects.harmonic(y)

        # 2) Chroma CQT
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        chroma_mean = chroma.mean(axis=1)

        # 3) Normalization
        chroma_norm = chroma_mean / chroma_mean.sum()

        max_corr = -999
        best_key = None
        mode = None

        KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F',
                'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Compare against rotated key profiles
        for i in range(12):
            corr_major = np.corrcoef(np.roll(MAJOR_PROFILE, i), chroma_norm)[0, 1]
            corr_minor = np.corrcoef(np.roll(MINOR_PROFILE, i), chroma_norm)[0, 1]

            if corr_major > max_corr:
                max_corr = corr_major
                best_key = KEYS[i]
                mode = "Major"

            if corr_minor > max_corr:
                max_corr = corr_minor
                best_key = KEYS[i]
                mode = "Minor"

        return f"{best_key} {mode}"
    except:
        return None


# ---------------------- Metadata Extraction ----------------------------

def get_metadata_mp3(file_path):
    try:
        audio = MP3(file_path)
        duration = audio.info.length
    except:
        duration = None

    title, artist = None, None
    try:
        tags = ID3(file_path)
        if tags.get("TIT2"):
            title = tags.get("TIT2").text[0]
        if tags.get("TPE1"):
            artist = tags.get("TPE1").text[0]
    except:
        pass

    return title, artist, duration


# ---------------------- Safe Measure Calculation -----------------------

def safe_calculate_measures(bpm, duration):
    # Invalid types or missing values → None
    if bpm is None or duration is None:
        return None
    if not isinstance(bpm, (float, int)):
        return None
    if not isinstance(duration, (float, int)):
        return None
    if bpm <= 0 or duration <= 0:
        return None
    if np.isnan(bpm) or np.isnan(duration):
        return None
    if np.isinf(bpm) or np.isinf(duration):
        return None

    # Safe calculation
    try:
        measures_value = duration / (60 / bpm)
        return round(measures_value)
    except:
        return None


# ---------------------- Streamlit UI ----------------------------------

st.set_page_config(page_title="🎵 Music Analyzer", layout="centered")

# Custom CSS
st.markdown("""
<style>
    .result-card {
        background: #1f2937;
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin-top: 20px;
    }
    .title {
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #10b981;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🎵 MP3 음악 분석기</h1>", unsafe_allow_html=True)
st.write("MP3 파일을 업로드하면 **제목, 가수, BPM, Key, 전체 마디 수**를 분석합니다.")


# ---------------------- File Upload -----------------------------------

uploaded_file = st.file_uploader("MP3 파일 업로드", type=["mp3"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("분석 중입니다... 🎧 잠시만 기다려주세요."):
        # Metadata
        title, artist, duration = get_metadata_mp3(tmp_path)

        # Audio load
        try:
            y, sr = librosa.load(tmp_path, sr=None)
        except Exception as e:
            st.error(f"오디오 불러오기 실패: {e}")
            st.stop()

        # BPM detection
        bpm, _ = librosa.beat.beat_track(y=y, sr=sr)

        # Key detection
        key = detect_key_advanced(y, sr)

        # Measures (error-safe)
        measures = safe_calculate_measures(bpm, duration)

    # ---------------------- Result Card UI ----------------------

    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.subheader("📌 분석 결과")

    st.write(f"**🎼 제목:** {title or '알 수 없음'}")
    st.write(f"**🎤 가수:** {artist or '알 수 없음'}")
    st.write(f"**⏱ BPM:** {round(bpm) if bpm else '추출 실패'}")
    st.write(f"**🎹 Key (조성):** {key or '추출 실패'}")
    st.write(f"**⏳ 전체 길이:** {round(duration, 2)} 초" if duration else "**⏳ 전체 길이:** 알 수 없음")

    if measures is not None:
        st.write(f"**📏 전체 마디 수:** {measures} 마디")
    else:
        st.write("**📏 전체 마디 수:** 계산 불가 (BPM 또는 길이 정보 부족)")

    st.markdown("</div>", unsafe_allow_html=True)
