import streamlit as st
import numpy as np
import librosa
import soundfile as sf

# ============================================================
# 🎼 KEY DETECTION — Ensemble (KS + Peak + TCS)
# ============================================================

MAJOR_PROFILE = np.array([
    6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
    2.52, 5.19, 2.39, 3.66, 2.29, 2.88
])
MINOR_PROFILE = np.array([
    6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
    2.54, 4.75, 3.98, 2.69, 3.34, 3.17
])
KEYS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

# 1) KS Key Detector
def ks_key(chroma_vector):
    major_corr = []
    minor_corr = []
    
    for i in range(12):
        major_corr.append(np.corrcoef(chroma_vector, np.roll(MAJOR_PROFILE, i))[0,1])
        minor_corr.append(np.corrcoef(chroma_vector, np.roll(MINOR_PROFILE, i))[0,1])

    major_key = np.argmax(major_corr)
    minor_key = np.argmax(minor_corr)

    if major_corr[major_key] >= minor_corr[minor_key]:
        return KEYS[major_key] + " Major"
    else:
        return KEYS[minor_key] + " Minor"

# 2) Weighted Chroma Peak Method
def chroma_peak_key(chroma):
    peaks = np.argmax(chroma, axis=0)
    hist = np.bincount(peaks, minlength=12)

    major_score = np.dot(hist, MAJOR_PROFILE)
    minor_score = np.dot(hist, MINOR_PROFILE)

    base_note = np.argmax(hist)

    if major_score >= minor_score:
        return KEYS[base_note] + " Major"
    else:
        return KEYS[base_note] + " Minor"

# 3) Tonality Candidate Search (TCS)
def tcs_key(chroma):
    scores_major = np.zeros(12)
    scores_minor = np.zeros(12)

    for i in range(12):
        profile_major = np.roll(MAJOR_PROFILE, i)
        profile_minor = np.roll(MINOR_PROFILE, i)

        scores_major[i] = np.sum(chroma * profile_major[:, None])
        scores_minor[i] = np.sum(chroma * profile_minor[:, None])

    if scores_major.max() >= scores_minor.max():
        return KEYS[np.argmax(scores_major)] + " Major"
    else:
        return KEYS[np.argmax(scores_minor)] + " Minor"

# 4) Ensemble Key Detector
def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_vector = np.mean(chroma, axis=1)

    k1 = ks_key(chroma_vector)
    k2 = chroma_peak_key(chroma)
    k3 = tcs_key(chroma)

    votes = [k1, k2, k3]
    final = max(set(votes), key=votes.count)

    if votes.count(final) == 1:
        return k1
    return final


# ============================================================
# 🎵 BPM DETECTOR
# ============================================================
def detect_bpm(y, sr):
    try:
        bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
        if bpm <= 0 or np.isnan(bpm) or np.isinf(bpm):
            return None
        return float(bpm)
    except:
        return None

# ============================================================
# 📏 MEASURES (SAFE)
# ============================================================
def safe_measures(bpm, duration):
    if bpm is None or duration is None:
        return None
    if bpm <= 0 or duration <= 0:
        return None
    try:
        return round(duration / (60 / bpm))
    except:
        return None

# ============================================================
# 🌈 STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Music Analyzer", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #6C63FF;'>
        🎵 Music Analyzer (MP3)
    </h1>
    <p style='text-align: center; color: #aaa;'>
        BPM · Key(Major/Minor) · Length · Measures 분석
    </p>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("🎧 MP3 파일을 업로드하세요", type=["mp3", "wav", "flac"])

if uploaded_file:
    st.success("파일 업로드 완료! 분석을 시작합니다...")

    # 오디오 로드
    y, sr = librosa.load(uploaded_file, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # 분석
    bpm = detect_bpm(y, sr)
    key = detect_key(y, sr)
    measures = safe_measures(bpm, duration)

    # 출력
    st.subheader("📊 분석 결과")

    st.write(f"**⏱ BPM:** {round(bpm) if bpm else '추출 실패'}")
    st.write(f"**🎼 Key (Major/Minor):** {key}")
    st.write(f"**⏳ 전체 길이:** {round(duration, 2)} 초")
    st.write(f"**📏 전체 마디 수:** {measures if measures else '계산 불가'}")

    st.markdown("---")
    st.audio(uploaded_file, format="audio/mp3")
