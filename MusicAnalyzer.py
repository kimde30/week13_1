import streamlit as st
import librosa
import numpy as np
from mutagen.mp3 import MP3
from mutagen.id3 import ID3
import tempfile


def detect_bpm(y, sr):
    try:
        bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(bpm)
    except:
        return None


def detect_key(y, sr):
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F',
                'F#', 'G', 'G#', 'A', 'A#', 'B']
        return keys[int(np.argmax(chroma_mean))]
    except:
        return None


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


# ------------------------------------------------------------------------------

st.title("🎵 MP3 음원 자동 분석기")
st.write("MP3 파일을 업로드하면 제목, 가수, BPM, Key, 전체 마디 수를 분석합니다.")

uploaded_file = st.file_uploader("MP3 파일 업로드", type=["mp3"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.success("파일 업로드 완료!")

    # 1. Metadata
    title, artist, duration = get_metadata_mp3(tmp_path)

    # 2. 오디오 로드
    try:
        y, sr = librosa.load(tmp_path)
    except Exception as e:
        st.error(f"오디오 로드 실패: {e}")
        st.stop()

    # 3. BPM
    bpm = detect_bpm(y, sr)

    # 4. Key
    key = detect_key(y, sr)

    # 5. 마디 수 계산(오류 방지)
    if bpm and bpm > 0 and duration and duration > 0:
        try:
            measures = duration / (60 / bpm)
            measures = round(measures)
        except:
            measures = None
    else:
        measures = None

    # 출력
    st.subheader("분석 결과")

    st.write(f"**제목:** {title or '알 수 없음'}")
    st.write(f"**가수:** {artist or '알 수 없음'}")
    st.write(f"**BPM:** {bpm if bpm else '추출 실패'}")
    st.write(f"**Key(조성):** {key if key else '추출 실패'}")
    st.write(f"**전체 길이:** {round(duration,2) if duration else '알 수 없음'}")

    if measures:
        st.write(f"**전체 마디 수:** {measures} 마디")
    else:
        st.write("**전체 마디 수:** 계산 불가 (BPM 또는 길이 정보 부족)")
