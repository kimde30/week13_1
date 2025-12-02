import streamlit as st
import librosa
import numpy as np
from mutagen.mp3 import MP3
from mutagen.id3 import ID3
import tempfile


def detect_bpm(y, sr):
    bpm, _ = librosa.beat.beat_track(y=y, sr=sr)
    return bpm


def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    keys = ['C', 'C#', 'D', 'D#', 'E', 'F',
            'F#', 'G', 'G#', 'A', 'A#', 'B']
    key_index = np.argmax(chroma_mean)
    return keys[key_index]


def get_metadata_mp3(file_path):
    audio = MP3(file_path)
    title, artist = None, None

    try:
        tags = ID3(file_path)
        title = tags.get("TIT2").text[0] if tags.get("TIT2") else None
        artist = tags.get("TPE1").text[0] if tags.get("TPE1") else None
    except:
        pass

    return title, artist, audio.info.length


# Streamlit UI -------------------------------------------------------

st.title("🎵 MP3 음원 자동 분석기")
st.write("MP3 파일을 업로드하면 제목, 가수, BPM, Key, 전체 마디 수를 분석합니다.")

uploaded_file = st.file_uploader("MP3 파일 업로드", type=["mp3"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.success("파일 업로드 완료!")

    # 1. Metadata (제목, 가수, 재생 시간)
    title, artist, duration = get_metadata_mp3(tmp_path)

    # 2. Librosa 로 오디오 로드
    y, sr = librosa.load(tmp_path)

    # 3. BPM
    bpm = detect_bpm(y, sr)

    # 4. Key
    key = detect_key(y, sr)

    # 5. 마디 수 계산
    if bpm > 0:
        measures = duration / (60 / bpm)
        measures = round(measures)
    else:
        measures = "계산 불가"

    # 출력
    st.subheader("분석 결과")
    st.write(f"**제목:** {title if title else '알 수 없음'}")
    st.write(f"**가수:** {artist if artist else '알 수 없음'}")
    st.write(f"**BPM:** {bpm}")
    st.write(f"**Key(조성):** {key}")
    st.write(f"**전체 길이:** {round(duration, 2)} 초")
    st.write(f"**전체 마디 수:** {measures} 마디")
