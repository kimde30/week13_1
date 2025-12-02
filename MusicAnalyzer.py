# app.py (FINAL — yt-dlp no-JS runtime safe)
import streamlit as st
import tempfile
import subprocess
import os
import json
import numpy as np
import librosa


st.set_page_config(page_title="YouTube Analyzer", layout="centered")
st.title("🎵 YouTube → Title · Artist · BPM · Key · Bars")
st.write("유튜브 링크만 넣으면 자동 분석합니다.")

url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
beats_per_bar = st.number_input("Beats per bar", value=4, min_value=1)

if st.button("Analyze") and url.strip():
    with st.spinner("유튜브 오디오 다운로드 및 분석 중..."):
        tmpdir = tempfile.mkdtemp()
        out_template = os.path.join(tmpdir, "audio.%(ext)s")

        # Only safe option for no-JS environment
        cmd = [
            "yt-dlp",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--write-info-json",
            "--extractor-args", "youtube:player_skip=js",
            "--no-playlist",
            "-o", out_template,
            url
        ]

        try:
            r = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            st.error("yt-dlp 실행 실패:\n\n" + e.stderr.decode('utf-8', errors='ignore'))
            raise SystemExit

        wav_path, info_json = None, None
        for f in os.listdir(tmpdir):
            if f.endswith(".wav"):
                wav_path = os.path.join(tmpdir, f)
            elif f.endswith(".info.json"):
                info_json = os.path.join(tmpdir, f)

        # metadata
        title, artist = "Unknown", "Unknown"
        if info_json:
            with open(info_json, "r", encoding="utf-8") as jf:
                meta = json.load(jf)
                title = meta.get("title") or "Unknown"
                artist = meta.get("uploader") or meta.get("artist") or "Unknown"

        if not wav_path:
            st.error("오디오 다운로드에 실패했습니다. (JS runtime 없음)")
            raise SystemExit

        # audio analysis
        y, sr = librosa.load(wav_path, sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        tempo = float(librosa.beat.tempo(y=y, sr=sr).mean())

        # Key detection
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
        minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
        pitch = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

        scores = []
        for i in range(12):
            scores.append((f"{pitch[i]} major", np.dot(chroma_mean, np.roll(major_profile, i))))
            scores.append((f"{pitch[i]} minor", np.dot(chroma_mean, np.roll(minor_profile, i))))
        best_key = max(scores, key=lambda x: x[1])[0]

        beats = duration * tempo / 60
        bars = beats / beats_per_bar

        st.subheader("분석 결과")
        st.write("**Title:**", title)
        st.write("**Artist:**", artist)
        st.write(f"**Duration:** {duration:.1f} sec")
        st.write(f"**BPM:** {tempo:.2f}")
        st.write("**Key:**", best_key)
        st.write("**Estimated Bars:**", f"{bars:.1f}")
