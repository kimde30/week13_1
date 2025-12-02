# app.py (patched for yt-dlp no-JS runtime errors)
import streamlit as st
import tempfile
import subprocess
import os
import json
import numpy as np
import librosa
import soundfile as sf

st.set_page_config(page_title="YouTube → Title/Artist/BPM/Key/Bars", layout="centered")

st.title("🎵 YouTube → Title · Artist · BPM · Key · Bars")

url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
beats_per_bar = st.number_input("Beats per bar", value=4, min_value=1)

if st.button("Analyze") and url.strip():
    with st.spinner("Downloading & analyzing..."):
        tmpdir = tempfile.mkdtemp()
        out_template = os.path.join(tmpdir, "audio.%(ext)s")

        # ---- IMPORTANT PATCH: no-JS safe options ----
        cmd = [
            "yt-dlp",
            "--no-exec",
            "--compat-options", "no-js",
            "--extractor-args", "youtube:nojs=1",
            "--no-playlist",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--write-info-json",
            "-o", out_template,
            url
        ]
        # -----------------------------------------------

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            st.error(f"yt-dlp 오류 발생:\n\n{e.stderr.decode('utf-8', errors='ignore')}")
            raise SystemExit

        wav_path, info_json = None, None
        for f in os.listdir(tmpdir):
            if f.endswith(".wav"):
                wav_path = os.path.join(tmpdir, f)
            elif f.endswith(".info.json"):
                info_json = os.path.join(tmpdir, f)

        title = "Unknown"
        uploader = "Unknown"

        if info_json:
            with open(info_json, "r", encoding="utf-8") as jf:
                meta = json.load(jf)
                title = meta.get("title", "Unknown")
                uploader = meta.get("uploader") or meta.get("artist") or "Unknown"

        if not wav_path:
            st.error("오디오 추출 실패 — JS runtime 없이 추출 가능한 포맷을 찾지 못했습니다.")
            raise SystemExit

        # ----- Audio analysis -----
        y, sr = librosa.load(wav_path, sr=None, mono=True)
        duration = librosa.get_duration(y=y, sr=sr)

        tempo = float(librosa.beat.tempo(y=y, sr=sr, aggregate=None).mean())

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

        total_beats = duration * tempo / 60
        bars = total_beats / beats_per_bar

        st.subheader("결과")
        st.write("**Title:**", title)
        st.write("**Artist/Uploader:**", uploader)
        st.write("**Duration:**", f"{duration:.1f} sec")
        st.write("**BPM:**", f"{tempo:.2f}")
        st.write("**Key:**", best_key)
        st.write("**Estimated bars:**", f"{bars:.1f}")

        st.info("※ 자동 분석이므로 BPM/Key는 오차가 있을 수 있습니다.")
