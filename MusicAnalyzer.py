# app.py
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
st.markdown("유튜브 링크를 넣으면 (1) 제목/업로더, (2) BPM, (3) Key(장/단), (4) 전체 마디 수(기본 4/4)를 계산합니다.")

url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
beats_per_bar = st.number_input("Beats per bar (마디당 박자)", value=4, min_value=1, step=1)

if st.button("Analyze") and url.strip():
    with st.spinner("Downloading audio and analyzing — 잠시만요..."):
        # 1) Download audio with yt-dlp (needs yt-dlp installed)
        tmpdir = tempfile.mkdtemp()
        out_template = os.path.join(tmpdir, "audio.%(ext)s")
        # try to fetch best audio and metadata as json
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "--write-info-json",
            "-o", out_template,
            url
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            st.error(f"yt-dlp 실패: {e.stderr.decode('utf-8', errors='ignore')[:300]}")
            raise SystemExit

        # find the downloaded wav and info.json
        wav_path = None
        info_json = None
        for f in os.listdir(tmpdir):
            if f.lower().endswith(".wav"):
                wav_path = os.path.join(tmpdir, f)
            if f.lower().endswith(".info.json"):
                info_json = os.path.join(tmpdir, f)

        title = None
        uploader = None
        if info_json and os.path.exists(info_json):
            try:
                with open(info_json, "r", encoding="utf-8") as jf:
                    meta = json.load(jf)
                    title = meta.get("title")
                    uploader = meta.get("uploader") or meta.get("uploader_id") or meta.get("uploader_url")
            except Exception:
                pass

        if not wav_path or not os.path.exists(wav_path):
            st.error("오디오를 다운받지 못했습니다.")
        else:
            # 2) Load audio with librosa (mono)
            y, sr = librosa.load(wav_path, sr=None, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)

            # 3) Tempo (BPM) estimation
            # librosa.beat.tempo returns array (can return multiple estimates). We'll take the median/first.
            try:
                tempo = float(librosa.beat.tempo(y=y, sr=sr, aggregate=None).mean())
            except Exception:
                # fallback using onset envelope
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                tempo = float(librosa.beat.tempo(onset_envelope=onset_env, sr=sr).mean())

            # 4) Key detection using chroma + Krumhansl-Schmuckler templates
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)  # 12-dim vector

            # Krumhansl-Schmuckler major/minor templates (classic)
            # source: common K-S profiles (normalized)
            major_profile = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
            minor_profile = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

            def detect_key(chroma_vector):
                scores = []
                for i in range(12):
                    # rotate templates for each root
                    maj = np.roll(major_profile, i)
                    minp = np.roll(minor_profile, i)
                    # correlation (dot product) as score
                    scores.append(("{} major".format(librosa.midi_to_note(60+i, octave=False)), np.dot(chroma_vector, maj)))
                    scores.append(("{} minor".format(librosa.midi_to_note(60+i, octave=False)), np.dot(chroma_vector, minp)))
                # choose best
                best = max(scores, key=lambda x: x[1])
                return best

            # helper to convert note names like 'C' 'C#' ... using chroma index 0 = C
            # librosa.midi_to_note returns like 'C4' — above used just to get pitch letter, but to be safe build mapping
            pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            scores = []
            for i in range(12):
                maj = np.roll(major_profile, i)
                minp = np.roll(minor_profile, i)
                scores.append((f"{pitch_names[i]} major", float(np.dot(chroma_mean, maj))))
                scores.append((f"{pitch_names[i]} minor", float(np.dot(chroma_mean, minp))))
            best_key, best_score = max(scores, key=lambda x: x[1])

            # 5) Bars calculation
            total_beats = duration * tempo / 60.0
            bars = total_beats / float(beats_per_bar) if beats_per_bar > 0 else None

            # Display results
            st.subheader("Result")
            st.write("**Title:**", title or "Unknown")
            st.write("**Uploader / Artist:**", uploader or "Unknown")
            st.write("**Duration:**", f"{duration:.1f} sec")
            st.write("**Estimated BPM (tempo):**", f"{tempo:.2f}")
            st.write("**Estimated Key:**", best_key)
            if bars:
                st.write("**Estimated total bars (assuming {} beats/bar):** {:.1f}".format(beats_per_bar, bars))
            else:
                st.write("**Estimated total bars:** unknown (invalid beats per bar)")

            st.info("참고: 키/템포 추정은 자동화 추정값입니다. 복잡한 곡(변속, 변조, 복수 박자 등)은 오차가 있을 수 있습니다.")

    # cleanup files if desired
    # shutil.rmtree(tmpdir, ignore_errors=True)
