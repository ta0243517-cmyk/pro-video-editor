import asyncio, os, re, subprocess, tempfile
import streamlit as st
from PIL import Image, ImageFilter, ImageOps

MAX_SEC = 600
RATIOS = {
    "YouTube / Facebook / X (16:9)": (16, 9),
    "TikTok / Reels / Shorts (9:16)": (9, 16),
    "Instagram square (1:1)": (1, 1),
    "Instagram / Facebook portrait (4:5)": (4, 5),
}
VOICES = {
    "Guy - male (US)": "en-US-GuyNeural",
    "Andrew - male (US)": "en-US-AndrewNeural",
    "Brian - male (US)": "en-US-BrianNeural",
    "Christopher - male (US)": "en-US-ChristopherNeural",
    "Ryan - male (UK)": "en-GB-RyanNeural",
    "Jenny - female (US)": "en-US-JennyNeural",
    "Aria - female (US)": "en-US-AriaNeural",
    "Ava - female (US)": "en-US-AvaNeural",
    "Emma - female (US)": "en-US-EmmaNeural",
    "Sonia - female (UK)": "en-GB-SoniaNeural",
}
M_A, M_B, M_C = "Audio + Images", "Audio + Script + Images", "Script + Images (app makes the voice)"
FITS = ["Blur background", "Fit (full image, black bars)", "Crop (fill screen)"]


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-800:])


def dur(p):
    o = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", p], capture_output=True, text=True).stdout
    return float(o.strip())


def nat(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def dims(ratio, q):
    rw, rh = ratio
    w, h = (q * rw / rh, q) if rw >= rh else (q, q * rh / rw)
    return int(w) // 2 * 2, int(h) // 2 * 2


def parse_script(text):
    if re.search(r"(?im)^\s*scene\s*\d+", text):
        parts = re.split(r"(?im)^\s*scene\s*\d+\s*[:.\-]?", text)
    else:
        parts = text.splitlines()
    return [" ".join(p.split()) for p in parts if p.strip()]


def srt_t(t):
    ms = int(round(t * 1000))
    return f"{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}"


def prep_image(src, dst, W, H, fit):
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    if fit.startswith("Crop"):
        out = ImageOps.fit(im, (W, H), Image.LANCZOS)
    else:
        fg = ImageOps.contain(im, (W, H), Image.LANCZOS)
        if fit.startswith("Blur"):
            bg = ImageOps.fit(im, (W // 4, H // 4)).filter(ImageFilter.GaussianBlur(8)).resize((W, H))
        else:
            bg = Image.new("RGB", (W, H))
        bg.paste(fg, ((W - fg.width) // 2, (H - fg.height) // 2))
        out = bg
    out.save(dst, quality=92)


async def synth(text, voice, rate, path):
    import edge_tts
    try:
        c = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    except TypeError:
        c = edge_tts.Communicate(text, voice, rate=rate)
    wb = []
    with open(path, "wb") as f:
        async for ch in c.stream():
            if ch["type"] == "audio":
                f.write(ch["data"])
            elif ch["type"] == "WordBoundary":
                wb.append(ch["offset"] / 1e7)
    return wb


def tts_scenes(scenes, voice, rate, tmp, prog):
    batches, cur, n = [], [], 0
    for s in scenes:
        if cur and n + len(s) > 2500:
            batches.append(cur); cur, n = [], 0
        cur.append(s); n += len(s) + 1
    batches.append(cur)
    starts, offset, parts = [], 0.0, []
    for bi, b in enumerate(batches):
        mp3 = os.path.join(tmp, f"tts_{bi}.mp3")
        for attempt in range(3):
            try:
                wb = asyncio.run(synth(" ".join(b), voice, rate, mp3)); break
            except Exception:
                if attempt == 2: raise
        d = dur(mp3)
        counts = [max(1, len(s.split())) for s in b]
        tot, acc = sum(counts), 0
        for j, c in enumerate(counts):
            if j == 0:
                starts.append(offset)
            else:
                t = wb[min(int(acc / tot * len(wb)), len(wb) - 1)] if wb else d * acc / tot
                starts.append(offset + t)
            acc += c
        offset += d; parts.append(mp3)
        prog.progress((bi + 1) / len(batches), "Making voice...")
    lst = os.path.join(tmp, "tts.txt")
    open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
    out = os.path.join(tmp, "tts.wav")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-ar", "44100", "-ac", "1", out])
    ends = starts[1:] + [offset]
    return out, list(zip(starts, ends))


def whisper_times(audio, scenes, total):
    from faster_whisper import WhisperModel
    segs, _ = WhisperModel("base.en", compute_type="int8").transcribe(
        audio, language="en", word_timestamps=True)
    words = [w for s in segs for w in s.words]
    if not words:
        return equal_times(len(scenes), total)
    counts = [max(1, len(s.split())) for s in scenes]
    tot, acc, b = sum(counts), 0, [0.0]
    for c in counts[:-1]:
        acc += c
        b.append(words[min(int(acc / tot * len(words)), len(words) - 1)].start)
    b.append(total)
    return list(zip(b[:-1], b[1:]))


def equal_times(n, total):
    return [(i * total / n, (i + 1) * total / n) for i in range(n)]


# ---------------- UI ----------------
st.title("Auto Video Editor")
mode = st.radio("What do you have?", [M_A, M_B, M_C])

audio_f = None
if mode != M_C:
    audio_f = st.file_uploader("Audio (English)", type=["mp3", "wav", "m4a"])
script = ""
if mode != M_A:
    script = st.text_area("Script (write 'Scene 1', 'Scene 2'... or put one scene per line)", height=250)
voice = None
if mode == M_C:
    vname = st.selectbox("Voice", list(VOICES))
    voice = VOICES[vname]
    if st.button("Preview voice"):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "p.mp3")
            asyncio.run(synth("Hello! This is how my voice sounds in your video.", voice, "+0%", p))
            st.audio(open(p, "rb").read())

def sk(f):
    return ([int(x) for x in re.findall(r"\d+", f.name)][-1:] or [0], nat(f.name))


nb = st.number_input("Image batches (use 2+ if you made images from different accounts)", 1, 6, 1)
groups = []
for bi in range(int(nb)):
    g = st.file_uploader(f"Images - batch {bi+1}" if nb > 1 else "Images (upload all together)",
                         type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True,
                         key=f"imgs{bi}")
    groups.append(sorted(g or [], key=sk))
files = [f for g in groups for f in g]
if files:
    key = tuple((f.name, f.size) for f in files)
    if st.session_state.get("okey") != key:
        st.session_state.okey, st.session_state.order = key, list(range(len(files)))
    files = [files[i] for i in st.session_state.order]
    with st.expander(f"Check image order ({len(files)} images)"):
        s0 = st.number_input("Show from scene", 1, len(files), 1)
        cols = st.columns(3)
        for k in range(s0 - 1, min(s0 + 8, len(files))):
            cols[(k - s0 + 1) % 3].image(files[k].getvalue(), caption=f"{k+1}. {files[k].name}")
        a = st.number_input("Swap scene", 1, len(files), 1)
        b = st.number_input("with scene", 1, len(files), 1)
        if st.button("Swap"):
            o = st.session_state.order
            o[a - 1], o[b - 1] = o[b - 1], o[a - 1]
            st.rerun()

st.subheader("Video settings")
rname = st.selectbox("Platform / ratio", list(RATIOS))
q = st.radio("Quality", ["1080p (HD)", "720p (faster)"], horizontal=True)
fit = st.selectbox("If image does not match the ratio", FITS)
speed = st.slider("Voice speed", 0.8, 1.5, 1.0, 0.05)
enhance = st.checkbox("Voice enhance (noise reduce + even volume)", False)
remove_sil = False
if mode != M_C:
    remove_sil = st.checkbox("Remove silent parts", True)
    thr = st.slider("Silence threshold (dB)", -60, -20, -40)
    min_sil = st.slider("Remove silence longer than (sec)", 0.2, 2.0, 0.5)
caps = False
if mode != M_A:
    caps = st.checkbox("Burn captions into video", True)
    cpos = st.radio("Caption position", ["Bottom", "Middle"], horizontal=True)

if st.button("Make video"):
    try:
        scenes = parse_script(script) if mode != M_A else []
        if not files:
            st.error("Please upload images."); st.stop()
        if mode != M_C and not audio_f:
            st.error("Please upload audio."); st.stop()
        if mode != M_A:
            if not scenes:
                st.error("Script is empty."); st.stop()
            if len(scenes) != len(files):
                st.error(f"{len(scenes)} scenes in script but {len(files)} images. They must match.")
                st.stop()
        prog = st.progress(0.0, "Starting...")
        with tempfile.TemporaryDirectory() as tmp:
            af = []
            if mode == M_C:
                base, times = tts_scenes(scenes, voice, f"{int((speed-1)*100):+d}%", tmp, prog)
            else:
                base = os.path.join(tmp, "in" + os.path.splitext(audio_f.name)[1])
                open(base, "wb").write(audio_f.getvalue())
                if remove_sil:
                    af.append(f"silenceremove=start_periods=1:start_threshold={thr}dB:"
                              f"stop_periods=-1:stop_duration={min_sil}:stop_threshold={thr}dB:stop_silence=0.15")
                if speed != 1.0:
                    af.append(f"atempo={speed}")
            if enhance:
                af.append("highpass=f=80,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11")
            audio = os.path.join(tmp, "clean.wav")
            cmd = ["ffmpeg", "-y", "-i", base] + (["-af", ",".join(af)] if af else []) + \
                  ["-ar", "44100", "-ac", "1", audio]
            prog.progress(0.0, "Cleaning audio..."); run(cmd)
            total = dur(audio)
            if total > MAX_SEC + 5:
                st.error(f"Audio is {total/60:.1f} min. Max is 10 min."); st.stop()
            if mode == M_B:
                prog.progress(0.0, "Listening to audio (Whisper)...")
                times = whisper_times(audio, scenes, total)
            elif mode == M_A:
                times = equal_times(len(files), total)
            times = [(a, b) for a, b in times]
            times[-1] = (times[-1][0], total)

            W, H = dims(RATIOS[rname], 1080 if q.startswith("1080") else 720)
            lst = os.path.join(tmp, "list.txt")
            with open(lst, "w") as lf:
                for i, (f, (a, b)) in enumerate(zip(files, times)):
                    p = os.path.join(tmp, f"img_{i}.jpg")
                    prep_image(f, p, W, H, fit)
                    lf.write(f"file '{p}'\nduration {max(b - a, 0.05):.3f}\n")
                    if i % 10 == 0:
                        prog.progress((i + 1) / len(files), f"Preparing images {i+1}/{len(files)}")
                lf.write(f"file '{p}'\n")
            vf = "format=yuv420p"
            if caps:
                srt = os.path.join(tmp, "cap.srt")
                with open(srt, "w") as sf:
                    for i, (t, (a, b)) in enumerate(zip(scenes, times), 1):
                        sf.write(f"{i}\n{srt_t(a)} --> {srt_t(b)}\n{t}\n\n")
                fs, al = int(min(W, H) * 0.055), (2 if cpos == "Bottom" else 5)
                vf = (f"subtitles={srt}:original_size={W}x{H}:force_style="
                      f"'FontName=DejaVu Sans,FontSize={fs},Bold=1,Outline=3,Shadow=0,"
                      f"Alignment={al},MarginV={int(H*0.08)}',format=yuv420p")
            out = os.path.join(tmp, "out.mp4")
            prog.progress(1.0, "Rendering video (please wait, keep this page open)...")
            run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", audio,
                 "-vf", vf, "-r", "24", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-c:a", "aac", "-b:a", "192k", "-shortest", out])
            st.session_state.video = open(out, "rb").read()
        prog.empty()
        st.success("Done!")
    except Exception as e:
        st.error(f"Error: {e}")

if "video" in st.session_state:
    st.video(st.session_state.video)
    st.download_button("Download MP4", st.session_state.video, "video.mp4", "video/mp4")
