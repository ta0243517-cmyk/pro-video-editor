import asyncio, bisect, math, os, random, re, subprocess, tempfile, wave
import cv2
import numpy as np
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
ZOOMS = ["Off", "Zoom in", "Zoom out", "Alternate in/out"]
TRANS = ["Cut (none)", "Fade", "Slide left", "Slide right", "Wipe", "Flash", "Random mix"]
EVERY = {"Every scene": 1, "Every 2nd scene": 2, "Every 3rd scene": 3, "Every 5th scene": 5}
SR = 22050
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
    if hasattr(src, "seek"):
        src.seek(0)
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
    if dst is None:
        return np.ascontiguousarray(np.asarray(out)[:, :, ::-1])
    out.save(dst, quality=95)


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
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", audio, "-ac", "1", "-ar", "16000",
                        "-f", "f32le", "-"], capture_output=True)
    if r.returncode:
        raise RuntimeError("Could not read audio for Whisper.")
    segs, _ = WhisperModel("base.en", compute_type="int8").transcribe(
        np.frombuffer(r.stdout, np.float32), language="en", word_timestamps=True)
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



# ---------------- effects: zoom, transitions, sound effects, music ----------------
def ease(p):
    p = min(max(p, 0.0), 1.0)
    return p * p * (3 - 2 * p)


def zval(i, t, a, b, mode, amt):
    if mode == "Off":
        return 1.0
    p = ease((t - a) / max(b - a, 1e-3))
    if mode == "Zoom in":
        return 1 + amt * p
    if mode == "Zoom out":
        return 1 + amt * (1 - p)
    return 1 + amt * (p if i % 2 == 0 else 1 - p)


def blend(A, B, p, kind):
    p = ease(p)
    H, W = A.shape[:2]
    x = int(W * p)
    if kind == "Fade":
        return cv2.addWeighted(A, 1 - p, B, p, 0)
    if kind == "Flash":
        white = np.full_like(A, 255)
        if p < 0.5:
            return cv2.addWeighted(A, 1 - 2 * p, white, 2 * p, 0)
        return cv2.addWeighted(white, 2 - 2 * p, B, 2 * p - 1, 0)
    out = np.empty_like(A)
    if kind == "Slide left":
        out[:, :W - x] = A[:, x:]; out[:, W - x:] = B[:, :x]
    elif kind == "Slide right":
        out[:, x:] = A[:, :W - x]; out[:, :x] = B[:, W - x:]
    else:  # Wipe
        out[:] = A; out[:, :x] = B[:, :x]
    return out


class SceneCache:
    def __init__(self, files, W, H, fit, s):
        self.files, self.W, self.H, self.fit = files, W, H, fit
        self.SW, self.SH = int(W * s) // 2 * 2, int(H * s) // 2 * 2
        self.cache = {}

    def get(self, i):
        if i not in self.cache:
            if len(self.cache) >= 4:
                self.cache.pop(next(iter(self.cache)))
            self.cache[i] = prep_image(self.files[i], None, self.SW, self.SH, self.fit)
        return self.cache[i]

    def view(self, i, z):
        img = self.get(i)
        cw, ch = min(int(self.SW / z) // 2 * 2, self.SW), min(int(self.SH / z) // 2 * 2, self.SH)
        x0, y0 = (self.SW - cw) // 2, (self.SH - ch) // 2
        crop = np.ascontiguousarray(img[y0:y0 + ch, x0:x0 + cw])
        if cw == self.W and ch == self.H:
            return crop
        return cv2.resize(crop, (self.W, self.H), interpolation=cv2.INTER_LINEAR)


def render_fx(files, times, kinds, W, H, fit, zmode, zamt, tkind, tdur, fps, total,
              audio, vf, out, log, prog):
    sc = SceneCache(files, W, H, fit, 1 + zamt if zmode != "Off" else 1.0)
    n, starts = len(times), [a for a, _ in times]

    def zv(i, t):
        return zval(i, t, times[i][0], times[i][1], zmode, zamt)

    def dk(k):
        return min(tdur, 0.5 * (times[k][1] - times[k][0]), 0.5 * (times[k + 1][1] - times[k + 1][0]))

    wr = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(fps), "-i", "-", "-i", audio, "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
         "-map", "0:v", "-map", "1:a", "-shortest", out], stdin=subprocess.PIPE, stderr=log)
    nf, last_i, last_bytes = int(round(total * fps)), -1, None
    for fi in range(nf):
        t = fi / fps
        i = min(max(bisect.bisect_right(starts, t) - 1, 0), n - 1)
        frame = None
        if tkind != "Cut (none)":
            if i + 1 < n:
                d = dk(i); s0 = times[i][1] - d / 2
                if d >= 0.04 and t >= s0:
                    frame = blend(sc.view(i, zv(i, t)), sc.view(i + 1, zv(i + 1, t)), (t - s0) / d, kinds[i])
            if frame is None and i > 0:
                d = dk(i - 1); s0 = times[i - 1][1] - d / 2
                if d >= 0.04 and t < s0 + d:
                    frame = blend(sc.view(i - 1, zv(i - 1, t)), sc.view(i, zv(i, t)), (t - s0) / d, kinds[i - 1])
        if frame is not None:
            data, last_i = frame.tobytes(), -1
        elif zmode == "Off" and i == last_i:
            data = last_bytes
        else:
            data = sc.view(i, zv(i, t)).tobytes()
            if zmode == "Off":
                last_i, last_bytes = i, data
        wr.stdin.write(data)
        if fi % 24 == 0:
            prog.progress(min(fi / nf, 1.0), f"Rendering video {fi}/{nf} frames (keep this page open)")
    wr.stdin.close(); wr.wait()
    if wr.returncode:
        raise RuntimeError("Video writer failed. " + open(log.name).read()[-400:])


def make_sfx():
    rng = np.random.default_rng(1)
    n = int(0.6 * SR); t = np.linspace(0, 1, n)
    noise = rng.standard_normal(n).astype(np.float32)
    coef = 0.02 + 0.5 * np.sin(np.pi * t) ** 2
    y, acc = np.zeros(n, np.float32), 0.0
    for i in range(n):
        acc += coef[i] * (noise[i] - acc); y[i] = acc
    y = y * np.sin(np.pi * t) ** 1.5
    whoosh = y / max(np.abs(y).max(), 1e-6)
    n = int(0.15 * SR); tt = np.arange(n) / SR
    pop = (np.sin(2 * np.pi * (600 + 500 * np.exp(-tt * 40)) * tt) * np.exp(-tt * 35)).astype(np.float32)
    return {"whoosh": (whoosh, 0.3), "pop": (pop, 0.0)}


def write_sfx(path, dur_s, events, vol):
    lib = make_sfx()
    total = int(dur_s * SR)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        for c0 in range(0, total, SR * 60):
            c1 = min(total, c0 + SR * 60)
            buf = np.zeros(c1 - c0, np.float32)
            for t, k in events:
                snd, lead = lib[k]
                s0 = int((t - lead) * SR); s1 = s0 + len(snd)
                if s1 <= c0 or s0 >= c1:
                    continue
                a, b = max(s0, c0), min(s1, c1)
                buf[a - c0:b - c0] += snd[a - s0:b - s0]
            w.writeframes((np.clip(buf * 0.9 * vol, -1, 1) * 32767).astype("<i2").tobytes())


def mix_audio(voice, sfx, music, total, mvol, duck, tmp):
    out = os.path.join(tmp, "mix.wav")
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", voice]
    fmt = "aformat=sample_rates=44100:channel_layouts=stereo"
    fl, mix, idx = [], ["[vm]"], 1
    fl.append(f"[0:a]{fmt},asplit=2[vm][vs]" if (music and duck) else f"[0:a]{fmt}[vm]")
    if sfx:
        cmd += ["-i", sfx]; fl.append(f"[{idx}:a]{fmt}[sx]"); mix.append("[sx]"); idx += 1
    if music:
        cmd += ["-stream_loop", "-1", "-i", music]
        fl.append(f"[{idx}:a]{fmt},volume={mvol},atrim=0:{total:.2f},"
                  f"afade=t=out:st={max(total - 3, 0):.2f}:d=3[mu]")
        if duck:
            fl.append("[mu][vs]sidechaincompress=threshold=0.03:ratio=10:attack=15:release=400[md]")
            mix.append("[md]")
        else:
            mix.append("[mu]")
    fl.append(f"{''.join(mix)}amix=inputs={len(mix)}:duration=first:normalize=0,alimiter=limit=0.95[out]")
    run(cmd + ["-filter_complex", ";".join(fl), "-map", "[out]", "-t", f"{total:.2f}",
               "-ar", "44100", out])
    return out


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

ORDERS = ["Auto (recommended)", "Number at start of file name", "Number at end of file name",
          "File name (A-Z)"]


def nums(f):
    return [int(x) for x in re.findall(r"\d+", f.name)] or [0]


def sort_files(g, how):
    if not g:
        return []
    first, last = [nums(f)[0] for f in g], [nums(f)[-1] for f in g]
    if how.startswith("Auto"):
        how = ORDERS[1] if len(set(first)) == len(g) else (ORDERS[2] if len(set(last)) == len(g) else ORDERS[3])
    if how == ORDERS[1]:
        return sorted(g, key=lambda f: (nums(f)[0], nat(f.name)))
    if how == ORDERS[2]:
        return sorted(g, key=lambda f: (nums(f)[-1], nat(f.name)))
    return sorted(g, key=lambda f: nat(f.name))


order_by = st.selectbox("Order images by", ORDERS)
nb = st.number_input("Image batches (use 2+ if you made images from different accounts)", 1, 6, 1)
groups = []
for bi in range(int(nb)):
    g = st.file_uploader(f"Images - batch {bi+1}" if nb > 1 else "Images (upload all together)",
                         type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True,
                         key=f"imgs{bi}")
    groups.append(sort_files(g or [], order_by))
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


st.subheader("Effects")
zmode = st.selectbox("Zoom on images", ZOOMS)
zamt = st.slider("Zoom strength (%)", 3, 20, 8) / 100 if zmode != "Off" else 0.0
tkind = st.selectbox("Transition between scenes", TRANS)
tdur = st.slider("Transition length (sec)", 0.2, 1.0, 0.4, 0.1) if tkind != "Cut (none)" else 0.0
sfx_kind = st.selectbox("Sound effect on scene changes", ["Off", "Whoosh", "Pop", "Mixed (auto)"])
sfx_vol, sfx_every = 0, 1
if sfx_kind != "Off":
    sfx_vol = st.slider("Sound effect volume (%)", 10, 100, 40)
    sfx_every = EVERY[st.selectbox("Play sound effect on", list(EVERY), index=1)]
music_f = st.file_uploader("Background music (optional)", type=["mp3", "wav", "m4a"])
mvol, duck = 18, True
if music_f:
    mvol = st.slider("Music volume (%)", 5, 60, 18)
    duck = st.checkbox("Lower music while the voice speaks", True)

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
            n = len(times)
            fx = zmode != "Off" or tkind != "Cut (none)"
            rng = random.Random(7)
            pool = ["Fade", "Slide left", "Slide right", "Wipe"]
            kinds = [rng.choice(pool) if tkind.startswith("Random") else tkind for _ in range(max(n - 1, 0))]

            final_audio, events = audio, []
            if sfx_kind != "Off" and n > 1:
                last = -99.0
                for k in range(n - 1):
                    if k % sfx_every:
                        continue
                    t = times[k][1]
                    if t - last < 0.6:
                        continue
                    if sfx_kind == "Whoosh":
                        kind = "whoosh"
                    elif sfx_kind == "Pop":
                        kind = "pop"
                    else:
                        kind = "whoosh" if kinds[k] in ("Slide left", "Slide right", "Wipe", "Flash") else "pop"
                    events.append((t, kind)); last = t
            sfx_path = mus_path = None
            if events:
                sfx_path = os.path.join(tmp, "sfx.wav")
                write_sfx(sfx_path, total, events, sfx_vol / 100)
            if music_f:
                mus_path = os.path.join(tmp, "music" + os.path.splitext(music_f.name)[1])
                open(mus_path, "wb").write(music_f.getvalue())
            if sfx_path or mus_path:
                prog.progress(0.0, "Mixing audio...")
                final_audio = mix_audio(audio, sfx_path, mus_path, total, mvol / 100, duck, tmp)

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
            if fx:
                with open(os.path.join(tmp, "ff.log"), "w") as log:
                    render_fx(files, times, kinds, W, H, fit, zmode, zamt, tkind, tdur, 24, total,
                              final_audio, vf, out, log, prog)
            else:
                lst = os.path.join(tmp, "list.txt")
                with open(lst, "w") as lf:
                    for i, (f, (a, b)) in enumerate(zip(files, times)):
                        p = os.path.join(tmp, f"img_{i}.jpg")
                        prep_image(f, p, W, H, fit)
                        lf.write(f"file '{p}'\nduration {max(b - a, 0.05):.3f}\n")
                        if i % 10 == 0:
                            prog.progress((i + 1) / len(files), f"Preparing images {i+1}/{len(files)}")
                    lf.write(f"file '{p}'\n")
                prog.progress(1.0, "Rendering video (please wait, keep this page open)...")
                run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", final_audio,
                     "-vf", vf, "-r", "24", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                     "-c:a", "aac", "-b:a", "192k", "-shortest", out])
            st.session_state.video = open(out, "rb").read()
        prog.empty()
        st.success("Done!")
    except Exception as e:
        st.error(f"Error: {e}")

if "video" in st.session_state:
    st.video(st.session_state.video)
    st.download_button("Download MP4", st.session_state.video, "video.mp4", "video/mp4")
