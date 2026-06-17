"""Trim a WAV to a short clip using only the Python stdlib (no ffmpeg).

Usage: py trim_audio.py <in.wav> <out.wav> <start_seconds> <dur_seconds>
Keeps the source sample rate. If multichannel, keeps channel 0 (DEMAND ships
16 separate single-channel files, so inputs here are already mono).
"""
import sys, wave

def trim(infile, outfile, start_s, dur_s):
    with wave.open(infile, "rb") as w:
        nch, sw, fr, nframes = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        start = max(0, min(int(start_s * fr), nframes))
        dur = max(0, min(int(dur_s * fr), nframes - start))
        w.setpos(start)
        frames = w.readframes(dur)
    if nch > 1:  # keep channel 0
        step = sw * nch
        frames = b"".join(frames[i:i+sw] for i in range(0, len(frames) - step + 1, step))
    with wave.open(outfile, "wb") as o:
        o.setnchannels(1); o.setsampwidth(sw); o.setframerate(fr)
        o.writeframes(frames)
    print(f"{outfile}: mono {fr} Hz, {dur/fr:.2f}s, {len(frames)} bytes (src {nch}ch {nframes/fr:.1f}s)")

if __name__ == "__main__":
    trim(sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
