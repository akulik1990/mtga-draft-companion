"""Transcribe Limited Resources episodes from survey.csv: download one mp3, transcribe, write text, delete the mp3.
Args: episode numbers, or "all". Episodes with a transcript already are skipped. Audio is never kept."""
import csv, os, re, sys, time, urllib.request
from faster_whisper import WhisperModel

D = os.path.dirname(os.path.abspath(__file__))
OUT = "/home/morris/Projects/mtga-draft-companion/data/podcasts"
PROMPT = ("Magic: The Gathering limited, draft, sealed, mana curve, two-drop, removal, bomb, splash, "
          "sideboard, LSV, Luis Scott-Vargas, Marshall Sutcliffe, Limited Resources.")
want = set(sys.argv[1:])
num = lambda r: re.search(r"Resources (\d+)", r["title"]).group(1)
rows = [r for r in csv.DictReader(open(f"{D}/survey.csv")) if "all" in want or num(r) in want]
os.makedirs(OUT, exist_ok=True)

try:
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
except Exception as e:
    print("int8_float16 failed, using float16:", e, flush=True)
    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")


def transcribe(r, slug, mp3):
    req = urllib.request.Request(r["audio_url"], headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as f, open(mp3, "wb") as o:
        o.write(f.read())
    t0 = time.time()
    segs, info = model.transcribe(mp3, language="en", vad_filter=True, initial_prompt=PROMPT, beam_size=5)
    paras, cur, last_end = [], [], 0.0
    for s in segs:
        # ponytail: paragraph break on a >1.5 s pause or every 10 segments; no speaker diarisation
        if cur and (s.start - last_end > 1.5 or len(cur) >= 10):
            paras.append(" ".join(cur)); cur = []
        cur.append(s.text.strip()); last_end = s.end
    if cur:
        paras.append(" ".join(cur))
    wall = time.time() - t0
    text = "\n\n".join(paras)
    with open(f"{OUT}/{slug}.txt", "w") as o:
        o.write(f"{r['title']}\n{r['date']}\n{r['url']}\n\n{text}\n")
    print(f"{slug}\taudio_min={info.duration/60:.1f}\twall_s={wall:.0f}\trtf={info.duration/wall:.1f}x\twords={len(text.split())}", flush=True)


for r in rows:
    slug = f"{int(num(r)):03d}-" + re.sub(r"^limited-resources-\d+-", "", r["url"].rstrip("/").rsplit("/", 1)[1])
    if os.path.exists(f"{OUT}/{slug}.txt"):
        continue
    mp3 = f"{D}/raw/{slug}.mp3"
    try:
        transcribe(r, slug, mp3)
    except Exception as e:
        print("FAILED", slug, repr(e), flush=True)
    finally:
        if os.path.exists(mp3):
            os.remove(mp3)  # audio is never kept, transcribed or not
print("DONE", flush=True)
