import csv, html, re, sys, time, urllib.request

D = sys.argv[1]
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}


def get(url, method="GET"):
    req = urllib.request.Request(url, headers=UA, method=method)
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.status, r.geturl(), r.headers, (r.read() if method == "GET" else b"")


def strip(s):
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s))).strip()


rows = []
for url in open(f"{D}/episodes.txt").read().split():
    src, page = "live", None
    for u, s in ((url, "live"), (f"https://web.archive.org/web/2016/{url}", "wayback")):
        try:
            st, _, _, body = get(u)
            if st == 200:
                page, src = body.decode("utf-8", "replace"), s
                break
        except Exception as e:
            print("fail", u, e, file=sys.stderr)
    if page is None:
        rows.append(dict(url=url, title="", date="", has_text="", audio_url="", audio_mb="", source="unreachable", notes_words=0, notes=""))
        continue
    title = strip((re.search(r"<h1[^>]*>(.*?)</h1>", page, re.S) or re.search(r"<title>(.*?)</title>", page, re.S)).group(1))
    m = re.search(r'episode-meta">.*?<span>([^<]+)</span></div>', page)
    date = m.group(1) if m else ""
    m = re.search(r'episode-notes">(.*?)</div>', page, re.S)
    notes = strip(re.sub(r"<h2>Episode notes</h2>", "", m.group(1))) if m else ""
    notes = re.sub(r"\s*Check out this episode$", "", notes)
    mp3 = sorted(set(re.findall(r'https?://[^"\'\s<>]+\.mp3', page)))
    audio, mb = (mp3[0] if mp3 else ""), ""
    if audio:
        try:
            _, _, h, _ = get(audio, "HEAD")
            mb = round(int(h.get("Content-Length", 0)) / 1e6, 1)
        except Exception as e:
            print("head fail", audio, e, file=sys.stderr)
    nw = len(notes.split())
    rows.append(dict(url=url, title=title, date=date, has_text="transcript" if nw > 1500 else ("notes" if nw > 150 else "short" if nw else "no"),
                     audio_url=audio, audio_mb=mb, source=src, notes_words=nw, notes=notes))
    print(src, nw, mb, title, file=sys.stderr)
    time.sleep(1)

with open(f"{D}/survey.csv", "w", newline="") as f:
    w = csv.DictWriter(f, ["url", "title", "date", "has_text", "audio_url", "audio_mb", "source", "notes_words"], extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
with open(f"{D}/raw/notes.tsv", "w") as f:
    for r in rows:
        f.write(f"{r['url']}\t{r['title']}\t{r['date']}\t{r['notes']}\n")
