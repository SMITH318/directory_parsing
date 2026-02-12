import json
import shutil
import re
import unicodedata
from pathlib import Path

SRC = Path("ocr_output_auto_cleaned.jsonl")
OUT = SRC.with_name(SRC.stem + ".cleaned.jsonl")
BACKUP = SRC.with_suffix(".backup.jsonl")

# simple mapping of rare/unusual -> common
REPL = {
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "‚": ",",
    "—": "-", "–": "-", "−": "-",
    "…": "...",
    "\u00A0": " ",  # NBSP
    "\u200B": "", "\u200C": "", "\u200D": "", "\u200E": "", "\u200F": "",
    "·": ".", "•": "-", "•": "-", "×": "x",
    "°": "°",  # keep degree symbol (optional)
    "©": "(c)", "®": "(r)"
}

# compile regexes
CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")
MULTI_WS_RE = re.compile(r"[ \t]{2,}")
TRAIL_LEAD_WS_RE = re.compile(r"^[ \t]+|[ \t]+$")

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    # normalize unicode compatibility
    s = unicodedata.normalize("NFKC", s)
    # direct replacements
    for a, b in REPL.items():
        if a in s:
            s = s.replace(a, b)
    # remove control chars (keep newline/tab handled outside JSON strings)
    s = CONTROL_RE.sub("", s)
    # collapse multiple spaces/tabs
    s = MULTI_WS_RE.sub(" ", s)
    # trim leading/trailing spaces
    s = TRAIL_LEAD_WS_RE.sub("", s)
    return s

def clean_obj(o):
    if isinstance(o, str):
        return clean_text(o)
    if isinstance(o, list):
        return [clean_obj(x) for x in o]
    if isinstance(o, dict):
        return {clean_obj(k) if isinstance(k, str) else k: clean_obj(v) for k,v in o.items()}
    return o

def main():
    if not SRC.exists():
        print("Source file not found:", SRC)
        return
    # backup original
    shutil.copy2(SRC, BACKUP)
    with SRC.open("r", encoding="utf-8", errors="replace") as inf, OUT.open("w", encoding="utf-8") as outf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # fallback: clean the raw line minimally and write
                cleaned = clean_text(line)
                outf.write(cleaned + "\n")
                continue
            cleaned_obj = clean_obj(obj)
            outf.write(json.dumps(cleaned_obj, ensure_ascii=False) + "\n")
    print("Cleaned saved to", OUT, "backup saved to", BACKUP)

if __name__ == "__main__":
    main()