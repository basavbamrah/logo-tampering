#!/usr/bin/env bash
# Downloads five Google Fonts (all SIL Open Font License) into data/fonts/.
# These are used by the synthetic data generator (EXP-003) and the font
# matching check (EXP-009). Each font folder also gets its OFL.txt so the
# licence travels with the file.
set -euo pipefail

cd "$(dirname "$0")/.."
BASE="https://raw.githubusercontent.com/google/fonts/main/ofl"
OUT="data/fonts"

get() {               # get <folder-in-repo> <file-name>
  mkdir -p "$OUT/$1"
  curl -sSfL "$BASE/$1/$2"     -o "$OUT/$1/$2"
  curl -sSfL "$BASE/$1/OFL.txt" -o "$OUT/$1/OFL.txt"
  echo "ok  $1/$2"
}

get opensans "OpenSans%5Bwdth,wght%5D.ttf"   # variable font, we pick weight 700 in code
get notosans "NotoSans%5Bwdth,wght%5D.ttf"   # variable font
get roboto   "Roboto%5Bwdth,wght%5D.ttf"     # variable font
get lato     "Lato-Bold.ttf"
get ptsans   "PT_Sans-Web-Bold.ttf"

# curl keeps the %5B / %5D in the name; rename to plain brackets.
for f in "$OUT"/*/*%5B*; do mv "$f" "$(echo "$f" | sed 's/%5B/[/; s/%5D/]/')"; done
