#!/bin/sh
# Build the paper into report.pdf, then report its page budget.
#
# Requires pdflatex and bibtex on PATH (TeX Live or MiKTeX).
#
#   sh report/build.sh

cd "$(dirname "$0")" || exit 1

TEX=${1:-report}

pdflatex -interaction=nonstopmode "$TEX.tex" > build1.log 2>&1
bibtex   "$TEX"                              > build2.log 2>&1
pdflatex -interaction=nonstopmode "$TEX.tex" > build3.log 2>&1
pdflatex -interaction=nonstopmode "$TEX.tex" > build4.log 2>&1

if [ ! -f "$TEX.pdf" ]; then
  echo "BUILD FAILED -- last errors:"; grep -A3 '^!' build4.log | head -40; exit 1
fi

echo "== $TEX.pdf: $(grep -oE 'Output written on [^ ]+ \([0-9]+ pages' build4.log | grep -oE '[0-9]+ pages')"
echo
echo "== bibtex"
grep -E "error|warning|I couldn't|I found no" build2.log | head -10 || true
echo "== undefined citations/references"
grep -E "Citation .* undefined|Reference .* undefined|There were undefined" build4.log | head -10 || echo "  none"
echo "== overfull boxes > 10pt"
grep -oE "Overfull \\\\hbox \(([0-9.]+)pt" build4.log | grep -oE "[0-9.]+" \
  | awk '$1 > 10 {n++} END {print "  " n+0 " overfull hboxes wider than 10pt"}'
echo "== other LaTeX warnings"
grep "LaTeX Warning" build4.log | grep -v "Font shape" | sort | uniq -c | head -10 || echo "  none"
