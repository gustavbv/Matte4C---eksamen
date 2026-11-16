#!/bin/sh
# Bygger oppgave-PDF og løsningsforslag for prøveeksamen C, D og E.
# Bruk: sh build.sh [C D E]. Krever pdflatex med tcolorbox, comment og lmodern.
set -e
cd "$(dirname "$0")"
for L in ${@:-C D E}; do
  for pass in 1 2; do
    pdflatex -interaction=nonstopmode -halt-on-error -jobname="exam$L" "exam$L.tex" > /dev/null
    pdflatex -interaction=nonstopmode -halt-on-error -jobname="exam$L-solutions" "\def\withsol{}\input{exam$L.tex}" > /dev/null
  done
  mv "exam$L.pdf" "../proveeksamen-$L.pdf"
  mv "exam$L-solutions.pdf" "../proveeksamen-$L-lf.pdf"
done
rm -f ./*.aux ./*.log ./*.out ./*.cut
