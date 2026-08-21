#!/usr/bin/env bash
# Fetch the public corpora the benchmark reads. Data is not vendored into this
# repository: it belongs to its authors and is fetched from their sources.
#
#   Course-prerequisite networks, Stavrinides & Zuev, Applied Network Science 2023
#   https://github.com/pstavrin/Course-Prerequisite-Networks
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$root/data/cpn"
base="https://raw.githubusercontent.com/pstavrin/Course-Prerequisite-Networks/main"

mkdir -p "$dest"

echo "Fetching Caltech course-prerequisite network..."
curl -fsSL "$base/2021-22-course-prereqs%20-%20clean.csv" -o "$dest/caltech.csv"

echo "Fetching Johns Hopkins course listing..."
curl -fsSL "$base/JHU_Courses.csv" -o "$dest/jhu.csv"

echo "Done. Files in $dest:"
ls -lh "$dest"
