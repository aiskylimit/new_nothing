#!/bin/sh

# Get relative path of the root directory of the project
rdir=`git rev-parse --git-dir`
rel_path="$(dirname "$rdir")"
# Change to that path and run the file
cd $rel_path

# antlr-4.11.1-complete.jar (3.4MB) da bo khoi repo de git nhe; gen/ da commit san nen
# chi can jar khi sua PS.g4. Tai lai: https://www.antlr.org/download/antlr-4.11.1-complete.jar
java -jar antlr-4.11.1-complete.jar PS.g4 -o gen
