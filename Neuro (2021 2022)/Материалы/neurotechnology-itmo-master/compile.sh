#!/bin/sh

BUILD_DIRECTORY=$1
OUTPUT_DIRECTORY=outputs

if [ -z "$BUILD_DIRECTORY" ]
then
    echo "Directory should be specified to compile. F.e. './compile.sh lab1'"
    exit 1
fi

if [ ! -d "$OUTPUT_DIRECTORY" ]
then
    mkdir -p "$OUTPUT_DIRECTORY"
fi

if [ ! -d "$OUTPUT_DIRECTORY" ]
then
    echo "Output directory cannot be created."
    exit 1
fi

docker exec -it latex_daemon pdflatex -output-directory="$OUTPUT_DIRECTORY" \
                                      -jobname="$BUILD_DIRECTORY" \
                                      "common/docs/main.tex"
