#!/bin/sh

IMAGE=blang/latex:ctanfull

docker run -d \
            --rm \
            --name latex_daemon \
            -i \
            --user="$(id -u):$(id -g)" \
            -t \
            -v $PWD:/data \
            "$IMAGE" \
            /bin/sh -c "sleep infinity"
