#!/usr/bin/bash
set -e

podman build -t iterative-marabou .

podman run -it --replace --name my-exec2 iterative-marabou $@

podman cp my-exec:./result/ ./
