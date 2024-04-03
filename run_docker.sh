#!/usr/bin/bash
set -e

podman build -t iterative-reluplex .

podman run -it --replace --name my-exec iterative-reluplex $@

podman cp my-exec:./result/ ./