# iterative Marabou

This is the code for the project of the TUDelft course Formal Methods for Learned Systems (CS4354), see: [TUDelft Studyguide](https://studiegids.tudelft.nl/a101_displayCourse.do?course_id=63112).

## How to run

To recomended way to run this project is using building the docker image and running this.

### Running podman/docker

#### Prerequisites

Please ensure that the following tool is installed

* [Docker](https://www.docker.com/get-started/) (or [podman](https://podman.io/docs/installation))

> Note I used podman to build the images but docker should also work, if you use docker replace `podman` with `docker` in the commands below:

1. To build the docker container run:

```bash
podman build -t iterative-marabou .
```

2. Then, to run the docker container run

```bash
podman run -it --replace --name my-exec iterative-marabou
```

3. Finally copy over the results from the container:

```bash
podman cp my-exec:./result/ ./
```

> Please note that all these commands are also included in the `run_docker.sh` script.