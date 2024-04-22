## Building the Worker without a Network Volume

This is the simpler option. No network volume is required.
The entire application will be stored within the Docker image
but will obviously create a more bulky Docker image as a result.

You can either use my pre-built Docker image:

```
ashleykza/runpod-worker-inswapper:standalone-1.5.0
```

Or alternatively, you can build it yourself by following the
instructions below.

### Clone the repo

```bash
git clone https://github.com/ashleykleynhans/runpod-worker-inswapper.git
```

### Build the Docker image

```bash
docker build --no-cache -t dockerhub-username/runpod-worker-inswapper:1.0.0 -f Dockerfile.Standalone .
docker login
docker push dockerhub-username/runpod-worker-inswapper:1.0.0
```
