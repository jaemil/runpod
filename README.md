Run locally

```bash
python3 -u rp_handler.py --rp_serve_api
```

# roop inswapper | RunPod Serverless Worker

This is the source code for a [RunPod](https://runpod.io?ref=2xxro4sy)
Serverless worker that uses roop ([inswapper](https://huggingface.co/deepinsight/inswapper/tree/main)) for face
swapping AI tasks.

## Model

The worker uses the `inswapper_128.onnx` model by [InsightFace](https://insightface.ai/).

## Testing

1. [Local Testing](docs/testing/local.md)
2. [RunPod Testing](docs/testing/runpod.md)

## Building the Docker image that will be used by the Serverless Worker

There are two options:

1. [Network Volume](docs/building/with-network-volume.md)
2. [Standalone](docs/building/without-network-volume.md) (without Network Volume)

## RunPod API Endpoint

You can send requests to your RunPod API Endpoint using the `/run`
or `/runsync` endpoints.

Requests sent to the `/run` endpoint will be handled asynchronously,
and are non-blocking operations. Your first response status will always
be `IN_QUEUE`. You need to send subsequent requests to the `/status`
endpoint to get further status updates, and eventually the `COMPLETED`
status will be returned if your request is successful.

Requests sent to the `/runsync` endpoint will be handled synchronously
and are blocking operations. If they are processed by a worker within
90 seconds, the result will be returned in the response, but if
the processing time exceeds 90 seconds, you will need to handle the
response and request status updates from the `/status` endpoint until
you receive the `COMPLETED` status which indicates that your request
was successful.

### RunPod API Examples

- [Swap as many source faces as possible into as many target faces as possible](docs/api/swap-as-many-faces-as-possible.md)
- [Swap a single source face into a specific target face in a target image containing multiple faces](docs/api/swap-single-source-face-into-specific-target-face.md)
- [Swap two faces from source image into 2 specific target faces in a target image containing multiple faces](docs/api/swap-two-faces-into-specific-target-faces.md)
- [Swap two specific faces from source image containing multiple faces into 2 specific target faces in a target image containing multiple faces](docs/api/swap-specific-faces-into-specific-target-faces.md)

### Endpoint Status Codes

| Status      | Description                                                                                                                    |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------ |
| IN_QUEUE    | Request is in the queue waiting to be picked up by a worker. You can call the `/status` endpoint to check for status updates.  |
| IN_PROGRESS | Request is currently being processed by a worker. You can call the `/status` endpoint to check for status updates.             |
| FAILED      | The request failed, most likely due to encountering an error.                                                                  |
| CANCELLED   | The request was cancelled. This usually happens when you call the `/cancel` endpoint to cancel the request.                    |
| TIMED_OUT   | The request timed out. This usually happens when your handler throws some kind of exception that does return a valid response. |
| COMPLETED   | The request completed successfully and the output is available in the `output` field of the response.                          |

## Serverless Handler

The serverless handler (`rp_handler.py`) is a Python script that handles
the API requests to your Endpoint using the [runpod](https://github.com/runpod/runpod-python)
Python library. It defines a function `handler(event)` that takes an
API request (event), runs the inference using the [inswapper](https://huggingface.co/deepinsight/inswapper/tree/main) model (and
CodeFormer where applicable) with the `input`, and returns the `output`
in the JSON response.

## Acknowledgements

- [Inswapper](https://github.com/haofanwang/inswapper)
- [Roop](https://github.com/s0md3v/roop)
- [Insightface](https://github.com/deepinsight)
- [CodeFormer](https://huggingface.co/spaces/sczhou/CodeFormer)
- [Real-ESRGAN (ai-forever)](https://github.com/ai-forever/Real-ESRGAN)
- [Generative Labs YouTube Tutorials](https://www.youtube.com/@generativelabs)

## Additional Resources

- [Generative Labs YouTube Tutorials](https://www.youtube.com/@generativelabs)
- [Getting Started With RunPod Serverless](https://trapdoor.cloud/getting-started-with-runpod-serverless/)
- [Serverless | Create a Custom Basic API](https://blog.runpod.io/serverless-create-a-basic-api/)

## Community and Contributing

Pull requests and issues on [GitHub](https://github.com/ashleykleynhans/runpod-worker-inswapper)
are welcome. Bug fixes and new features are encouraged.

You can contact me and get help with deploying your Serverless
worker to RunPod on the RunPod Discord Server below,
my username is **ashleyk**.

<a target="_blank" href="https://discord.gg/pJ3P2DbUUq">![Discord Banner 2](https://discordapp.com/api/guilds/912829806415085598/widget.png?style=banner2)</a>

## Appreciate my work?

<a href="https://www.buymeacoffee.com/ashleyk" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>
