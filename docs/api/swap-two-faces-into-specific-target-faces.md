# Swap two faces from source image into 2 specific target faces in a target image containing multiple faces

## Request

If you have a source image that contains two faces, but a target
image that contains multiple faces, you can specify the indexes
of the faces that you want to swap in the target image.  For
example, the payload below will swap the 2 faces from the
source image into the first and third face in the target image.

```json
{
  "input": {
    "source_image": "base64 encoded source image content",
    "target_image": "base64 encoded target image content",
    "source_indexes": "-1",
    "target_indexes": "0,2",
    "background_enhance": true,
    "face_restore": true,
    "face_upsample": true,
    "upscale": 1,
    "codeformer_fidelity": 0.5,
    "output_format": "JPEG"
  }
}
```

## Response

## RUN

```json
{
  "id": "83bbc301-5dcd-4236-9293-a65cdd681858",
  "status": "IN_QUEUE"
}
```

## RUNSYNC


```json
{
  "delayTime": 20275,
  "executionTime": 43997,
  "id": "sync-a3b54383-e671-4e24-a7bd-c5fec16fda3b",
  "output": {
    "status": "ok",
    "image": "base64 encoded output image"
  },
  "status": "COMPLETED"
}
```
