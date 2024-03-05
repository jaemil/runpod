# Swap as many source faces as possible into as many target faces as possible

## Request

This payload can handle any of the following conditions:

1. Swapping a face from a source image that contains a single
   face into a target image that contains a single face.
2. Swapping a face from a source image that contains a single
   face into the first face in a target image that contains
   multiple faces.
3. Swapping all of the faces from a source image containing
   multiple faces into the faces of a target image containing
   the same number of faces.
4. Swapping all of the faces from a source image containing
   multiple faces into the first faces that are found in a
   target image containing more faces than the source image.
5. If a source image contains more faces than the target image,
   the first X number of faces from the source image will be
   swapped in the target image, where X is the total number
   of faces in the target image.

```json
{
  "input": {
    "source_image": "base64 encoded source image content",
    "target_image": "base64 encoded target image content",
    "source_indexes": "-1",
    "target_indexes": "-1",
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
