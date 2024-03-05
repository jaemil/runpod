# Swap two specific faces from source image containing multiple faces into 2 specific target faces in a target image containing multiple faces

## Request

If you have a source image that contains multiple faces, and a target
image that contains multiple faces, you can specify the indexes
of the faces that you want to use from the source image and the indexes
of the faces that you want to replace in the target image.  For
example, the payload below will swap the third and first faces from the
source image into the second and third faces in the target image.

```json
{
  "input": {
    "source_image": "base64 encoded source image content",
    "target_image": "base64 encoded target image content",
    "source_indexes": "2,0",
    "target_indexes": "1,2",
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
