import os
import io
import shutil
import base64
import mimetypes
import copy
import cv2
import insightface
import numpy as np
import traceback
import runpod
import boto3
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger
from typing import List, Union
from PIL import Image
from capturer import get_video_frame
from restoration import *
from schemas.input import INPUT_SCHEMA
from typing import Optional
from utilities import create_video, detect_fps, extract_frames
from dotenv import load_dotenv
from runpod.serverless.utils import rp_upload, rp_download
from botocore.exceptions import ClientError

# load_dotenv()
FACE_SWAP_MODEL = "checkpoints/inswapper_128.onnx"
TMP_PATH = "/tmp/inswapper"
logger = RunPodLogger()
logger.info("env1", os.environ["BUCKET_ENDPOINT_URL"])
logger.info("env2", os.environ.get("BUCKET_ENDPOINT_URL"))
logger.info("env3", os.environ.get("RUNPOD_BUCKET_ENDPOINT_URL"))
logger.info("env4", os.environ["RUNPOD_BUCKET_ENDPOINT_URL"])

aws_access_key_id = os.environ.get('AWS_S3_ACCESS_KEY_ID', None)
aws_secret_access_key = os.environ.get('AWS_S3_SECRET_ACCESS_KEY', None)
aws_region = os.environ.get('AWS_S3_REGION', None)
bucket_name = os.environ.get('AWS_S3_BUCKET_NAME', None)
bucket_endpoint_url = os.environ.get('BUCKET_ENDPOINT_URL', None)


# ---------------------------------------------------------------------------- #
# Application Functions                                                        #
# ---------------------------------------------------------------------------- #
def get_face_swap_model(model_path: str):
    model = insightface.model_zoo.get_model(model_path)
    return model


def get_face_analyser(model_path: str, torch_device: str, det_size=(320, 320)):

    if torch_device == "cuda":
        providers = ["CUDAExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    face_analyser = insightface.app.FaceAnalysis(
        name="buffalo_l", root="./checkpoints", providers=providers
    )

    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser


def get_one_face(face_analyser, frame: np.ndarray):
    face = face_analyser.get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(face_analyser, frame: np.ndarray):
    """
    get faces from left to right by order
    """
    try:
        face = face_analyser.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None


def clamp_cut_values(startX, endX, startY, endY, image):
    if startX < 0:
        startX = 0
    if endX > image.shape[1]:
        endX = image.shape[1]
    if startY < 0:
        startY = 0
    if endY > image.shape[0]:
        endY = image.shape[0]
    return startX, endX, startY, endY


def swap_face(source_faces, target_faces, source_index, target_index, temp_frame):
    """
    paste source_face on target image
    """
    global FACE_SWAPPER

    source_face = source_faces[source_index]
    target_face = target_faces[target_index]

    return FACE_SWAPPER.get(temp_frame, target_face, source_face, paste_back=True)


def process(
    source_img: Union[Image.Image, List],
    target_img: Image.Image,
    source_indexes: str,
    target_indexes: str,
):

    global MODEL, FACE_ANALYSER

    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    target_faces = get_many_faces(FACE_ANALYSER, target_img)
    num_target_faces = len(target_faces)
    num_source_images = len(source_img)

    if target_faces is not None:
        if num_target_faces == 0:
            logger.info("The target image does not contain any faces!")
            return

        temp_frame = copy.deepcopy(target_img)

        if isinstance(source_img, list) and num_source_images == num_target_faces:
            logger.info(
                "Replacing the faces in the target image from left to right by order"
            )
            for i in range(num_target_faces):
                source_faces = get_many_faces(
                    FACE_ANALYSER,
                    cv2.cvtColor(np.array(source_img[i]), cv2.COLOR_RGB2BGR),
                )
                source_index = i
                target_index = i

                if source_faces is None:
                    raise Exception("No source faces found!")

                temp_frame = swap_face(
                    source_faces, target_faces, source_index, target_index, temp_frame
                )
        elif num_source_images == 1:
            # detect source faces that will be replaced into the target image
            source_faces = get_many_faces(
                FACE_ANALYSER, cv2.cvtColor(np.array(source_img[0]), cv2.COLOR_RGB2BGR)
            )
            num_source_faces = len(source_faces)
            logger.info(f"Source faces: {num_source_faces}")
            logger.info(f"Target faces: {num_target_faces}")

            if source_faces is None or num_source_faces == 0:
                raise Exception("No source faces found!")

            if target_indexes == "-1":
                if num_source_faces == 1:
                    logger.info(
                        "Replacing the first face in the target image with the face from the source image"
                    )
                    num_iterations = num_source_faces
                elif num_source_faces < num_target_faces:
                    logger.info(
                        f"There are less faces in the source image than the target image, replacing the first {num_source_faces} faces"
                    )
                    num_iterations = num_source_faces
                elif num_target_faces < num_source_faces:
                    logger.info(
                        f"There are less faces in the target image than the source image, replacing {num_target_faces} faces"
                    )
                    num_iterations = num_target_faces
                else:
                    logger.info(
                        "Replacing all faces in the target image with the faces from the source image"
                    )
                    num_iterations = num_target_faces

                for i in range(num_iterations):
                    source_index = 0 if num_source_faces == 1 else i
                    target_index = i

                    temp_frame = swap_face(
                        source_faces,
                        target_faces,
                        source_index,
                        target_index,
                        temp_frame,
                    )
            elif source_indexes == "-1" and target_indexes == "-1":
                logger.info(
                    "Replacing specific face(s) in the target image with the face from the source image"
                )
                target_indexes = target_indexes.split(",")
                source_index = 0

                for target_index in target_indexes:
                    target_index = int(target_index)

                    temp_frame = swap_face(
                        source_faces,
                        target_faces,
                        source_index,
                        target_index,
                        temp_frame,
                    )
            else:
                logger.info(
                    "Replacing specific face(s) in the target image with specific face(s) from the source image"
                )

                if source_indexes == "-1":
                    source_indexes = ",".join(
                        map(lambda x: str(x), range(num_source_faces))
                    )

                if target_indexes == "-1":
                    target_indexes = ",".join(
                        map(lambda x: str(x), range(num_target_faces))
                    )

                source_indexes = source_indexes.split(",")
                target_indexes = target_indexes.split(",")
                num_source_faces_to_swap = len(source_indexes)
                num_target_faces_to_swap = len(target_indexes)

                if num_source_faces_to_swap > num_source_faces:
                    raise Exception(
                        "Number of source indexes is greater than the number of faces in the source image"
                    )

                if num_target_faces_to_swap > num_target_faces:
                    raise Exception(
                        "Number of target indexes is greater than the number of faces in the target image"
                    )

                if num_source_faces_to_swap > num_target_faces_to_swap:
                    num_iterations = num_source_faces_to_swap
                else:
                    num_iterations = num_target_faces_to_swap

                if num_source_faces_to_swap == num_target_faces_to_swap:
                    for index in range(num_iterations):
                        source_index = int(source_indexes[index])
                        target_index = int(target_indexes[index])

                        if source_index > num_source_faces - 1:
                            raise ValueError(
                                f"Source index {source_index} is higher than the number of faces in the source image"
                            )

                        if target_index > num_target_faces - 1:
                            raise ValueError(
                                f"Target index {target_index} is higher than the number of faces in the target image"
                            )

                        temp_frame = swap_face(
                            source_faces,
                            target_faces,
                            source_index,
                            target_index,
                            temp_frame,
                        )
        else:
            logger.error("Unsupported face configuration")
            raise Exception("Unsupported face configuration")
        result = temp_frame
    else:
        logger.error("No target faces found")
        raise Exception("No target faces found!")

    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result_image


def face_swap(
    src_img_path,
    target_img_path,
    source_indexes,
    target_indexes,
    background_enhance,
    face_restore,
    face_upsample,
    upscale,
    codeformer_fidelity,
    output_format,
):

    global TORCH_DEVICE, CODEFORMER_DEVICE, CODEFORMER_NET

    source_img_paths = src_img_path.split(";")
    source_img = [Image.open(img_path) for img_path in source_img_paths]
    logger.info(f"target_img_path: {target_img_path}")
    if is_video(target_img_path):
        target_frame = get_video_frame(target_img_path)
        target_img = Image.fromarray(target_frame)
    else:
        target_img = Image.open(target_img_path)

    try:
        logger.info("Performing face swap")
        result_image = process(source_img, target_img, source_indexes, target_indexes)
        logger.info("Face swap complete")
    except Exception as e:
        raise

    if face_restore:
        result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        logger.info("Performing face restoration using CodeFormer")

        try:
            result_image = face_restoration(
                result_image,
                background_enhance,
                face_upsample,
                upscale,
                codeformer_fidelity,
                upsampler,
                CODEFORMER_NET,
                CODEFORMER_DEVICE,
            )
        except Exception as e:
            raise

        logger.info("CodeFormer face restoration completed successfully")
        result_image = Image.fromarray(result_image)

    output_buffer = io.BytesIO()
    result_image.save(output_buffer, format=output_format)
    image_data = output_buffer.getvalue()

    return base64.b64encode(image_data).decode("utf-8")


def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith("video/"))
    return False


def clean_up_temporary_files(
    jobFolder: str
):
    shutil.rmtree(jobFolder)


def extract_objectkey_from_url(url):
    # Split the URL by '/'
    parts = url.split('/')
    # Get the last part which contains the filename
    filename = parts[-1]
    return filename

# Generate pre-signed URL
def generate_presigned_url(bucket_name, object_key, expiration=3600):
    try:
        s3_client = boto3.client('s3', aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key, region_name=aws_region)
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_key},
                                                    ExpiresIn=expiration)
    except ClientError as e:
        logger.info("Error generating presigned URL: ", e)
        return {
            "error": str(e),
            "output": traceback.format_exc(),
            "refresh_worker": True,
        }
    return response


async def face_swap_api(event, input):
    global FACE_ANALYSER

    if not os.path.exists(TMP_PATH):
        os.makedirs(TMP_PATH)
    
    
    logger.info("bucket_name", bucket_name)
    
    source_image_link = generate_presigned_url(bucket_name, extract_objectkey_from_url(input["source_image"]))
    target_video_link = generate_presigned_url(bucket_name, extract_objectkey_from_url(input["target_video"]))


    logger.info("presigned_url", source_image_link)

    # Download files
    source_image_path = rp_download.download_files_from_urls(
            event["id"], [source_image_link]
        )[0]
    target_video_path = rp_download.download_files_from_urls(
            event["id"], [target_video_link]
        )[0]
    

    logger.info("source_image", source_image_path)
    logger.info("target_video", target_video_path)


    try:
        logger.info(f'Source indexes: {input["source_indexes"]}')
        logger.info(f'Target indexes: {input["target_indexes"]}')
        logger.info(f'Background enhance: {input["background_enhance"]}')
        logger.info(f'Face Restoration: {input["face_restore"]}')
        logger.info(f'Face Upsampling: {input["face_upsample"]}')
        logger.info(f'Upscale: {input["upscale"]}')
        logger.info(f'Codeformer Fidelity: {input["codeformer_fidelity"]}')
        logger.info(f'Output Format: {input["output_format"]}')

        # extract frames
        fps = detect_fps(target_video_path)
        logger.info("fps: ", fps)
        frame_paths = await extract_frames(target_video_path, fps)
        logger.info("frames: ", frame_paths)

        # face swap

        for frame_path in frame_paths:
            target_img = Image.open(frame_path)
            target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
            target_faces = get_many_faces(FACE_ANALYSER, target_img)
            num_target_faces = len(target_faces)

            if num_target_faces > 0:
                result_image_base64 = face_swap(
                    source_image_path,
                    frame_path,  # Use the current frame path as the target image path
                    input["source_indexes"],
                    input["target_indexes"],
                    input["background_enhance"],
                    input["face_restore"],
                    input["face_upsample"],
                    input["upscale"],
                    input["codeformer_fidelity"],
                    input["output_format"],
                )

                # Convert base64 to image
                result_image_bytes = base64.b64decode(result_image_base64)
                result_image = Image.open(io.BytesIO(result_image_bytes))

                # Save the result image as JPEG
                result_image_path = os.path.splitext(frame_path)[0] + ".png"
                result_image.save(result_image_path, format="PNG")

        # create video
        logger.info("create_video")
        await create_video(frame_paths, target_video_path, fps)
        logger.info("create_video success")

        # upload video
        video_path = rp_upload.upload_file_to_bucket(file_name=os.path.basename(target_video_path), file_location=target_video_path, bucket_name=bucket_name, bucket_creds={"endpointUrl": bucket_endpoint_url, "accessId": aws_access_key_id, "accessSecret": aws_secret_access_key})

        clean_up_temporary_files(f'jobs/{event["id"]}')

        return {"video_path": video_path}
    except Exception as e:
        logger.error(f"An exception was raised: {e}")
        clean_up_temporary_files(f'jobs/{event["id"]}')

        return {
            "error": str(e),
            "output": traceback.format_exc(),
            "refresh_worker": True,
        }


# ---------------------------------------------------------------------------- #
# RunPod Handler                                                               #
# ---------------------------------------------------------------------------- #
def handler(event):
    validated_input = validate(event["input"], INPUT_SCHEMA)
    
    logger.info(f"Handler validating input")
    if "errors" in validated_input:
        return {"error": validated_input["errors"]}

    return face_swap_api(event, validated_input["validated_input"])


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL = os.path.join(script_dir, FACE_SWAP_MODEL)
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), MODEL)
    logger.info(f"Face swap model: {MODEL}")

    if torch.cuda.is_available():
        TORCH_DEVICE = "cuda"
    else:
        TORCH_DEVICE = "cpu"

    logger.info(f"Torch device: {TORCH_DEVICE.upper()}")
    FACE_ANALYSER = get_face_analyser(MODEL, TORCH_DEVICE)
    FACE_SWAPPER = get_face_swap_model(model_path)

    # Ensure that CodeFormer weights have been successfully downloaded,
    # otherwise download them
    check_ckpts()

    logger.info("Setting upsampler to RealESRGAN_x2plus")
    upsampler = set_realesrgan()
    CODEFORMER_DEVICE = torch.device(TORCH_DEVICE)

    CODEFORMER_NET = ARCH_REGISTRY.get("CodeFormer")(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=["32", "64", "128", "256"],
    ).to(CODEFORMER_DEVICE)

    ckpt_path = os.path.join(
        script_dir, "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"
    )
    logger.info(f"Loading CodeFormer model: {ckpt_path}")
    codeformer_checkpoint = torch.load(ckpt_path)["params_ema"]
    CODEFORMER_NET.load_state_dict(codeformer_checkpoint)
    CODEFORMER_NET.eval()

    logger.info("Starting RunPod Serverless...")
    runpod.serverless.start({"handler": handler})