import os
import io
import uuid
import base64
import mimetypes
import copy
import cv2
import insightface
import numpy as np
import traceback
import runpod
from runpod.serverless.utils.rp_validator import validate
from runpod.serverless.modules.rp_logger import RunPodLogger
from typing import List, Union
from PIL import Image
from capturer import get_video_frame
from face_util import resize_image_keep_content
from restoration import *
from schemas.input import INPUT_SCHEMA
from typing import Optional

from utilities import create_video, detect_fps, extract_frames


FACE_SWAP_MODEL = 'checkpoints/inswapper_128.onnx'
TMP_PATH = '/tmp/inswapper'
logger = RunPodLogger()


# ---------------------------------------------------------------------------- #
# Application Functions                                                        #
# ---------------------------------------------------------------------------- #
def get_face_swap_model(model_path: str):
    model = insightface.model_zoo.get_model(model_path)
    return model


def get_face_analyser(model_path: str,
                      torch_device: str,
                      det_size=(320, 320)):

    if torch_device == 'cuda':
        providers=['CUDAExecutionProvider']
    else:
        providers=['CPUExecutionProvider']

    face_analyser = insightface.app.FaceAnalysis(
        name="buffalo_l",
        root="./checkpoints",
        providers=providers
    )

    face_analyser.prepare(ctx_id=0, det_size=det_size)
    return face_analyser


def get_one_face(face_analyser,
                 frame:np.ndarray):
    face = face_analyser.get(frame)
    try:
        return min(face, key=lambda x: x.bbox[0])
    except ValueError:
        return None


def get_many_faces(face_analyser,
                   frame:np.ndarray):
    """
    get faces from left to right by order
    """
    try:
        face = face_analyser.get(frame)
        return sorted(face, key=lambda x: x.bbox[0])
    except IndexError:
        return None
    
    
# def extract_face_images(source_filename, video_info, extra_padding=-1.0):
#     global FACE_ANALYSER
    
#     face_data = []
#     source_image = None

#     logger.info("video info", video_info[0])

#     if video_info[0]:
#         frame = get_video_frame(source_filename, video_info[1])
#         if frame is not None:
#             source_image = frame
#         else:
#             return face_data
#     else:
#         logger.info("source_filename", source_filename)
#         source_image = cv2.imread(source_filename)

#     faces = get_many_faces(FACE_ANALYSER, source_image)
#     if faces is None:
#         return face_data

#     i = 0
#     for face in faces:
#         (startX, startY, endX, endY) = face["bbox"].astype("int")
#         if extra_padding > 0.0:
#             if source_image.shape[:2] == (512, 512):
#                 i += 1
#                 face_data.append([face, source_image])
#                 continue

#             found = False
#             for i in range(1, 3):
#                 (startX, startY, endX, endY) = face["bbox"].astype("int")
#                 cutout_padding = extra_padding
#                 # top needs extra room for detection
#                 padding = int((endY - startY) * cutout_padding)
#                 oldY = startY
#                 startY -= padding

#                 factor = 0.25 if i == 1 else 0.5
#                 cutout_padding = factor
#                 padding = int((endY - oldY) * cutout_padding)
#                 endY += padding
#                 padding = int((endX - startX) * cutout_padding)
#                 startX -= padding
#                 endX += padding
#                 startX, endX, startY, endY = clamp_cut_values(
#                     startX, endX, startY, endY, source_image
#                 )
#                 face_temp = source_image[startY:endY, startX:endX]
#                 face_temp = resize_image_keep_content(face_temp)
#                 testfaces = get_many_faces(FACE_ANALYSER, face_temp)
#                 if testfaces is not None and len(testfaces) > 0:
#                     i += 1
#                     face_data.append([testfaces[0], face_temp])
#                     found = True
#                     break

#             if not found:
#                 print("No face found after resizing, this shouldn't happen!")
#             continue

#         face_temp = source_image[startY:endY, startX:endX]
#         if face_temp.size < 1:
#             continue

#         i += 1
#         face_data.append([face, face_temp])
#     return face_data


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


def swap_face(source_faces,
              target_faces,
              source_index,
              target_index,
              temp_frame):
    """
    paste source_face on target image
    """
    global FACE_SWAPPER

    source_face = source_faces[source_index]
    target_face = target_faces[target_index]

    return FACE_SWAPPER.get(temp_frame, target_face, source_face, paste_back=True)




def process(source_img: Union[Image.Image, List],
            target_img: Image.Image,
            source_indexes: str,
            target_indexes: str):

    global MODEL, FACE_ANALYSER

    target_img = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
    target_faces = get_many_faces(FACE_ANALYSER, target_img)
    num_target_faces = len(target_faces)
    num_source_images = len(source_img)

    if target_faces is not None:
        if num_target_faces == 0:
            logger.info('The target image does not contain any faces!')
            return

        temp_frame = copy.deepcopy(target_img)

        if isinstance(source_img, list) and num_source_images == num_target_faces:
            logger.info('Replacing the faces in the target image from left to right by order')
            for i in range(num_target_faces):
                source_faces = get_many_faces(FACE_ANALYSER, cv2.cvtColor(np.array(source_img[i]), cv2.COLOR_RGB2BGR))
                source_index = i
                target_index = i

                if source_faces is None:
                    raise Exception('No source faces found!')

                temp_frame = swap_face(
                    source_faces,
                    target_faces,
                    source_index,
                    target_index,
                    temp_frame
                )
        elif num_source_images == 1:
            # detect source faces that will be replaced into the target image
            source_faces = get_many_faces(FACE_ANALYSER, cv2.cvtColor(np.array(source_img[0]), cv2.COLOR_RGB2BGR))
            num_source_faces = len(source_faces)
            logger.info(f'Source faces: {num_source_faces}')
            logger.info(f'Target faces: {num_target_faces}')

            if source_faces is None or num_source_faces == 0:
                raise Exception('No source faces found!')

            if target_indexes == "-1":
                if num_source_faces == 1:
                    logger.info('Replacing the first face in the target image with the face from the source image')
                    num_iterations = num_source_faces
                elif num_source_faces < num_target_faces:
                    logger.info(f'There are less faces in the source image than the target image, replacing the first {num_source_faces} faces')
                    num_iterations = num_source_faces
                elif num_target_faces < num_source_faces:
                    logger.info(f'There are less faces in the target image than the source image, replacing {num_target_faces} faces')
                    num_iterations = num_target_faces
                else:
                    logger.info('Replacing all faces in the target image with the faces from the source image')
                    num_iterations = num_target_faces

                for i in range(num_iterations):
                    source_index = 0 if num_source_faces == 1 else i
                    target_index = i

                    temp_frame = swap_face(
                        source_faces,
                        target_faces,
                        source_index,
                        target_index,
                        temp_frame
                    )
            elif source_indexes == '-1' and target_indexes == '-1':
                logger.info('Replacing specific face(s) in the target image with the face from the source image')
                target_indexes = target_indexes.split(',')
                source_index = 0

                for target_index in target_indexes:
                    target_index = int(target_index)

                    temp_frame = swap_face(
                        source_faces,
                        target_faces,
                        source_index,
                        target_index,
                        temp_frame
                    )
            else:
                logger.info('Replacing specific face(s) in the target image with specific face(s) from the source image')

                if source_indexes == "-1":
                    source_indexes = ','.join(map(lambda x: str(x), range(num_source_faces)))

                if target_indexes == "-1":
                    target_indexes = ','.join(map(lambda x: str(x), range(num_target_faces)))

                source_indexes = source_indexes.split(',')
                target_indexes = target_indexes.split(',')
                num_source_faces_to_swap = len(source_indexes)
                num_target_faces_to_swap = len(target_indexes)

                if num_source_faces_to_swap > num_source_faces:
                    raise Exception('Number of source indexes is greater than the number of faces in the source image')

                if num_target_faces_to_swap > num_target_faces:
                    raise Exception('Number of target indexes is greater than the number of faces in the target image')

                if num_source_faces_to_swap > num_target_faces_to_swap:
                    num_iterations = num_source_faces_to_swap
                else:
                    num_iterations = num_target_faces_to_swap

                if num_source_faces_to_swap == num_target_faces_to_swap:
                    for index in range(num_iterations):
                        source_index = int(source_indexes[index])
                        target_index = int(target_indexes[index])

                        if source_index > num_source_faces-1:
                            raise ValueError(f'Source index {source_index} is higher than the number of faces in the source image')

                        if target_index > num_target_faces-1:
                            raise ValueError(f'Target index {target_index} is higher than the number of faces in the target image')

                        temp_frame = swap_face(
                            source_faces,
                            target_faces,
                            source_index,
                            target_index,
                            temp_frame
                        )
        else:
            logger.error('Unsupported face configuration')
            raise Exception('Unsupported face configuration')
        result = temp_frame
    else:
        logger.error('No target faces found')
        raise Exception('No target faces found!')

    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result_image


def face_swap(src_img_path,
              target_img_path,
              source_indexes,
              target_indexes,
              background_enhance,
              face_restore,
              face_upsample,
              upscale,
              codeformer_fidelity,
              output_format):

    global TORCH_DEVICE, CODEFORMER_DEVICE, CODEFORMER_NET

    source_img_paths = src_img_path.split(';')
    source_img = [Image.open(img_path) for img_path in source_img_paths]
    logger.info(f'target_img_path: {target_img_path}')
    if is_video(target_img_path):
        target_frame = get_video_frame(target_img_path)
        target_img = Image.fromarray(target_frame)
    else:
        target_img = Image.open(target_img_path)

    try:
        logger.info('Performing face swap')
        result_image = process(
            source_img,
            target_img,
            source_indexes,
            target_indexes
        )
        logger.info('Face swap complete')
    except Exception as e:
        raise

    if face_restore:
        result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        logger.info('Performing face restoration using CodeFormer')

        try:
            result_image = face_restoration(
                result_image,
                background_enhance,
                face_upsample,
                upscale,
                codeformer_fidelity,
                upsampler,
                CODEFORMER_NET,
                CODEFORMER_DEVICE
            )
        except Exception as e:
            raise

        logger.info('CodeFormer face restoration completed successfully')
        result_image = Image.fromarray(result_image)

    output_buffer = io.BytesIO()
    result_image.save(output_buffer, format=output_format)
    image_data = output_buffer.getvalue()

    return base64.b64encode(image_data).decode('utf-8')


def determine_file_extension(image_data):
    try:
        if image_data.startswith('/9j/'):
            image_extension = '.jpg'
        elif image_data.startswith('iVBORw0Kg'):
            image_extension = '.png'
        elif image_data.startswith(b'\x1f\x43\x44\x30'):  # MKV format
            file_extension = '.mkv'
        elif image_data.startswith(b'\x00\x00\x00\x18moov'):  # MP4 format
            file_extension = '.mp4'
        elif image_data.startswith(b'\x1a\x45\xdf\xa3') or image_data.startswith(b'\x1a\x45\xdf\xa3'):  # WebM format
            file_extension = '.webm'
        else:
            # Default to png if we can't figure out the extension
            image_extension = '.mp4'
    except Exception as e:
        image_extension = '.mp4'

    return image_extension


def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith('video/'))
    return False


def clean_up_temporary_files(source_image_path: str, target_image_path: str, source_frames_folder_path: str):
    os.remove(source_image_path)
    os.remove(target_image_path)
    os.remove(source_frames_folder_path)


def face_swap_api(input):
    global FACE_ANALYSER

    if not os.path.exists(TMP_PATH):
        os.makedirs(TMP_PATH)

    unique_id = uuid.uuid4()
    source_image_data = input['source_image']
    target_image_data = input['target_image']

    # Decode the source image data
    source_image = base64.b64decode(source_image_data)
    source_file_extension = determine_file_extension(source_image_data)
    source_image_path = f'{TMP_PATH}/source_{unique_id}{source_file_extension}'

    # Save the source image to disk
    with open(source_image_path, 'wb') as source_file:
        source_file.write(source_image)

    # Decode the target image data
    target_image = base64.b64decode(target_image_data)
    target_file_extension = determine_file_extension(target_image_data)
    target_image_path = f'{TMP_PATH}/target_{unique_id}{target_file_extension}'

    logger.info(f'target image: {target_image}')
    logger.info(f'Meme type: {is_video(target_image)}')

    # Save the target image to disk
    with open(target_image_path, 'wb') as target_file:
        target_file.write(target_image)

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
        fps = detect_fps(target_image_path)
        logger.info('fps: ', fps)
        frame_paths = extract_frames(target_image_path, fps)
        logger.info('frames: ', frame_paths)

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
                    input['source_indexes'],
                    input['target_indexes'],
                    input['background_enhance'],
                    input['face_restore'],
                    input['face_upsample'],
                    input['upscale'],
                    input['codeformer_fidelity'],
                    input['output_format']
                )

                # Convert base64 to image
                result_image_bytes = base64.b64decode(result_image_base64)
                result_image = Image.open(io.BytesIO(result_image_bytes))

                # Save the result image as JPEG
                result_image_path = os.path.splitext(frame_path)[0] + '.jpg'
                result_image.save(result_image_path, format='JPEG')

        
        # create video
        logger.info('create_video')
        create_video(frame_paths, target_image_path, fps)
        logger.info('create_video success')

       ## clean_up_temporary_files(source_image_path, clean_up_temporary_files, clean_up_temporary_files)

        return {
            'image': result_image_base64
        }
    except Exception as e:
        logger.error(f'An exception was raised: {e}')
        clean_up_temporary_files(source_image_path, target_image_path)

        return {
            'error': str(e),
            'output': traceback.format_exc(),
            'refresh_worker': True
        }


# ---------------------------------------------------------------------------- #
# RunPod Handler                                                               #
# ---------------------------------------------------------------------------- #
def handler(event):
    validated_input = validate(event['input'], INPUT_SCHEMA)

    if 'errors' in validated_input:
        return {
            'error': validated_input['errors']
        }

    return face_swap_api(validated_input['validated_input'])


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL = os.path.join(script_dir, FACE_SWAP_MODEL)
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), MODEL)
    logger.info(f'Face swap model: {MODEL}')

    if torch.cuda.is_available():
        TORCH_DEVICE = 'cuda'
    else:
        TORCH_DEVICE = 'cpu'

    logger.info(f'Torch device: {TORCH_DEVICE.upper()}')
    FACE_ANALYSER = get_face_analyser(MODEL, TORCH_DEVICE)
    FACE_SWAPPER = get_face_swap_model(model_path)

    # Ensure that CodeFormer weights have been successfully downloaded,
    # otherwise download them
    check_ckpts()

    logger.info('Setting upsampler to RealESRGAN_x2plus')
    upsampler = set_realesrgan()
    CODEFORMER_DEVICE = torch.device(TORCH_DEVICE)

    CODEFORMER_NET = ARCH_REGISTRY.get('CodeFormer')(
        dim_embd=512,
        codebook_size=1024,
        n_head=8,
        n_layers=9,
        connect_list=['32', '64', '128', '256'],
    ).to(CODEFORMER_DEVICE)

    ckpt_path = os.path.join(script_dir, 'CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth')
    logger.info(f'Loading CodeFormer model: {ckpt_path}')
    codeformer_checkpoint = torch.load(ckpt_path)['params_ema']
    CODEFORMER_NET.load_state_dict(codeformer_checkpoint)
    CODEFORMER_NET.eval()

    logger.info('Starting RunPod Serverless...')
    runpod.serverless.start(
        {
            'handler': handler
        }
    )