import asyncio
import os
import subprocess
from typing import List

async def run_ffmpeg(args: List[str]) -> bool:
    commands = ['ffmpeg', '-hide_banner', '-loglevel', 'error']
    commands.extend(args)
    
    # Use asyncio.create_subprocess_exec instead of subprocess.Popen
    process = await asyncio.create_subprocess_exec(*commands, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Read output and error asynchronously
    stdout, stderr = await process.communicate()

    # Check if the process has completed successfully
    if process.returncode == 0:
        return True
    else:
        # Print error output
        print(stderr.decode())
        return False

def detect_fps(target_path: str) -> float:
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', target_path]
    output = subprocess.check_output(command).decode().strip().split('/')
    try:
        numerator, denominator = map(int, output)
        return numerator / denominator
    except Exception:
        pass
    return 30

async def create_video(frame_paths, output_video_path, fps=30):
    output_video_quality = 23  # Adjust as needed
    output_video_encoder = 'libx264'

    # Input pattern for the frames
    input_pattern = os.path.join(os.path.dirname(frame_paths[0]), '%04d.jpg')

    commands = [
        '-r', str(fps),
        '-i', input_pattern,
        '-c:v', output_video_encoder
    ]

    if output_video_encoder in ['libx264', 'libx265', 'libvpx']:
        commands.extend(['-crf', str(output_video_quality)])
    if output_video_encoder in ['h264_nvenc', 'hevc_nvenc']:
        commands.extend(['-cq', str(output_video_quality)])

    commands.extend([
        '-pix_fmt', 'yuv420p',
        '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1',
        '-y', output_video_path
    ])

    await run_ffmpeg(commands)

async def extract_frames(target_path: str, fps: float = 30) -> List[str]:
    temp_frame_quality = 100 * 31 // 100

    # Use the video file's name (without extension) as the base name for the frames directory
    video_name = os.path.splitext(os.path.basename(target_path))[0]
    output_directory = os.path.join(os.path.dirname(target_path), f'{video_name}_frames')
    os.makedirs(output_directory, exist_ok=True)
    
    output_pattern = os.path.join(output_directory, '%04d.jpg')

    await run_ffmpeg(['-hwaccel', 'auto', '-i', target_path, '-q:v', str(temp_frame_quality), '-pix_fmt', 'rgb24', '-vf', f'fps={fps}', output_pattern])

    # List all files in the output directory and return their paths
    image_paths = [os.path.join(output_directory, filename) for filename in os.listdir(output_directory) if filename.endswith('.jpg')]
    sorted_image_paths = sorted(image_paths, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

    return sorted_image_paths