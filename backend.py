import base64
from flask import Flask, request, jsonify, send_from_directory
from io import BytesIO
import torch
from PIL import Image
import os
from typing import Tuple, Optional
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image
import random
from flask_cors import CORS
import io


#import transformers
# ... import diffusers pipeline and other code from the improved Gradio example
def resize_image(image: Image, output_size: Tuple[int, int] =(1024, 576)):
    # Calculate aspect ratios
    target_aspect = output_size[0] / output_size[1]  # Aspect ratio of the desired size
    image_aspect = image.width / image.height  # Aspect ratio of the original image

    # Resize then crop if the original image is larger
    if image_aspect > target_aspect:
        # Resize the image to match the target height, maintaining aspect ratio
        new_height = output_size[1]
        new_width = int(new_height * image_aspect)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # Calculate coordinates for cropping
        left = (new_width - output_size[0]) / 2
        top = 0
        right = (new_width + output_size[0]) / 2
        bottom = output_size[1]
    else:
        # Resize the image to match the target width, maintaining aspect ratio
        new_width = output_size[0]
        new_height = int(new_width / image_aspect)
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        # Calculate coordinates for cropping
        left = 0
        top = (new_height - output_size[1]) / 2
        right = output_size[0]
        bottom = (new_height + output_size[1]) / 2

    # Crop the image
    cropped_image = resized_image.crop((left, top, right, bottom))
    # set correct image mode
    if cropped_image.mode == "RGBA":
        cropped_image = cropped_image.convert("RGB")

    return cropped_image
app = Flask(__name__)
max_64_bit_int = 2 ** 63 - 1
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt",
    variant="fp16"
)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)
pipe.enable_model_cpu_offload()
# ... (include all the functions from the improved Gradio example)
#     resize_image, sample, etc.
def sample(
    image: Image,
    seed: Optional[int] = 42,
    randomize_seed: bool = True,
    motion_bucket_id: int = 127,
    fps_id: int = 6,
    version: str = "svd_xt",
    cond_aug: float = 0.02,
    decoding_t: int = 3,  
    device: str = "cpu",  # Force CPU usage
    output_folder: str = output_folder,
):

    if randomize_seed:
        seed = random.randint(0, max_64_bit_int)

    files = os.listdir(output_folder) 
    base_count = len([f for f in files if f.endswith('.mp4')])  # Get the count
    video_path = os.path.join(output_folder, f"{base_count:06d}.mp4") 


    frames = pipe(
        image,
        decode_chunk_size=decoding_t,
        generator=torch.manual_seed(seed),
        motion_bucket_id=motion_bucket_id,
        noise_aug_strength=0.1,
        num_frames=25,
    ).frames[0]

    export_to_video(frames, video_path, fps=fps_id)

    return video_path, seed


    
def generate(image, seed, randomize_seed, motion_bucket_id, fps_id):
    img = resize_image(image, output_size=(1024, 576))
    video, seed = sample(img, seed, randomize_seed, motion_bucket_id, fps_id)
    return video, seed
@app.route('/')  # Changed route
def index():
    return send_from_directory(app.static_folder, 'index.html')
@app.route('/main.js') 
def main_js():
    return send_from_directory(app.static_folder, 'main.js')
@app.route('/generate_video', methods=['POST'])
def generate_video():
    data = request.get_json()

    image_data = str(data['imageData'])
    with open('image_data.txt', 'w') as f:
        f.write(image_data)
    seed = int(data['seed'])
    motion_bucket_id = int(data['motionId'])
    fps_id = int(data['fps'])

    print(f"Image data: {image_data[:100]}...")  # Print the first 100 characters of the image data
    print(f"Seed: {seed}")  # Print the seed
    print(f"Motion bucket ID: {motion_bucket_id}")  # Print the motion bucket ID
    print(f"FPS ID: {fps_id}")  # Print the FPS ID

    try:
        # Decode the image from base64
        image_data = image_data.split(',')[1]
        decoded_image_data = base64.b64decode(image_data)

        image = Image.open(io.BytesIO(decoded_image_data))
        image.save("output.jpg")
        image.show()    
        video_path, seed = generate(image, seed, False, motion_bucket_id, fps_id) 

        # Encode the video as base64
        with open(video_path, "rb") as f:
            video_data = base64.b64encode(f.read()).decode('utf-8')

        return jsonify({ 
            'video': video_data,
            'seed': seed
        })

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({'error': 'Error generating video'}), 500


if __name__ == '__main__':
    app.run(debug=True) 
    CORS(app)
