import os
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import cv2
from PIL import Image

app = Flask(__name__)

# Configure Ffolder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_edge_map(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.medianBlur(gray, 7)
    edges = cv2.Canny(blurred, 100, 200)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(edges)

def apply_scanner_darkly_ai(input_path, output_path, prompt, negative_prompt, steps, seed):
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16
    )

    # Optimize performance
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_xformers_memory_efficient_attention()
    pipe.to("cuda")

    init_image = load_image(input_path).resize((512, 512))
    control_image = generate_edge_map(input_path).resize((512, 512))

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        generator=torch.manual_seed(seed) if seed else None,
        image=control_image,
        height=512,
        width=512,
    ).images[0]

    result.save(output_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Get the form data
        prompt = request.form['prompt']
        negative_prompt = request.form.get('negative_prompt', '')  # Default to empty string if not provided
        steps = int(request.form['steps'])
        seed = 0  # Default to 0 if not provided

        # Process the image and generate the result
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"output_{filename}")
        apply_scanner_darkly_ai(file_path, output_path, prompt, negative_prompt, steps, seed)

        # Send the result back to the user
        return send_from_directory(app.config['UPLOAD_FOLDER'], f"output_{filename}", as_attachment=True)

    return "Invalid file type", 400

if __name__ == '__main__':
    app.run(debug=True)
