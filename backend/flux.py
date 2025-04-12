from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from gradio_client import Client
import os
import tempfile
import base64
import time
import shutil
import requests
from urllib.parse import urlparse
from pathlib import Path

app = Flask(__name__)

CORS(app, resources={
    r"/llm": {
        "origins": "http://localhost:3000",
        "methods": ["POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

def generate_image_from_prompt(prompt):
    try:

        client = Client("https://52b7cbeeeb47a101eb.gradio.live/")

        result = client.predict(
            prompt,  # Use the prompt from the request
            512,     # width
            512,     # height
            0,       # seed
            20,      # steps
            "euler", # sampler
            "normal", # scheduler
            7.5,     # guidance scale
            0,       # negative prompt weight
            0,       # style strength
            fn_index=0,
            timeout=60
        )
        
        print(f"Gradio result: {result}")
        
        image_path = result[0]

        temp_dir = tempfile.mkdtemp()
        local_image_path = os.path.join(temp_dir, f"generated_image_{int(time.time())}.png")
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path, stream=True)
            if response.status_code == 200:
                with open(local_image_path, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
            else:
                raise Exception(f"Failed to download image from URL: {response.status_code}")
        else:
            try:
                shutil.copy(image_path, local_image_path)
            except Exception as copy_error:
                if os.path.exists(image_path):
                    with open(image_path, 'rb') as src, open(local_image_path, 'wb') as dst:
                        dst.write(src.read())
                else:
                    raise Exception(f"Cannot access the image at path: {image_path}")
        with open(local_image_path, "rb") as img_file:
            img_data = img_file.read()
            encoded_img = base64.b64encode(img_data).decode('utf-8')
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        
        return {
            "success": True,
            "image": encoded_img,
            "message": f"Image generated successfully for prompt: {prompt}"
        }
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return {
            "success": False,
            "image": None,
            "message": f"Error generating image: {str(e)}"
        }

@app.route('/llm', methods=['POST', 'OPTIONS'])
def process_text():
    if request.method == 'OPTIONS':
        return jsonify({}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
