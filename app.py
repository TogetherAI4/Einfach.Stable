import numpy as np
import gradio as gr
import requests
import time
import json
import base64
import os
from PIL import Image
from io import BytesIO


class Prodia:
    def __init__(self, api_key, base=None):
        self.base = base or "https://api.prodia.com/v1"
        self.headers = {
            "X-Prodia-Key": api_key
        }

    def generate(self, params):
        response = self._post(f"{self.base}/sdxl/generate", params)
        return response.json()

    def get_job(self, job_id):
        response = self._get(f"{self.base}/job/{job_id}")
        return response.json()

    def wait(self, job):
        job_result = job

        while job_result['status'] not in ['succeeded', 'failed']:
            time.sleep(0.25)
            job_result = self.get_job(job['job'])

        return job_result

    def list_models(self):
        response = self._get(f"{self.base}/sdxl/models")
        return response.json()

    def list_samplers(self):
        response = self._get(f"{self.base}/sdxl/samplers")
        return response.json()

    def _post(self, url, params):
        headers = {
            **self.headers,
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers, data=json.dumps(params))

        if response.status_code != 200:
            raise Exception(f"Bad Prodia Response: {response.status_code}")

        return response

    def _get(self, url):
        response = requests.get(url, headers=self.headers)

        if response.status_code != 200:
            raise Exception(f"Bad Prodia Response: {response.status_code}")

        return response


def image_to_base64(image_path):
    # Open the image with PIL
    with Image.open(image_path) as image:
        # Convert the image to bytes
        buffered = BytesIO()
        image.save(buffered, format="PNG")  # You can change format to PNG if needed

        # Encode the bytes to base64
        img_str = base64.b64encode(buffered.getvalue())

    return img_str.decode('utf-8')  # Convert bytes to string



prodia_client = Prodia(api_key=os.getenv("PRODIA_API_KEY"))

def flip_text(prompt, negative_prompt, model, steps, sampler, cfg_scale, width, height, seed):
    result = prodia_client.generate({
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "model": model,
        "steps": steps,
        "sampler": sampler,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height,
        "seed": seed
    })

    job = prodia_client.wait(result)

    return job["imageUrl"]

css = """
#generate {
    height: 100%;
}
"""
with gr.Blocks(css=css, theme='syddharth/gray-minimal') as demo:


    with gr.Row():
        with gr.Column(scale=6):
            model = gr.Dropdown(interactive=True,value="sd_xl_base_1.0.safetensors [be9edd61]", show_label=True, label="Stable Diffusion Checkpoint", choices=prodia_client.list_models())

        with gr.Column(scale=1):
            gr.Markdown(elem_id="powered-by-prodia", value="AUTOMATIC1111 Stable Diffusion Web UI for SDXL V1.0.<br>Powered by [Prodia](https://prodia.com).")

    with gr.Tab("txt2img"):
        with gr.Row():
            with gr.Column(scale=6, min_width=600):
                prompt = gr.Textbox("space warrior, beautiful, female, ultrarealistic, soft lighting, 8k", placeholder="Prompt", show_label=False, lines=3)
                negative_prompt = gr.Textbox(placeholder="Negative Prompt", show_label=False, lines=3, value="3d, cartoon, anime, (deformed eyes, nose, ears, nose), bad anatomy, ugly")
            with gr.Column():
                text_button = gr.Button("Generate", variant='primary', elem_id="generate")

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tab("Generation"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            sampler = gr.Dropdown(value="DPM++ 2M Karras", show_label=True, label="Sampling Method", choices=prodia_client.list_samplers())

                        with gr.Column(scale=1):
                            steps = gr.Slider(label="Sampling Steps", minimum=1, maximum=25, value=20, step=1)

                    with gr.Row():
                        with gr.Column(scale=1):
                            width = gr.Slider(label="Width", minimum=512, maximum=1536, value=1024, step=8)
                            height = gr.Slider(label="Height", minimum=512, maximum=1536, value=1024, step=8)
                            gr.Markdown(elem_id="resolution", value="*Resolution Maximum: 1MP (1048576 px)*")

                        with gr.Column(scale=1):
                            batch_size = gr.Slider(label="Batch Size", maximum=1, value=1)
                            batch_count = gr.Slider(label="Batch Count", maximum=1, value=1)

                    cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=20, value=7, step=1)
                    seed = gr.Number(label="Seed", value=-1)


            with gr.Column(scale=2):
                image_output = gr.Image(value="https://einfachalex.net/wp-content/uploads/2024/03/ba00c3e6292cc2a52a34e7b373a4a9eb.png")

        text_button.click(flip_text, inputs=[prompt, negative_prompt, model, steps, sampler, cfg_scale, width, height, seed], outputs=image_output)

demo.queue(concurrency_count=24, max_size=32, api_open=False).launch(max_threads=128)
