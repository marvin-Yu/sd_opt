from diffusers import StableDiffusionPipeline
import torch
from time import perf_counter

# 指定模型目录:
# model_id = "runwayml/stable-diffusion-v1-5"
model_id = "/data/models/stable-diffusion-v1-5"

prompt = "a photo of an astronaut riding a horse on mars"

Height = 512
Width = 512
step = 20

enable_ipex = True

if enable_ipex:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, custom_pipeline="stable_diffusion_ipex")
    pipe.prepare_for_ipex(prompt, dtype=torch.bfloat16, height=Height, width=Width)
else:
    pipe = StableDiffusionPipeline.from_pretrained(model_id)

start_time = perf_counter()

if enable_ipex:
    with torch.no_grad(), torch.cpu.amp.autocast(dtype=torch.bfloat16):
        image = pipe(prompt, num_inference_steps=step, height=Height, width=Width).images[0]
else:
    with torch.no_grad():
        image = pipe(prompt, num_inference_steps=step, height=Height, width=Width).images[0]

latency = perf_counter() - start_time
print(f">>> total time = {latency} s, time of step = {latency/step} s")
image.save("astronaut_rides_horse.png")
