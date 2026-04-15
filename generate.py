import sys
import argparse
import torch
from diffusers import ErnieImagePipeline

SIZES = {
    "1:1":  (1024, 1024),
    "16:9": (1264, 848),
    "9:16": (848, 1264),
    "4:3":  (1200, 896),
    "3:4":  (896, 1200),
}

parser = argparse.ArgumentParser(description="ERNIE-Image local generator")
parser.add_argument("prompt", nargs="?", default="a serene mountain landscape at sunset, photorealistic, golden hour lighting")
parser.add_argument("--steps", type=int, default=50)
parser.add_argument("--ratio", default="1:1", choices=SIZES.keys())
parser.add_argument("--out", default=None)
parser.add_argument("--guidance", type=float, default=4.0)
parser.add_argument("--no-pe", action="store_true", help="Disable prompt enhancer")
args = parser.parse_args()

width, height = SIZES[args.ratio]
output_path = args.out or f"output_{args.ratio.replace(':','x')}.png"

print(f"Prompt : {args.prompt}")
print(f"Size   : {width}x{height} ({args.ratio})")
print(f"Steps  : {args.steps}")

model_dir = "./models/PaddlePaddle/ERNIE-Image"
print("Loading model...")
load_kwargs = dict(
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
if args.no_pe:
    load_kwargs['pe'] = None  # 跳过加载 PE 模型，省 7GB
pipe = ErnieImagePipeline.from_pretrained(model_dir, **load_kwargs)
pipe.enable_model_cpu_offload(device="mps")

print("Generating...")
image = pipe(
    prompt=args.prompt,
    height=height,
    width=width,
    num_inference_steps=args.steps,
    guidance_scale=args.guidance,
    use_pe=not args.no_pe,
).images[0]

image.save(output_path)
print(f"Saved → {output_path}")
