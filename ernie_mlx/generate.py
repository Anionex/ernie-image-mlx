"""CLI for ERNIE-Image MLX generation."""
import argparse
from pathlib import Path

from .pipeline import ErnieImagePipeline


def main():
    parser = argparse.ArgumentParser(description="ERNIE-Image MLX Image Generation")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt")
    parser.add_argument("--prompt-file", type=str, default=None, help="Read prompt from file")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--height", type=int, default=768, help="Image height")
    parser.add_argument("--width", type=int, default=432, help="Image width")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps")
    parser.add_argument("--guidance-scale", type=float, default=5.0, help="CFG guidance scale")
    parser.add_argument("--cfg-cutoff", type=float, default=1.0,
                        help="Fraction of steps using CFG (0.0-1.0). E.g. 0.5 = first 50%% uses CFG")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--model-dir", type=str,
                        default="models/PaddlePaddle/ERNIE-Image",
                        help="Path to model directory")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    parser.add_argument("--quantize", type=int, default=0, choices=[0, 4, 8],
                        help="Quantize transformer: 0=none, 4=INT4, 8=INT8")
    args = parser.parse_args()

    # Read prompt from file if specified
    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text().strip()

    pipe = ErnieImagePipeline(args.model_dir)
    pipe.load(quantize_bits=args.quantize)
    image = pipe(
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        cfg_cutoff=args.cfg_cutoff,
        seed=args.seed,
    )

    if args.output:
        output_path = args.output
    else:
        output_path = f"output_mlx_{args.width}x{args.height}.png"

    image.save(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
