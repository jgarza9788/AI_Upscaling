import argparse
from pathlib import Path
from typing import List, Optional

from PIL import Image
import torch
try:
    from diffusers import StableDiffusionUpscalePipeline
    HAS_DIFFUSERS = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_DIFFUSERS = False
import os
import requests


def upscale_with_diffusers(
    input_path: Path,
    output_path: Optional[Path] = None,
    *,
    device: Optional[str] = None,
    model_id: str = "stabilityai/stable-diffusion-x4-upscaler",
    num_inference_steps: int = 20,
) -> Path:
    """Upscale an image locally using a Hugging Face model.

    Parameters
    ----------
    input_path : Path
        Path to the input image.
    output_path : Optional[Path], optional
        Destination path. If ``None``, ``<input>_upscaled`` is used.
    device : Optional[str], optional
        PyTorch device to run the model on. Defaults to CUDA if available.
    model_id : str, optional
        Identifier of the Hugging Face model to load.
    num_inference_steps : int, optional
        Number of diffusion steps, by default 20.

    Returns
    -------
    Path
        Path of the saved, upscaled image.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    img = Image.open(input_path).convert("RGB")

    pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id)
    pipe = pipe.to(device)
    result = pipe(prompt="", image=img, num_inference_steps=num_inference_steps)
    sr_img = result.images[0]

    if output_path is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_upscaled{input_path.suffix}"
        )
    sr_img.save(output_path)
    return output_path


def upscale_via_api(
    input_path: Path,
    output_path: Optional[Path],
    *,
    endpoint: str,
    api_key: str,
    scale: int = 4,
) -> Path:
    """Upscale an image using an HTTP API."""
    if output_path is None:
        output_path = input_path.with_name(
            f"{input_path.stem}_upscaled{input_path.suffix}"
        )

    with open(input_path, "rb") as f:
        files = {"image": f}
        data = {"scale": str(scale)}
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.post(endpoint, files=files, data=data, headers=headers)
    response.raise_for_status()
    with open(output_path, "wb") as fout:
        fout.write(response.content)
    return output_path


def upscale_image(
    input_path: Path,
    output_path: Optional[Path] = None,
    *,
    scale: int = 4,
    device: Optional[str] = None,
    api_endpoint: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Path:
    """Upscale an image using a Hugging Face model or an HTTP API."""
    if HAS_DIFFUSERS:
        return upscale_with_diffusers(
            input_path,
            output_path,
            device=device,
        )

    if not api_endpoint or not api_key:
        raise RuntimeError(
            "No local upscaling model available and no API information provided"
        )

    return upscale_via_api(
        input_path,
        output_path,
        endpoint=api_endpoint,
        api_key=api_key,
        scale=scale,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upscale an image using a local Hugging Face model or an API",
        add_help=False,
    )
    parser.add_argument("input", nargs="?", type=Path, help="Path to the input image")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to save the upscaled image (default: <input>_upscaled.ext)",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Upscaling factor used by the model (default: 4)",
    )
    parser.add_argument(
        "--api-endpoint",
        help="URL of the fallback API",
        default=os.environ.get("UPSCALE_API_ENDPOINT"),
    )
    parser.add_argument(
        "--api-key",
        help="API key used for the fallback API",
        default=os.environ.get("UPSCALE_API_KEY"),
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        help="Device used to run the model (default: auto)",
    )
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.input is None:
        parser.print_help()
        print(
            "\nExamples:\n"
            "  python3 main.py image.jpg\n"
            "  python3 main.py image.jpg --scale 2 -o output.png"
        )
        return

    output_path = upscale_image(
        args.input,
        args.output,
        scale=args.scale,
        device=args.device,
        api_endpoint=args.api_endpoint,
        api_key=args.api_key,
    )
    print(f"Upscaled image saved to {output_path}")


if __name__ == "__main__":
    main()
