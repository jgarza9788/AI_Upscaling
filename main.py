import argparse
from pathlib import Path
from typing import List, Optional

from PIL import Image
import torch
from realesrgan import RealESRGAN


def upscale_image(input_path: Path, output_path: Optional[Path] = None, *, scale: int = 4, device: Optional[str] = None) -> Path:
    """Upscale an image using Real-ESRGAN.

    Parameters
    ----------
    input_path : Path
        Path to the input image.
    output_path : Optional[Path], optional
        Destination path. If ``None``, ``<input>_upscaled`` is used.
    scale : int, optional
        Upscaling factor, by default 4.
    device : Optional[str], optional
        PyTorch device to run the model on. Defaults to CUDA if available.

    Returns
    -------
    Path
        Path of the saved, upscaled image.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    img = Image.open(input_path).convert("RGB")

    model = RealESRGAN(torch.device(device), scale=scale)
    model.load_weights(f"weights/RealESRGAN_x{scale}plus.pth")

    sr_img = model.predict(img)
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_upscaled{input_path.suffix}")
    sr_img.save(output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upscale an image using Real-ESRGAN",
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
    )
    print(f"Upscaled image saved to {output_path}")


if __name__ == "__main__":
    main()
