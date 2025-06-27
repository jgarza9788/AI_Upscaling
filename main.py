import argparse
import os
import sys


def upscale_local(image_path: str, output_path: str, scale: int = 4):
    """Upscale using local RealESRGAN model if GPU is available."""
    try:
        import torch
        from PIL import Image
        from realesrgan import RealESRGAN
    except ImportError as exc:
        raise RuntimeError("Required libraries for local upscaling are missing") from exc

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available")

    device = torch.device("cuda")
    model = RealESRGAN(device, scale=scale)
    try:
        # This will download weights if they are not present
        model.load_weights(f"RealESRGAN_x{scale}.pth")
    except Exception:
        # Attempt to use the default weights path from the package
        model.load_weights()

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        sr_image = model.predict(img)
        sr_image.save(output_path)


def upscale_via_api(image_path: str, output_path: str, api_endpoint: str, api_key: str | None = None):
    import requests
    """Fallback method that sends the image to an external API."""
    headers = {}
    if api_key:
        headers['api-key'] = api_key

    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(api_endpoint, files=files, headers=headers, timeout=60)

    response.raise_for_status()
    data = response.json()

    output_url = data.get('output_url') or data.get('url')
    if not output_url:
        raise RuntimeError('API did not return an output URL')

    r = requests.get(output_url, stream=True, timeout=60)
    r.raise_for_status()
    with open(output_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upscale an image using local GPU or fallback API.",
        epilog="Example: python main.py image.jpg --scale 2",
    )
    parser.add_argument('image', nargs='?', help='Path to the input image')
    parser.add_argument('-o', '--output', help='Output path for the upscaled image')
    parser.add_argument('--scale', type=int, default=4, help='Upscaling factor for the local model')
    parser.add_argument('--api-endpoint', default=os.getenv('UPSCALE_API_ENDPOINT', 'https://api.deepai.org/api/torch-srgan'), help='URL of the fallback API')
    parser.add_argument('--api-key', default=os.getenv('UPSCALE_API_KEY'), help='API key for the fallback API')
    return parser.parse_args(), parser


def main(path_to_file: str | None = None):
    args, parser = parse_args()

    image_path = path_to_file or args.image
    if not image_path:
        parser.print_help()
        return 1

    if not os.path.exists(image_path):
        print(f"Input image {image_path!r} not found.")
        return 1

    output_path = args.output
    if not output_path:
        name, ext = os.path.splitext(image_path)
        output_path = f"{name}_upscaled{ext}"

    try:
        upscale_local(image_path, output_path, scale=args.scale)
        print(f"Image upscaled using local GPU: {output_path}")
    except Exception as exc:
        print(f"Local upscaling failed ({exc}), using API fallback...")
        upscale_via_api(image_path, output_path, args.api_endpoint, args.api_key)
        print(f"Image upscaled using API: {output_path}")


if __name__ == '__main__':
    raise SystemExit(main())
