# AI Upscaling

This repository provides a simple script to upscale images. The script will try to use a local NVIDIA GPU when available via [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN). If a GPU is not available or the necessary libraries are missing, it falls back to a web API.

## Requirements

- Python 3.9+
- [Pillow](https://pillow.readthedocs.io/en/stable/)
- Optional: [PyTorch](https://pytorch.org/) with CUDA and the `realesrgan` package for local upscaling
- Optional: `requests` for API fallback

## Usage

```bash
python3 main.py path/to/image.jpg
```

If run without arguments, the script prints its available options along with an example.

### Options

- `-o`, `--output`: Path to save the upscaled image. Defaults to `input_upscaled.ext`.
- `--scale`: Upscaling factor used by the local model (default: `4`).
- `--api-endpoint`: URL of the fallback API.
- `--api-key`: API key used for the fallback API.

The fallback API requires a valid API key. If the key is missing or invalid,
the script will exit with an error message. Supply the key on the command line
or via the `UPSCALE_API_KEY` environment variable.

Set `UPSCALE_API_ENDPOINT` and `UPSCALE_API_KEY` environment variables to avoid passing them on every run.

```bash
UPSCALE_API_KEY=your-key python3 main.py image.png
```

## Notes

The Real-ESRGAN model weights are not included in this repository. The `realesrgan` library will attempt to download them automatically if it has internet access.
