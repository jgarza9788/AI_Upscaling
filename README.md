# AI Upscaling

This repository provides a simple script to upscale images. When PyTorch and the `diffusers` library are available it uses the [`stable-diffusion-x4-upscaler`](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler) model locally. If the model or its dependencies are missing, it falls back to a web API.

## Requirements

- Python 3.9+
- [Pillow](https://pillow.readthedocs.io/en/stable/)
- Optional: [PyTorch](https://pytorch.org/) with CUDA and the [`diffusers`](https://github.com/huggingface/diffusers) package for local upscaling
- Optional: `requests` for API fallback

## Usage

```bash
python3 main.py path/to/image.jpg
```

If run without arguments, the script prints its available options along with an example.

### Options

- `-o`, `--output`: Path to save the upscaled image. Defaults to `input_upscaled.ext`.
- `--scale`: Upscaling factor (used only with the API). The built-in model upscales by a fixed factor of `4`.
- `--api-endpoint`: URL of the fallback API.
- `--api-key`: API key used for the fallback API.

The fallback API requires a valid API key. If the key is missing or invalid,
the script will exit with an error message. Supply the key on the command line
or via the `UPSCALE_API_KEY` environment variable.

Set `UPSCALE_API_ENDPOINT` and `UPSCALE_API_KEY` environment variables to avoid passing them on every run. If the local model is unavailable, the script automatically uses the API when these values are provided.

```bash
UPSCALE_API_KEY=your-key python3 main.py image.png
```

## Notes

The Stable Diffusion upscaler weights are downloaded automatically from the Hugging Face Hub on first use if internet access is available.
