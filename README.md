# Image Recognition

A Python-based application for optical character recognition (OCR) from images using the Qwen3-VL-2B-Instruct vision-language model from Hugging Face.

## Features

- **GPU-accelerated OCR**: Leverages CUDA for fast inference using the lightweight Qwen3-VL-2B-Instruct model
- **Receipt text extraction**: Optimized prompts for extracting text from receipts and documents
- **Automatic image preprocessing**: Resizes images to reduce VRAM usage while maintaining quality
- **Simple API**: Easy-to-use interface for processing images and extracting text

## Requirements

- Python 3.11+
- CUDA-compatible GPU (required for model inference)
- PyTorch with CUDA support

## Installation

1. **Install PyTorch with CUDA support** (adjust CUDA version as needed for your system):
   ```bash
   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   ```

2. **Install project dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your image file in the project directory (or update the `image_path` variable in `model.py`)

2. Run the model:
   ```bash
   python model.py
   ```

3. The extracted text will be saved to `example.txt` and printed to the console.

## Example

### Input
A restaurant receipt image (`sample.jpg`):

![Restaurant Receipt](sample.jpg)

### Output
The extracted text from the receipt:

```
RUBY SOFT
Located in the Garment District
587 King St W
Toronto, ON M5V 1V5
Ruby Scho
Table 30
2025-07-15 4:51 p.m.
Server: Monikha S
Check #53
Seat 3
Ordered:
1-HH Stella Draft
$10.00
1 Maguy Smash Burger
$25.00
Baby Gem Caesar
$3.00
1-HH Casamigos Blanco 1oz
$11.00
Subtotal
$49.00
Tax
$6.37
Total
$55.37
Suggested Tip:
15%: (Tip $8.31 Total $63.68)
25%: (Tip $13.84 Total $69.21)
20%: (Tip $11.07 Total $66.44)
18%: (Tip $9.97 Total $65.34)
Tip percentages are based on the check
price after taxes.
Thank you!
Happy Hour 7 days a week 3pm-7pm
Brunch served Sat & Sun 10am-3pm
Keep Portland weird
HST776462277
AMERICAN EXPRESS
CARDS ACCEPTED HERE
```

## Customization

- **Change the image**: Update the `image_path` variable on line 26 of `model.py`
- **Modify the prompt**: Change the prompt on line 40 to extract different information (e.g., "Describe the image in detail" for general image description)
- **Adjust image resolution**: Modify `max_resolution` on line 30 to change the maximum image size (useful for managing VRAM usage)

## How It Works

The application uses the Qwen3-VL-2B-Instruct model, a lightweight vision-language model that can understand both images and text prompts. The model processes the input image and generates text based on the provided prompt. For OCR tasks, the prompt "Read all the text in this image" is used to extract all visible text.

The model is loaded with:
- `float16` precision for reduced memory usage
- Automatic device mapping for optimal GPU utilization
- Image resizing to manage VRAM constraints

## Output

The extracted text is saved to `example.txt` in UTF-8 encoding. The console output includes:
- Model loading time
- Image processing information
- Inference time
- Extracted text preview

## Google Vision Utility

The `google_vision_util/` directory contains a small utility script for extracting text from Google Vision API JSON responses. This can be useful if you're working with Google Cloud Vision API outputs and need to parse the JSON format. See `google_vision_util/extract_text.py` for usage details.

## Notes

- **GPU Required**: This project requires a CUDA-compatible GPU. The model will not run on CPU-only systems.
- **VRAM Usage**: The default image resolution is limited to 1728px to reduce VRAM usage. If you have more VRAM available, you can increase or remove this limit.
- **Model Size**: The Qwen3-VL-2B-Instruct model is relatively lightweight (~2B parameters) but still requires significant GPU memory.

## License

See LICENSE file for details.
