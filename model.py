from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import time

# Check PyTorch and CUDA compatibility at https://pytorch.org/get-started/previous-versions/
if not torch.cuda.is_available():
    raise RuntimeError("CUDA not available! Install GPU-enabled PyTorch.")

device = torch.device("cuda")
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

start_time = time.time()
print("Loading model...")
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct",
    dtype=torch.float16,
    device_map="auto"
)
model.to(device)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
print("Model loaded!")
print(f"Model load time: {time.time() - start_time:.2f} seconds")

image_path = "sample.jpg"
image = Image.open(image_path).convert("RGB")

# Resize to reduce VRAM usage. This is due to local machine's VRAM's limitation. If you have more VRAM, you can comment this out.
max_resolution = 1728
image.thumbnail((max_resolution, max_resolution))
print(f"Image size after resize: {image.size}")

# This prompt is tailored for OCR purposes (e.g. receipts), for other purposes, you can change the prompt to "Describe the image in detail".
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Read all the text in this image."},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(device)

print("Running model...")
start_inference = time.time()
generated_ids = model.generate(**inputs, max_new_tokens=512)
inference_time = time.time() - start_inference
print(f"Inference completed in {inference_time:.2f} seconds")

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)

output_text_str = "\n".join(output_text)
with open("example.txt", "w", encoding="utf-8") as f:
    f.write(output_text_str)

print("Extracted text saved to example.txt")
print("\nExtracted text:\n", output_text_str)
