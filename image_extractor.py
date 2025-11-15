"""
Module for extracting text from receipt images using Qwen3-VL model.
Handles batch processing of images from a directory.
"""
import os
import time
from pathlib import Path
from typing import Dict
from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


class ImageExtractor:
    """Handles text extraction from receipt images."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct", 
                 max_resolution: int = 3024, dtype=torch.float16, max_new_tokens: int = 1024):
        """
        Initialize the image extractor with the vision-language model.
        
        Args:
            model_name: HuggingFace model identifier
            max_resolution: Maximum image resolution for resizing
            dtype: Model data type (torch.float16 or torch.float32)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU")
        
        print("Loading model...")
        start_time = time.time()
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto"
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.max_resolution = max_resolution
        self.max_new_tokens = max_new_tokens
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from a single image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Extracted text as a string
        """
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        print(f"  Original image size: {original_size}")

        image.thumbnail((self.max_resolution, self.max_resolution), Image.Resampling.LANCZOS)
        resized_size = image.size
        print(f"  Resized image size: {resized_size}")
        
        prompt = """Read all the text in this receipt image. Include every detail:
- Merchant name and address
- Date and time
- Item descriptions and prices
- Subtotal, discounts, taxes
- Total amount
- Payment method (cash, card, etc.)
- Cash tendered and change (if applicable)
- Transaction IDs and reference numbers
- Any other text visible on the receipt

Extract all text exactly as it appears, line by line."""
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.device)
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return "\n".join(output_text)
    
    def extract_text_from_directory(self, directory: str, 
                                     output_dir: str = "extracted_texts",
                                     image_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp")) -> Dict[str, str]:
        """
        Extract text from all images in a directory.
        
        Args:
            directory: Directory containing receipt images
            output_dir: Directory to save extracted text files
            image_extensions: Tuple of valid image file extensions
            
        Returns:
            Dictionary mapping image filenames to extracted text
        """
        directory_path = Path(directory)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(directory_path.glob(f"*{ext}"))
            image_files.extend(directory_path.glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"No image files found in {directory}")
            return {}
        
        print(f"Found {len(image_files)} image(s) to process")
        
        extracted_texts = {}
        total_start = time.time()
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\nProcessing {i}/{len(image_files)}: {image_path.name}")
            start_time = time.time()
            
            try:
                text = self.extract_text_from_image(str(image_path))
                extracted_texts[image_path.name] = text
                text_file = output_path / f"{image_path.stem}.txt"
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write(text)
                elapsed = time.time() - start_time
                print(f"  Extracted text in {elapsed:.2f} seconds")
                print(f"  Saved to {text_file}")
                
            except Exception as e:
                print(f"  Error processing {image_path.name}: {e}")
                extracted_texts[image_path.name] = ""
        
        total_elapsed = time.time() - total_start
        print(f"\nTotal extraction time: {total_elapsed:.2f} seconds")
        print(f"Extracted texts saved to {output_path}")
        
        return extracted_texts
    
    def cleanup(self):
        """
        Clean up model and free GPU memory.
        Call this when done with extraction to free memory for other operations.
        """
        import gc
        if hasattr(self, 'model'):
            del self.model
        if hasattr(self, 'processor'):
            del self.processor
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("GPU memory cleared")


def extract_texts_from_directory(directory: str, output_dir: str = "extracted_texts") -> Dict[str, str]:
    """
    Convenience function to extract texts from a directory.
    
    Args:
        directory: Directory containing receipt images
        output_dir: Directory to save extracted text files
        
    Returns:
        Dictionary mapping image filenames to extracted text
    """
    extractor = ImageExtractor()
    return extractor.extract_text_from_directory(directory, output_dir)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python image_extractor.py <image_directory> [output_directory]")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "extracted_texts"
    
    extractor = ImageExtractor()
    results = extractor.extract_text_from_directory(image_dir, output_dir)
    
    print(f"\nExtracted text from {len(results)} image(s)")

