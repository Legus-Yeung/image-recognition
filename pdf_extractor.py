"""
Module for extracting text from PDF files where each page contains an image.
Extracts images from PDF pages and processes them using the ImageExtractor.
"""
import os
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
import torch
from pdf2image import convert_from_path
from image_extractor import ImageExtractor


class PDFExtractor:
    """Handles text extraction from PDF files with images on each page."""
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct", 
                 max_resolution: int = 3024, dtype=torch.float16, max_new_tokens: int = 1024):
        """
        Initialize the PDF extractor with the vision-language model.
        
        Args:
            model_name: HuggingFace model identifier
            max_resolution: Maximum image resolution for resizing
            dtype: Model data type (torch.float16 or torch.float32)
            max_new_tokens: Maximum number of tokens to generate
        """
        self.image_extractor = ImageExtractor(
            model_name=model_name,
            max_resolution=max_resolution,
            dtype=dtype,
            max_new_tokens=max_new_tokens
        )
    
    def extract_images_from_pdf(self, pdf_path: str, 
                                output_dir: Optional[str] = None,
                                dpi: int = 300) -> List[Path]:
        """
        Extract images from each page of a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images (None for temp directory)
            dpi: Resolution for PDF to image conversion
            
        Returns:
            List of paths to extracted image files
        """
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if output_dir is None:
            # Use temporary directory
            temp_dir = tempfile.mkdtemp(prefix="pdf_images_")
            output_path = Path(temp_dir)
        else:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Converting PDF pages to images (DPI: {dpi})...")
        start_time = time.time()
        
        try:
            # Convert PDF pages to images
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                output_folder=None,
                fmt='png',
                thread_count=1
            )
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to images: {e}")
        
        print(f"Converted {len(images)} page(s) in {time.time() - start_time:.2f} seconds")
        
        # Save images to files
        image_paths = []
        pdf_stem = pdf_path_obj.stem
        
        for i, image in enumerate(images, 1):
            image_filename = f"{pdf_stem}_page_{i:03d}.png"
            image_path = output_path / image_filename
            image.save(image_path, "PNG")
            image_paths.append(image_path)
            print(f"  Saved page {i} to {image_path}")
        
        return image_paths
    
    def extract_text_from_pdf(self, pdf_path: str,
                              output_dir: str = "extracted_texts",
                              temp_image_dir: Optional[str] = None,
                              dpi: int = 300,
                              keep_images: bool = False) -> Dict[str, str]:
        """
        Extract text from all pages of a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted text files
            temp_image_dir: Directory to save extracted images (None for temp directory)
            dpi: Resolution for PDF to image conversion
            keep_images: Whether to keep extracted images after processing
            
        Returns:
            Dictionary mapping page numbers to extracted text
        """
        pdf_path_obj = Path(pdf_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing PDF: {pdf_path_obj.name}")
        print(f"Output directory: {output_path}")
        
        # Extract images from PDF
        image_paths = self.extract_images_from_pdf(
            pdf_path,
            output_dir=temp_image_dir,
            dpi=dpi
        )
        
        if not image_paths:
            print("No images extracted from PDF")
            return {}
        
        print(f"\nFound {len(image_paths)} page(s) to process")
        
        extracted_texts = {}
        total_start = time.time()
        
        # Process each page image
        for i, image_path in enumerate(image_paths, 1):
            page_num = i
            print(f"\nProcessing page {page_num}/{len(image_paths)}: {image_path.name}")
            start_time = time.time()
            
            try:
                text = self.image_extractor.extract_text_from_image(str(image_path))
                page_key = f"page_{page_num:03d}"
                extracted_texts[page_key] = text
                
                # Save text to file
                text_file = output_path / f"{pdf_path_obj.stem}_page_{page_num:03d}.txt"
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write(text)
                
                elapsed = time.time() - start_time
                print(f"  Extracted text in {elapsed:.2f} seconds")
                print(f"  Saved to {text_file}")
                
            except Exception as e:
                print(f"  Error processing page {page_num}: {e}")
                extracted_texts[f"page_{page_num:03d}"] = ""
        
        # Clean up extracted images if not keeping them
        if not keep_images and temp_image_dir is None:
            print(f"\nCleaning up temporary image files...")
            for image_path in image_paths:
                try:
                    image_path.unlink()
                except Exception as e:
                    print(f"  Warning: Could not delete {image_path}: {e}")
            
            # Try to remove temp directory if it's empty
            if image_paths:
                temp_dir = image_paths[0].parent
                try:
                    if temp_dir.exists() and not any(temp_dir.iterdir()):
                        temp_dir.rmdir()
                except Exception:
                    pass
        
        total_elapsed = time.time() - total_start
        print(f"\nTotal extraction time: {total_elapsed:.2f} seconds")
        print(f"Extracted texts saved to {output_path}")
        
        return extracted_texts
    
    def cleanup(self):
        """
        Clean up model and free GPU memory.
        Call this when done with extraction to free memory for other operations.
        """
        self.image_extractor.cleanup()


def extract_texts_from_pdf(pdf_path: str, 
                           output_dir: str = "extracted_texts",
                           temp_image_dir: Optional[str] = None,
                           dpi: int = 300,
                           keep_images: bool = False) -> Dict[str, str]:
    """
    Convenience function to extract texts from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save extracted text files
        temp_image_dir: Directory to save extracted images (None for temp directory)
        dpi: Resolution for PDF to image conversion
        keep_images: Whether to keep extracted images after processing
        
    Returns:
        Dictionary mapping page numbers to extracted text
    """
    extractor = PDFExtractor()
    try:
        return extractor.extract_text_from_pdf(
            pdf_path,
            output_dir,
            temp_image_dir,
            dpi,
            keep_images
        )
    finally:
        extractor.cleanup()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract text from PDF files where each page contains an image"
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to the PDF file"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="extracted_texts",
        help="Directory to save extracted text files (default: extracted_texts)"
    )
    parser.add_argument(
        "-i", "--image-dir",
        type=str,
        default=None,
        help="Directory to save extracted images (default: temporary directory)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF to image conversion (default: 300)"
    )
    parser.add_argument(
        "--keep-images",
        action="store_true",
        help="Keep extracted images after processing"
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=3024,
        help="Maximum image resolution for model processing (default: 3024)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate (default: 1024)"
    )
    
    args = parser.parse_args()
    
    extractor = PDFExtractor(
        max_resolution=args.max_resolution,
        max_new_tokens=args.max_new_tokens
    )
    
    try:
        results = extractor.extract_text_from_pdf(
            args.pdf_path,
            args.output_dir,
            args.image_dir,
            args.dpi,
            args.keep_images
        )
        
        print(f"\nExtracted text from {len(results)} page(s)")
    finally:
        extractor.cleanup()

