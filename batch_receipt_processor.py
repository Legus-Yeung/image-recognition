"""
Main script for batch processing receipt images.
Orchestrates image extraction, text processing, and CSV export.
"""
import argparse
import sys
import gc
from pathlib import Path
import torch
from image_extractor import ImageExtractor
from text_processor import TextProcessor
from csv_exporter import CSVExporter


def main():
    """Main function to process receipt images and export to CSV."""
    parser = argparse.ArgumentParser(
        description="Batch process receipt images to extract text and export to CSV"
    )
    parser.add_argument(
        "image_directory",
        type=str,
        help="Directory containing receipt images"
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="receipts.csv",
        help="Output CSV file path (default: receipts.csv)"
    )
    parser.add_argument(
        "--extracted-texts-dir",
        type=str,
        default="extracted_texts",
        help="Directory to save extracted text files (default: extracted_texts)"
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip image extraction step (use existing extracted texts)"
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip text processing step (use existing processed data)"
    )
    parser.add_argument(
        "--max-resolution",
        type=int,
        default=3024,
        help="Maximum image resolution for resizing (default: 3024)"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate (default: 1024)"
    )
    
    args = parser.parse_args()
    
    image_dir = Path(args.image_directory)
    if not image_dir.exists():
        print(f"Error: Directory {image_dir} does not exist")
        sys.exit(1)
    
    extracted_texts_dir = Path(args.extracted_texts_dir)
    
    extracted_texts = {}
    if not args.skip_extraction:
        print("=" * 60)
        print("STEP 1: Extracting text from images")
        print("=" * 60)
        extractor = ImageExtractor(
            max_resolution=args.max_resolution,
            max_new_tokens=args.max_new_tokens
        )
        extracted_texts = extractor.extract_text_from_directory(
            str(image_dir),
            str(extracted_texts_dir)
        )
        print("\nCleaning up image extractor to free GPU memory...")
        extractor.cleanup()
        del extractor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("GPU cache cleared")
    else:
        print("=" * 60)
        print("STEP 1: Skipping extraction (using existing texts)")
        print("=" * 60)
        for text_file in extracted_texts_dir.glob("*.txt"):
            with open(text_file, "r", encoding="utf-8") as f:
                extracted_texts[text_file.stem] = f.read()
        print(f"Loaded {len(extracted_texts)} existing text file(s)")
    
    if not extracted_texts:
        print("No extracted texts found. Exiting.")
        sys.exit(1)
    
    processed_data = []
    if not args.skip_processing:
        print("\n" + "=" * 60)
        print("STEP 2: Processing extracted texts")
        print("=" * 60)
        processor = TextProcessor()
        processed_data = processor.process_texts_batch(extracted_texts)
    else:
        print("\n" + "=" * 60)
        print("STEP 2: Skipping text processing")
        print("=" * 60)
        print("Note: You need to provide processed data manually for CSV export")
        sys.exit(1)
    
    if not processed_data:
        print("No processed data found. Exiting.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("STEP 3: Exporting to CSV")
    print("=" * 60)
    exporter = CSVExporter(args.output_csv)
    exporter.export_to_csv(processed_data, include_image_filename=True)
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)
    print(f"Extracted texts saved to: {extracted_texts_dir}")
    print(f"CSV exported to: {args.output_csv}")


if __name__ == "__main__":
    main()

