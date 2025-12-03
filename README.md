# Image Recognition - Receipt Processing System

A Python-based system for extracting, processing, and organizing receipt information from images and PDFs using vision-language models.

## Overview

This project provides a complete pipeline for processing receipt images and PDFs:
1. **Extract text** from receipt images using a vision-language model (Qwen3-VL)
2. **Process and categorize** the extracted text to identify dates, amounts, and purchase categories
3. **Export structured data** to CSV format for easy analysis

## Root-Level Scripts

### 1. `batch_receipt_processor.py`

**Main orchestrator script** that coordinates the entire receipt processing workflow.

**Purpose:**
- Processes receipt images in batch from a directory
- Orchestrates the three-step pipeline: extraction → processing → CSV export
- Manages GPU memory efficiently by cleaning up models between steps

**Usage:**
```bash
python batch_receipt_processor.py <image_directory> [options]
```

**Key Features:**
- Extracts text from all images in a directory
- Processes extracted text to identify dates, amounts, and categories
- Exports results to CSV file
- Supports skipping steps (e.g., re-processing existing extracted texts)
- Configurable image resolution and token limits

**Options:**
- `--output-csv`: Output CSV file path (default: `receipts.csv`)
- `--extracted-texts-dir`: Directory to save extracted text files (default: `extracted_texts`)
- `--skip-extraction`: Skip image extraction step (use existing extracted texts)
- `--skip-processing`: Skip text processing step
- `--max-resolution`: Maximum image resolution for resizing (default: 3024)
- `--max-new-tokens`: Maximum number of tokens to generate (default: 1024)

**Example:**
```bash
python batch_receipt_processor.py images/ --output-csv my_receipts.csv
```

---

### 2. `image_extractor.py`

**Module for extracting text from receipt images** using the Qwen3-VL vision-language model.

**Purpose:**
- Extracts all text content from receipt images
- Handles image preprocessing (resizing, format conversion)
- Saves extracted text to individual files

**Key Features:**
- Uses Qwen3-VL-2B-Instruct model for text extraction
- Supports GPU acceleration (CUDA) when available
- Processes images in batch from a directory
- Automatically resizes large images to optimize processing
- Saves extracted text to `.txt` files

**Usage as standalone script:**
```bash
python image_extractor.py <image_directory> [output_directory]
```

**Example:**
```bash
python image_extractor.py images/ extracted_texts/
```

**Class: `ImageExtractor`**
- `extract_text_from_image(image_path)`: Extract text from a single image
- `extract_text_from_directory(directory, output_dir)`: Process all images in a directory
- `cleanup()`: Free GPU memory after processing

---

### 3. `pdf_extractor.py`

**Module for extracting text from PDF files** where each page contains a receipt image.

**Purpose:**
- Converts PDF pages to images
- Extracts text from each page using the image extractor
- Handles multi-page PDF documents

**Key Features:**
- Converts PDF pages to images using `pdf2image`
- Processes each page as a separate receipt
- Supports configurable DPI for PDF conversion
- Optionally keeps or cleans up intermediate image files
- Integrates with `ImageExtractor` for text extraction

**Usage as standalone script:**
```bash
python pdf_extractor.py <pdf_path> [options]
```

**Options:**
- `-o, --output-dir`: Directory to save extracted text files (default: `extracted_texts`)
- `-i, --image-dir`: Directory to save extracted images (default: temporary)
- `--dpi`: DPI for PDF to image conversion (default: 300)
- `--keep-images`: Keep extracted images after processing
- `--max-resolution`: Maximum image resolution for model processing (default: 3024)
- `--max-new-tokens`: Maximum number of tokens to generate (default: 1024)

**Example:**
```bash
python pdf_extractor.py receipts.pdf -o extracted_texts/ --dpi 300
```

**Class: `PDFExtractor`**
- `extract_images_from_pdf(pdf_path, output_dir, dpi)`: Convert PDF pages to images
- `extract_text_from_pdf(pdf_path, output_dir, ...)`: Extract text from all PDF pages
- `cleanup()`: Free GPU memory after processing

---

### 4. `text_processor.py`

**Module for processing extracted receipt text** to extract structured information and categorize purchases.

**Purpose:**
- Extracts date, amount, and category from raw receipt text
- Uses AI model to intelligently categorize purchases
- Handles various date formats and currency symbols

**Key Features:**
- Extracts purchase date (converts to YYYY-MM-DD format)
- Extracts payment amount with currency symbol
- Categorizes purchases into: dining, entertainment, travel, utility, health
- Uses keyword matching and AI model for accurate categorization
- Processes multiple receipts in batch

**Categories:**
- **dining**: Restaurants, cafes, food delivery, bars, fast food, grocery stores
- **entertainment**: Movies, concerts, sports events, games, streaming services, bouldering/climbing gyms, arcades, bowling
- **travel**: Hotels, airlines, trains, buses, car rentals, parking, taxis/rideshares
- **utility**: Electricity, water, gas, internet, phone, cable, rent, property taxes
- **health**: Pharmacies, hospitals, clinics, doctors, dentists, medical supplies, fitness centers

**Usage as standalone script:**
```bash
python text_processor.py <extracted_text_file_or_directory>
```

**Example:**
```bash
python text_processor.py extracted_texts/
```

**Class: `TextProcessor`**
- `process_text(text, image_filename)`: Process a single receipt text
- `process_texts_batch(extracted_texts)`: Process multiple receipts
- `extract_all_info(text)`: Extract date, amount, and category in one call
- `cleanup()`: Free GPU memory after processing

---

### 5. `csv_exporter.py`

**Module for exporting receipt data to CSV format** with intelligent sorting.

**Purpose:**
- Exports processed receipt data to CSV files
- Sorts receipts by category and date
- Provides clean, structured output for analysis

**Key Features:**
- Sorts data by category priority, then by date
- Includes optional image filename column
- Handles missing or incomplete data gracefully
- Can export directly from extracted text directory

**Usage as standalone script:**
```bash
python csv_exporter.py <extracted_texts_directory> [output_csv]
```

**Example:**
```bash
python csv_exporter.py extracted_texts/ receipts.csv
```

**Class: `CSVExporter`**
- `export_to_csv(data, include_image_filename)`: Export data to CSV
- `sort_data(data)`: Sort receipts by category and date
- `export_from_texts_directory(texts_directory, text_processor_func)`: Export from text files

**CSV Columns:**
- `date`: Purchase date in YYYY-MM-DD format
- `amount`: Payment amount with currency symbol (e.g., $45.20)
- `category`: Purchase category (dining, entertainment, travel, utility, health)
- `image_filename`: (optional) Original image filename

---

## Workflow

### Typical Usage Flow

1. **Extract text from images:**
   ```bash
   python image_extractor.py images/ extracted_texts/
   ```

2. **Process extracted text:**
   ```bash
   python text_processor.py extracted_texts/
   ```

3. **Export to CSV:**
   ```bash
   python csv_exporter.py extracted_texts/ receipts.csv
   ```

### Or Use the Batch Processor (Recommended)

```bash
python batch_receipt_processor.py images/ --output-csv receipts.csv
```

This single command performs all three steps automatically.

---

## Requirements

See `requirements.txt` for full dependency list. Key requirements:

- **PyTorch** (with CUDA support recommended)
- **transformers** (Hugging Face)
- **Pillow** (image processing)
- **pdf2image** (for PDF processing)
- **poppler-utils** (system package required for pdf2image)

### Installation Notes

1. Install PyTorch with CUDA support first (check compatibility at https://pytorch.org)
2. Install system dependencies:
   - Ubuntu/Debian: `sudo apt-get install poppler-utils`
   - macOS: `brew install poppler`
3. Install Python packages: `pip install -r requirements.txt`

---

## Directory Structure

```
.
├── batch_receipt_processor.py  # Main orchestrator
├── image_extractor.py         # Image text extraction
├── pdf_extractor.py           # PDF text extraction
├── text_processor.py          # Text processing & categorization
├── csv_exporter.py            # CSV export
├── images/                    # Input receipt images
├── extracted_texts/           # Extracted text files
├── receipts.csv               # Output CSV file
└── requirements.txt           # Python dependencies
```

---

## Notes

- The system uses the **Qwen3-VL-2B-Instruct** model for both image and text processing
- GPU acceleration is recommended for faster processing
- Large images are automatically resized to optimize processing speed
- The batch processor manages GPU memory by cleaning up models between steps
- All scripts can be used standalone or as part of the integrated pipeline

