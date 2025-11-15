"""
Module for exporting receipt data to CSV format.
Handles sorting by category and date.
"""
import csv
from typing import List, Dict
from pathlib import Path


class CSVExporter:
    """Handles exporting receipt data to CSV files."""
    
    CSV_COLUMNS = ["date", "amount", "category"]
    
    def __init__(self, output_path: str = "receipts.csv"):
        """
        Initialize the CSV exporter.
        
        Args:
            output_path: Path to the output CSV file
        """
        self.output_path = Path(output_path)
    
    def sort_data(self, data: List[Dict]) -> List[Dict]:
        """
        Sort data by category, then by date.
        
        Args:
            data: List of dictionaries with date, amount, and category
            
        Returns:
            Sorted list of dictionaries
        """
        def sort_key(item: Dict) -> tuple:
            category = item.get("category", "").lower()
            date = item.get("date", "")
            
            category_order = {
                "dining": 0,
                "entertainment": 1,
                "travel": 2,
                "utility": 3,
                "health": 4
            }
            category_priority = category_order.get(category, 99)
            
            return (category_priority, date or "9999-99-99")
        
        return sorted(data, key=sort_key)
    
    def export_to_csv(self, data: List[Dict], include_image_filename: bool = False) -> None:
        """
        Export receipt data to CSV file.
        
        Args:
            data: List of dictionaries with date, amount, and category
            include_image_filename: Whether to include image filename column
        """
        if not data:
            print("No data to export")
            return
        
        sorted_data = self.sort_data(data)
        
        columns = self.CSV_COLUMNS.copy()
        if include_image_filename:
            columns.append("image_filename")
        
        with open(self.output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            
            for row in sorted_data:
                csv_row = {col: row.get(col, "") for col in columns}
                writer.writerow(csv_row)
        
        print(f"Exported {len(sorted_data)} receipt(s) to {self.output_path}")
        print(f"Sorted by category, then by date")
    
    def export_from_texts_directory(self, texts_directory: str, 
                                     text_processor_func=None) -> None:
        """
        Export receipts from a directory of extracted text files.
        
        Args:
            texts_directory: Directory containing extracted text files
            text_processor_func: Function to process text (from text_processor module)
        """
        if text_processor_func is None:
            from text_processor import process_texts
            text_processor_func = process_texts
        
        texts_dir = Path(texts_directory)
        extracted_texts = {}
        
        for text_file in texts_dir.glob("*.txt"):
            with open(text_file, "r", encoding="utf-8") as f:
                extracted_texts[text_file.stem] = f.read()
        
        if not extracted_texts:
            print(f"No text files found in {texts_directory}")
            return
        
        processed_data = text_processor_func(extracted_texts)
        
        self.export_to_csv(processed_data, include_image_filename=True)


def export_to_csv(data: List[Dict], output_path: str = "receipts.csv", 
                  include_image_filename: bool = False) -> None:
    """
    Convenience function to export data to CSV.
    
    Args:
        data: List of dictionaries with date, amount, and category
        output_path: Path to the output CSV file
        include_image_filename: Whether to include image filename column
    """
    exporter = CSVExporter(output_path)
    exporter.export_to_csv(data, include_image_filename)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python csv_exporter.py <extracted_texts_directory> [output_csv]")
        sys.exit(1)
    
    texts_dir = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "receipts.csv"
    
    exporter = CSVExporter(output_csv)
    exporter.export_from_texts_directory(texts_dir)

