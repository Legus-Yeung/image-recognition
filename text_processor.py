"""
Module for processing extracted text to extract date, amount, and determine category.
Uses the Qwen3-VL model to analyze receipt text and categorize purchases.
"""
import re
from pathlib import Path
from typing import Dict, Optional
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


class TextProcessor:
    """Processes extracted receipt text to extract structured information."""
    
    CATEGORIES = ["dining", "entertainment", "travel", "utility", "health"]
    
    def __init__(self, model_name: str = "Qwen/Qwen3-VL-2B-Instruct", dtype=torch.float16):
        """
        Initialize the text processor with the vision-language model.
        
        Args:
            model_name: HuggingFace model identifier
            dtype: Model data type (torch.float16 or torch.float32)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
    
    def _query_model(self, prompt: str, max_new_tokens: int = 100) -> str:
        """
        Helper method to query the model with a text prompt.
        
        Args:
            prompt: The prompt text to send to the model
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Model response as a string
        """
        messages = [
            {
                "role": "user",
                "content": [
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
        
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return output_text[0].strip()
    
    def extract_all_info(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract date, amount, and category from receipt text using a single model call.
        
        Args:
            text: Extracted receipt text
            
        Returns:
            Dictionary with keys: date, amount, category
        """
        from datetime import datetime
        
        month_map = {
            'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
            'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6,
            'jul': 7, 'july': 7, 'aug': 8, 'august': 8, 'sep': 9, 'september': 9,
            'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
        }
        
        date_pattern_text = r'(\d{1,2})[-/\s]+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-/\s]+(\d{4})'
        date_match = re.search(date_pattern_text, text, re.IGNORECASE)
        quick_date = None
        if date_match:
            try:
                day, month_str, year = date_match.groups()
                month = month_map.get(month_str.lower()[:3])
                if month:
                    quick_date = f"{int(year):04d}-{month:02d}-{int(day):02d}"
            except (ValueError, KeyError):
                pass
        
        prompt = f"""Extract the following information from this receipt text:

1. DATE: Extract the purchase date and convert it to YYYY-MM-DD format.
   Handle formats like: DD-MMM-YYYY (03-Dec-2024 → 2024-12-03), MM/DD/YYYY, DD/MM/YYYY, YYYY-MM-DD.

2. AMOUNT: Extract the actual amount paid/charged (not subtotal or pre-tax total).
   Prioritize payment lines like "Amount Paid", "Charged", "Round of Paying", "Payment", "Paid", etc.
   If payment amount is not found, use the final TOTAL amount after discounts and taxes.
   Include the currency symbol (e.g., $45.20, €30.50, £25.99, ¥1000).
   Format: CURRENCY_SYMBOL + number (e.g., $45.20 or €30.50).

3. CATEGORY: Determine the purchase category. The MERCHANT NAME is the most important factor.
   Categories:
   - dining: Restaurants, cafes, food delivery, bars, fast food, grocery stores (food items)
   - entertainment: Movies, concerts, sports events, games, streaming services, amusement parks, bouldering gyms, climbing gyms, rock climbing facilities, arcades, bowling alleys
   - travel: Hotels, motels, airlines, trains, buses, car rentals, travel agencies, parking, taxis/rideshares
   - utility: Electricity, water, gas, internet, phone, cable, rent, property taxes
   - health: Pharmacies, hospitals, clinics, doctors, dentists, medical supplies, gyms (fitness/health), fitness centers, vitamins, supplements
   
   Respond with only one word: dining, entertainment, travel, utility, or health.

Receipt text:
{text}

Respond in this exact format:
DATE: YYYY-MM-DD or N/A
AMOUNT: CURRENCY_SYMBOLnumber or N/A (e.g., $45.20, €30.50, £25.99)
CATEGORY: category_name"""

        response = self._query_model(prompt, max_new_tokens=150)
        response = response.strip()
        result = {
            "date": quick_date,  # Use quick extraction if available
            "amount": None,
            "category": "dining"  # Default
        }
        
        if not result["date"]:
            date_match = re.search(r'DATE:\s*(\d{4}-\d{2}-\d{2}|N/A)', response, re.IGNORECASE)
            if date_match:
                date_str = date_match.group(1)
                if date_str.upper() != "N/A":
                    result["date"] = date_str
            else:
                date_pattern = r'(\d{4}-\d{2}-\d{2})'
                match = re.search(date_pattern, response)
                if match:
                    result["date"] = match.group(1)
                else:
                    month_pattern = r'(\d{1,2})[-/\s]+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-/\s]+(\d{4})'
                    match = re.search(month_pattern, response, re.IGNORECASE)
                    if match:
                        try:
                            day, month_str, year = match.groups()
                            month = month_map.get(month_str.lower()[:3])
                            if month:
                                result["date"] = f"{int(year):04d}-{month:02d}-{int(day):02d}"
                        except (ValueError, KeyError):
                            pass
        
        amount_match = re.search(r'AMOUNT:\s*([$€£¥₹]\s*[\d,]+(?:\.\d{1,2})?|[\d,]+(?:\.\d{1,2})?\s*[$€£¥₹]|N/A)', response, re.IGNORECASE)
        if amount_match:
            amount_str = amount_match.group(1)
            if amount_str.upper() != "N/A":
                amount_str = re.sub(r'\s+', '', amount_str)
                currency_match = re.search(r'([$€£¥₹])', amount_str)
                if currency_match:
                    currency = currency_match.group(1)
                    numeric_part = re.sub(r'[$€£¥₹,]', '', amount_str)
                    try:
                        amount = float(numeric_part)
                        if 0 < amount < 1000000:
                            result["amount"] = f"{currency}{amount:.2f}"
                    except ValueError:
                        pass
        
        if not result["amount"]:
            currency_patterns = [
                r'([$€£¥₹])\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*([$€£¥₹])',
            ]
            for pattern in currency_patterns:
                match = re.search(pattern, response)
                if match:
                    try:
                        if match.group(1) in ['$', '€', '£', '¥', '₹']:
                            currency = match.group(1)
                            numeric_part = match.group(2).replace(',', '')
                        else:
                            numeric_part = match.group(1).replace(',', '')
                            currency = match.group(2)
                        amount = float(numeric_part)
                        if 0 < amount < 1000000:
                            result["amount"] = f"{currency}{amount:.2f}"
                            break
                    except (ValueError, IndexError):
                        continue
        
        if not result["amount"]:
            text_lower = text.lower()
            payment_keywords = [
                r'round\s+of\s+paying(?:\s+\w+)?',
                r'amount\s+paid',
                r'charged',
                r'payment(?:\s+\w+)?',
                r'paid\s+(?:cash|card|credit)',
                r'cash\s+paid',
                r'total\s+paid',
            ]
            
            for keyword_pattern in payment_keywords:
                matches = list(re.finditer(keyword_pattern, text_lower, re.IGNORECASE))
                for match in matches:
                    remaining_text = text[match.end():match.end()+200]
                    currency_match = re.search(r'([$€£¥₹])\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', remaining_text)
                    if currency_match:
                        currency = currency_match.group(1)
                        numeric_part = currency_match.group(2).replace(',', '')
                        try:
                            amount = float(numeric_part)
                            if 0 < amount < 1000000:
                                result["amount"] = f"{currency}{amount:.2f}"
                                break
                        except ValueError:
                            continue
                if result["amount"]:
                    break
            
            if not result["amount"]:
                total_match = re.search(r'TOTAL[:\s]+([$€£¥₹])\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', text, re.IGNORECASE)
                if total_match:
                    currency = total_match.group(1)
                    numeric_part = total_match.group(2).replace(',', '')
                    try:
                        amount = float(numeric_part)
                        if 0 < amount < 1000000:
                            result["amount"] = f"{currency}{amount:.2f}"
                    except ValueError:
                        pass
        
        category_match = re.search(r'CATEGORY:\s*(\w+)', response, re.IGNORECASE)
        if category_match:
            category = category_match.group(1).lower()
            if category in self.CATEGORIES:
                result["category"] = category
        else:
            for category in self.CATEGORIES:
                if category in response.lower():
                    result["category"] = category
                    break
        
        text_lower = text.lower()
        
        entertainment_keywords = ['boulder', 'bouldering', 'climbing', 'rock climbing', 'arcade', 
                                'bowling', 'cinema', 'theater', 'movie', 'concert', 'amusement']
        if any(keyword in text_lower for keyword in entertainment_keywords):
            result["category"] = "entertainment"
        
        if result["category"] != "entertainment":
            travel_keywords = ['hotel', 'motel', 'inn', 'lodge', 'resort', 'airline', 'airport', 
                              'car rental', 'taxi', 'uber', 'lyft', 'parking', 'train', 'bus']
            if any(keyword in text_lower for keyword in travel_keywords):
                if 'hotel' in text_lower or 'motel' in text_lower or 'resort' in text_lower:
                    result["category"] = "travel"
                elif any(keyword in text_lower for keyword in ['airline', 'airport', 'car rental', 'taxi', 'uber', 'lyft']):
                    result["category"] = "travel"
        
        return result
    
    def process_text(self, text: str, image_filename: str = "") -> Dict[str, Optional[str]]:
        """
        Process extracted text to extract date, amount, and category.
        Uses a single model call for efficiency.
        
        Args:
            text: Extracted receipt text
            image_filename: Original image filename (for reference)
            
        Returns:
            Dictionary with keys: date, amount, category
        """
        result = self.extract_all_info(text)
        result["image_filename"] = image_filename
        return result
    
    def process_texts_batch(self, extracted_texts: Dict[str, str]) -> list:
        """
        Process multiple extracted texts in batch.
        
        Args:
            extracted_texts: Dictionary mapping image filenames to extracted text
            
        Returns:
            List of dictionaries with date, amount, and category for each receipt
        """
        results = []
        
        for image_filename, text in extracted_texts.items():
            if not text.strip():
                print(f"Skipping {image_filename}: empty text")
                continue
            
            print(f"Processing {image_filename}...")
            result = self.process_text(text, image_filename)
            results.append(result)
            print(f"  Date: {result['date']}, Amount: {result['amount']}, Category: {result['category']}")
        
        return results
    
    def cleanup(self):
        """
        Clean up model and free GPU memory.
        Call this when done with processing to free memory.
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


def process_texts(extracted_texts: Dict[str, str]) -> list:
    """
    Convenience function to process extracted texts.
    
    Args:
        extracted_texts: Dictionary mapping image filenames to extracted text
        
    Returns:
        List of dictionaries with date, amount, and category for each receipt
    """
    processor = TextProcessor()
    return processor.process_texts_batch(extracted_texts)


if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 2:
        print("Usage: python text_processor.py <extracted_text_file_or_directory>")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    processor = TextProcessor()
    
    if input_path.is_dir():
        extracted_texts = {}
        for text_file in sorted(input_path.glob("*.txt")):
            with open(text_file, "r", encoding="utf-8") as f:
                extracted_texts[text_file.stem] = f.read()
        
        if not extracted_texts:
            print(f"No text files found in {input_path}")
            sys.exit(1)
        
        print(f"Processing {len(extracted_texts)} text file(s) from {input_path}...\n")
        results = processor.process_texts_batch(extracted_texts)
        
        print("\nAll processed results:")
        print(json.dumps(results, indent=2))
    else:
        if not input_path.exists():
            print(f"Error: File not found: {input_path}")
            sys.exit(1)
        
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        result = processor.process_text(text, input_path.stem)
        
        print("\nProcessed result:")
        print(json.dumps(result, indent=2))
    
    processor.cleanup()

