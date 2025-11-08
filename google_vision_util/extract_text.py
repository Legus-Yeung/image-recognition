"""
Script to extract text from Google Vision API OCR output
"""

import json

def extract_text_from_vision_json(json_file_path, output_file_path=None):
    """
    Extract text from Google Vision API JSON response

    Args:
        json_file_path (str): Path to the Vision API JSON file
        output_file_path (str): Optional path to save extracted text

    Returns:
        str: Extracted text
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        responses = data.get('responses', [])
        if not responses:
            print("No responses found in JSON")
            return ""

        response = responses[0]
        full_text_annotation = response.get('fullTextAnnotation', {})

        text = full_text_annotation.get('text', '')

        if not text:
            print("No text found in fullTextAnnotation")
            return ""

        print("Text extracted successfully!")
        print("=" * 50)
        print(text)
        print("=" * 50)

        if output_file_path:
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(text)
            print(f"\nText saved to: {output_file_path}")

        return text

    except FileNotFoundError:
        print(f"Error: File '{json_file_path}' not found")
        return ""
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{json_file_path}'")
        return ""
    except Exception as e:
        print(f"Error: {str(e)}")
        return ""

def extract_individual_text_blocks(json_file_path, output_file_path=None):
    """
    Extract individual text blocks from the JSON (alternative method)

    Args:
        json_file_path (str): Path to the Vision API JSON file
        output_file_path (str): Optional path to save extracted text
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        responses = data.get('responses', [])
        if not responses:
            print("No responses found in JSON")
            return ""

        response = responses[0]
        text_annotations = response.get('textAnnotations', [])

        extracted_text = []
        for annotation in text_annotations:
            description = annotation.get('description', '')
            if description:
                extracted_text.append(description)

        full_text = '\n'.join(extracted_text)

        print("Individual text blocks extracted:")
        print("=" * 50)
        print(full_text)
        print("=" * 50)

        if output_file_path:
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(full_text)
            print(f"\nText saved to: {output_file_path}")

        return full_text

    except Exception as e:
        print(f"Error: {str(e)}")
        return ""

if __name__ == "__main__":
    json_file = "vision_output.json"

    print("Using full text annotation method (recommended)...")
    text = extract_text_from_vision_json(json_file, "extracted_receipt.txt")

    # print("\n" + "="*50)
    # print("Alternative method (individual text blocks):")
    # extract_individual_text_blocks(json_file, "extracted_receipt_blocks.txt")
