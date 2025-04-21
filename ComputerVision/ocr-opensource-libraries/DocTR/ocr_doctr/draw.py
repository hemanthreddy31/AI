import cv2
import numpy as np
from doctr.models import ocr_predictor
from doctr.io import DocumentFile


def detect_and_mark_paragraphs(image_path, output_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")

    # Initialize OCR predictor
    predictor = ocr_predictor(pretrained=True)

    # Perform OCR
    doc = DocumentFile.from_images(image_path)
    result = predictor(doc)

    # Create output image
    output_image = image.copy()

    # Define colors and settings
    colors = {
        'paragraph': (0, 0, 255),  # Red
        'table': (255, 0, 0),  # Blue
        'heading': (0, 255, 0),  # Green
        'other': (255, 255, 0)  # Cyan
    }
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5

    # Process each page
    for page in result.pages:
        # Export page to dict format
        page_dict = page.export()

        # First pass: identify all blocks
        blocks = []
        for block_idx, block in enumerate(page_dict['blocks']):
            # Get coordinates
            x_min = int(block['geometry'][0][0] * image.shape[1])
            y_min = int(block['geometry'][0][1] * image.shape[0])
            x_max = int(block['geometry'][1][0] * image.shape[1])
            y_max = int(block['geometry'][1][1] * image.shape[0])

            # Calculate block properties
            width = x_max - x_min
            height = y_max - y_min
            aspect_ratio = width / max(height, 1)

            # Extract text from lines
            block_text = ""
            line_count = 0
            total_chars = 0

            for line in block['lines']:
                for word in line['words']:
                    if 'value' in word:
                        block_text += word['value'] + " "
                        total_chars += len(word['value'])
                line_count += 1

            block_text = block_text.strip()
            avg_line_length = total_chars / max(line_count, 1)

            # Classify block type
            if line_count > 3 and avg_line_length > 30:
                block_type = 'paragraph'
            elif aspect_ratio > 2.5 and line_count > 1:
                block_type = 'table'
            elif line_count == 1 and len(block_text) < 50 and block_text.isupper():
                block_type = 'heading'
            else:
                block_type = 'other'

            blocks.append({
                'coords': (x_min, y_min, x_max, y_max),
                'type': block_type,
                'text': block_text[:50] + '...' if len(block_text) > 50 else block_text
            })

        # Second pass: draw boundaries and labels
        for i, block in enumerate(blocks):
            x_min, y_min, x_max, y_max = block['coords']
            color = colors[block['type']]

            # Draw rectangle
            cv2.rectangle(output_image, (x_min, y_min), (x_max, y_max), color, thickness)

            # Draw label background
            label = f"{block['type']} {i + 1}"
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, 1)
            cv2.rectangle(output_image,
                          (x_min, y_min - text_height - 5),
                          (x_min + text_width, y_min - 5),
                          color, -1)

            # Draw label text
            cv2.putText(output_image, label,
                        (x_min, y_min - 10),
                        font, font_scale, (255, 255, 255), 1)

    # Save and display results
    cv2.imwrite(output_path, output_image)
    print(f"Processed image saved to {output_path}")

    # Display with resizing for large images
    display_image = cv2.resize(output_image, (800, int(800 * output_image.shape[0] / output_image.shape[1])))
    cv2.imshow("Paragraph Detection", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Usage
input_path = r"C:\Dotnet\CompleteWorkDocs\20Experiment\Images\CG210-merged_page-0006.jpg"
output_path = 'output_with_paragraphs.jpg'
detect_and_mark_paragraphs(input_path, output_path)