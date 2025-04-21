import cv2
from doctr.models import ocr_predictor
from doctr.io import DocumentFile

# Load your image
image_path = r"C:\Dotnet\CompleteWorkDocs\20Experiment\Images\CG210-merged_page-0006.jpg"

# Initialize the OCR predictor
predictor = ocr_predictor(pretrained=True)

# Perform OCR
doc = DocumentFile.from_images(image_path)
result = predictor(doc)
print(result)
# Extract and print the recognized text
extracted_text = []
for page in result.pages:
    for block in page.blocks:
        for line in block.lines:
            line_text = " ".join(word.value for word in line.words)
            extracted_text.append(line_text)

# Join all lines into a single string
full_text = "\n".join(extracted_text)
print(full_text)
