from docx import Document
import fitz  

filepath = "input.pdf"

# Extract text from PDF
text = ''
with fitz.open(filepath) as doc:
    for page in doc:
        text += page.get_text()
    
# Extract text from Word file
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return '\n'.join(text)
    
file_path = 'summary.docx'
extracted_text = extract_text_from_docx(file_path)
text += extracted_text

print(text)
