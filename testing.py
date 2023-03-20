import PyPDF2

# Open the PDF file in binary mode
pdf_file = open('input.pdf', 'rb')

# Create a PDF reader object
pdf_reader = PyPDF2.PdfFileReader(pdf_file)

# Get the total number of pages in the PDF document
num_pages = pdf_reader.numPages

# Iterate through each page and extract the text
for i in range(num_pages):
    page = pdf_reader.getPage(i)
    page_text = page.extractText()
    print(page_text)
    
# Close the PDF file
pdf_file.close()