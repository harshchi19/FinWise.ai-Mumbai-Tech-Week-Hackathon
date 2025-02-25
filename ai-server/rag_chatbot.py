# rag_chatbot.py
import google.generativeai as genai
from pypdf import PdfReader
import os, fitz
import re
import glob


# Paths to your directories
path2 = '/home/sameer42/Desktop/Hackathons/mumbai-tech-week/ai-server/' 
financial_inclusion_folder = 'Data for Financial Inclusion and Development Department'
financial_markets_folder = 'Data For Financial Markets Regulation Department'
financial_report = 'Diagram Report'

GOOGLE_API_KEY = "AIzaSyAPEQrNxCvH8vna78l4YO8Wc9sMISpDHH4" 
genai.configure(api_key=GOOGLE_API_KEY)


def delete_files_in_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except OSError:
        print("Error occurred while deleting files.")


def setup_documents(pdf_file_path):
    to_delete_path = path2
    delete_files_in_directory(to_delete_path)
    doc = fitz.open(pdf_file_path)
    os.chdir(to_delete_path)
    for page in doc:
        pix = page.get_pixmap(matrix=fitz.Identity, dpi=None,
                              colorspace=fitz.csRGB, clip=None, alpha=False, annots=True)
        pix.save("pdfimage-%i.jpg" % page.number)


def extract_text_with_links(pdf_file):
    """
    Enhanced function to extract text and hyperlinks from PDF files.
    This version clearly associates text with hyperlinks and uses definitive language.
    """
    if isinstance(pdf_file, str):  # It's a filepath
        doc = fitz.open(pdf_file)
        file_content = None
    else:  # It's an uploaded file
        doc = fitz.open(stream=pdf_file.file.read(), filetype="pdf")
        file_content = pdf_file

    text_with_links = ""

    # Extract and store all links by page
    all_links = {}
    link_texts = {}

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_with_links += f"\n--- Page {page_num + 1} ---\n"

        # Get all links from the page
        links = page.get_links()
        page_links = {}

        # Create a map of link areas to URLs
        for link in links:
            if 'uri' in link and 'from' in link:
                rect = link["from"]
                page_links[rect] = link['uri']

        # Store the links for this page
        all_links[page_num] = page_links
        link_texts[page_num] = {}

        # Get text blocks with their bounding boxes
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    if "spans" in line:
                        line_text = ""
                        for span in line["spans"]:
                            text = span["text"]
                            if text.strip():
                                rect = fitz.Rect(span["bbox"])

                                # Check if this text has an associated link
                                found_link = False
                                for link_rect, uri in page_links.items():
                                    link_rect = fitz.Rect(link_rect)
                                    # If the text rectangle is inside or overlaps with a link rectangle
                                    if rect.intersects(link_rect):
                                        line_text += text + " "
                                        # Store the text associated with this link
                                        if uri not in link_texts[page_num]:
                                            link_texts[page_num][uri] = []
                                        link_texts[page_num][uri].append(text)
                                        found_link = True
                                        break

                                if not found_link:
                                    line_text += text + " "

                        text_with_links += line_text + "\n"

    # Add a clear reference section at the end
    if any(links for links in all_links.values()):
        text_with_links += "\n\n=== DOCUMENT REFERENCES ===\n"
        for page_num, page_links in all_links.items():
            for link_rect, uri in page_links.items():
                if uri in link_texts[page_num]:
                    linked_text = " ".join(link_texts[page_num][uri])
                    # Clean up the linked text if it's too long
                    if len(linked_text) > 100:
                        linked_text = linked_text[:97] + "..."
                    text_with_links += f"• Reference on page {page_num + 1}: \"{linked_text}\" - Link: {uri}\n"

    # If we used a file object, we need to reset its position
    if file_content is not None:
        file_content.file.seek(0)

    return text_with_links


def get_pdf_files_from_folders():
    """Get all PDF files from the default folders"""
    pdf_files = []

    # Check if the folders exist
    for folder_name in [financial_inclusion_folder, financial_markets_folder, financial_report]:
        folder_path = os.path.abspath(folder_name)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            # Get all PDF files in the folder
            folder_pdfs = glob.glob(os.path.join(folder_path, "*.pdf"))
            for pdf_path in folder_pdfs:
                pdf_files.append({
                    "path": pdf_path,
                    "name": os.path.basename(pdf_path),
                    "department": folder_name
                })

    return pdf_files


def generate_system_prompt():
    """Generate a system prompt to guide the LLM's responses about hyperlinks."""
    return """
When responding to questions about documents that contain hyperlinks, please follow these guidelines:

1. When mentioning a document or reference that has an associated link, always provide the exact link.
2. Present links as definitive sources: "The reference can be found at [URL]" instead of suggesting uncertainty.
3. Present URLs as clickable links in your response.
4. Do not use phrases like "may or may not contain" or "you will have to navigate" when referring to links.
5. Use phrases like "All references can be found at the following links:" or "The corresponding document is available at:".
6. When presented with a DOCUMENT REFERENCES section, use that information to connect document mentions with their proper links.

Remember, all links referenced in the document are valid and direct connections to their referenced materials.
"""


def process_pdf_and_generate_response(pdf_files, user_input, model_name="gemini-1.5-pro", temperature=1.0, top_p=0.94, max_tokens=2000):
    """
    Processes PDF files, extracts text, and generates a response using the Gemini model.

    Args:
        pdf_files (List[UploadFile]): A list of uploaded PDF files.
        user_input (str): The user's query.
        model_name (str, optional): The name of the Gemini model to use. Defaults to "gemini-1.5-pro".
        temperature (float, optional): Controls the randomness of the response. Defaults to 1.0.
        top_p (float, optional): Controls the diversity of the response. Defaults to 0.94.
        max_tokens (int, optional): The maximum number of tokens in the response. Defaults to 2000.

    Returns:
        str: The generated response from the Gemini model.
    """

    text_content = ""
    for pdf_file in pdf_files:
        text_content += f"\n\n==== PDF: {pdf_file.filename} ====\n"
        text_content += extract_text_with_links(pdf_file)

    return get_response(user_input, text_content, model_name, temperature, top_p, max_tokens)

async def get_response(user_input, text_content="", model_name="gemini-1.5-pro", temperature=0.5, top_p=0.94, max_tokens=2000):
    """
    Generates a response using the Gemini model, optionally incorporating text content.

    Args:
        user_input (str): The user's query.
        text_content (str, optional):  Content extracted from PDFs. Defaults to "".
        model_name (str, optional): The name of the Gemini model to use. Defaults to "gemini-1.5-pro".
        temperature (float, optional): Controls the randomness of the response. Defaults to 0.5.
        top_p (float, optional): Controls the diversity of the response. Defaults to 0.94.
        max_tokens (int, optional): The maximum number of tokens in the response. Defaults to 2000.

    Returns:
        str: The generated response from the Gemini model.
    """
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_tokens,
        "response_mime_type": "text/plain",
    }

    model_instance = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
    )

    conversation = []
    system_prompt = generate_system_prompt()
    conversation.append(system_prompt)

    if text_content:
        conversation.append(text_content)

    conversation.append(user_input)

    response = model_instance.generate_content(conversation)
    print(response)
    response_text = response.text

    url_pattern = r'(https?://[^\s\)]+)'
    response_text = re.sub(url_pattern, r'[\1](\1)', response_text)

    return response_text

def load_default_pdfs():
    """Loads and extracts text from the default PDF folders."""
    default_pdfs = get_pdf_files_from_folders()
    text_content = ""
    for pdf in default_pdfs:
        text_content += f"\n\n==== PDF: {pdf['name']} (Department: {pdf['department']}) ====\n"
        text_content += extract_text_with_links(pdf["path"])
    return text_content