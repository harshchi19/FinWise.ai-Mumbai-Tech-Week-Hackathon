import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader 
import os, fitz, PIL.Image, time
import re
import glob

# Paths to your directories
path2 = '/Users/....'  # Replace with your directory path
financial_inclusion_folder = 'Data for Financial Inclusion and Development Department'
financial_markets_folder = 'Data For Financial Markets Regulation Department'
financial_report = 'Diagram Report'

GOOGLE_API_KEY = "AIzaSyBaVmGidt0Sb8oIr0TZJI6ly26zhK6wxNI"  # Replace with your API key
genai.configure(api_key=GOOGLE_API_KEY)

def page_setup():
    st.header("Chat with different types of media/files!", anchor=False, divider="blue")

    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

def get_typeofpdf():
    st.sidebar.header("Select type of Media", divider='orange')
    typepdf = st.sidebar.radio("Choose one:",
                               ("PDF files",
                                "Images",
                                "Video, mp4 file",
                                "Audio files"))
    return typepdf

def get_llminfo():
    st.sidebar.header("Options", divider='rainbow')
    tip1="Select a model you want to use."
    model = st.sidebar.radio("Choose LLM:",
                                  ("gemini-1.5-flash",
                                   "gemini-1.5-pro",
                                   ), help=tip1)
    tip2="Lower temperatures are good for prompts that require a less open-ended or creative response, while higher temperatures can lead to more diverse or creative results. A temperature of 0 means that the highest probability tokens are always selected."
    temp = st.sidebar.slider("Temperature:", min_value=0.0,
                                    max_value=2.0, value=1.0, step=0.25, help=tip2)
    tip3="Used for nucleus sampling. Specify a lower value for less random responses and a higher value for more random responses."
    topp = st.sidebar.slider("Top P:", min_value=0.0,
                             max_value=1.0, value=0.94, step=0.01, help=tip3)
    tip4="Number of response tokens, 8194 is limit."
    maxtokens = st.sidebar.slider("Maximum Tokens:", min_value=100,
                                  max_value=5000, value=2000, step=100, help=tip4)
    return model, temp, topp, maxtokens

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
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
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
        file_content.seek(0)
        
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
        else:
            st.warning(f"Folder not found: {folder_name}")
    
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

def init_chat_history():
    """Initialize chat history in session state if not already present"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'context' not in st.session_state:
        st.session_state.context = ""

def add_to_chat(role, content):
    """Add a new message to the chat history"""
    st.session_state.chat_history.append({"role": role, "content": content})

def display_chat():
    """Display the chat history"""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            st.chat_message("assistant").markdown(message["content"])

def clear_chat():
    """Clear the chat history"""
    st.session_state.chat_history = []
    st.session_state.context = ""

def handle_chat(model_instance, content_text=""):
    """Handle chat interactions"""
    # Initialize message container for new messages
    message_placeholder = st.empty()
    
    # Get user input
    user_input = st.chat_input("Ask a question...")
    
    if user_input:
        # Add user message to chat
        add_to_chat("user", user_input)
        
        # Show updated chat
        message_placeholder = st.chat_message("assistant")
        
        # Build the conversation history for context
        conversation = []
        system_prompt = generate_system_prompt()
        conversation.append(system_prompt)
        
        if content_text:
            conversation.append(content_text)
        
        # Add previous messages for context
        for message in st.session_state.chat_history[:-1]:  # Exclude the latest user message which we'll add separately
            conversation.append(message["content"])
        
        # Add the latest user message
        conversation.append(user_input)
        
        # Generate response
        with st.spinner("Thinking..."):
            response = model_instance.generate_content(conversation)
            
            # Process the response to ensure links are clickable in Streamlit
            response_text = response.text
            
            # Convert plain URLs to markdown links if they're not already
            url_pattern = r'(https?://[^\s\)]+)'
            response_text = re.sub(url_pattern, r'[\1](\1)', response_text)
            
            # Add assistant message to chat
            add_to_chat("assistant", response_text)
            
            # Display the message
            message_placeholder.markdown(response_text)

def main():
    page_setup()
    typepdf = get_typeofpdf()
    model, temperature, top_p, max_tokens = get_llminfo()
    
    # Initialize chat history
    init_chat_history()
    
    # Add a clear chat button
    if st.button("Clear Chat"):
        clear_chat()
    
    # Display existing chat history
    display_chat()
    
    # Set up the generation config for the model
    generation_config = {
      "temperature": temperature,
      "top_p": top_p,
      "max_output_tokens": max_tokens,
      "response_mime_type": "text/plain",
    }
    
    model_instance = genai.GenerativeModel(
      model_name=model,
      generation_config=generation_config,
    )
    
    if typepdf == "PDF files":
        # Get default PDFs from folders
        default_pdfs = get_pdf_files_from_folders()
        
        # Display info about default PDFs
        if default_pdfs:
            st.info(f"Loaded {len(default_pdfs)} PDFs from default folders.")
            
            # Group PDFs by department for display
            departments = {}
            for pdf in default_pdfs:
                if pdf["department"] not in departments:
                    departments[pdf["department"]] = []
                departments[pdf["department"]].append(pdf["name"])
            
            # Display the default PDFs by department
            with st.expander("View default PDFs by department"):
                for dept, files in departments.items():
                    st.subheader(dept)
                    for file in files:
                        st.write(f"- {file}")
        else:
            st.warning("No PDF files found in the default folders.")
            
        # Allow additional PDF uploads
        uploaded_files = st.file_uploader("Upload additional PDFs (optional)", type='pdf', accept_multiple_files=True)
        
        # Process all PDFs (default + uploaded)
        if default_pdfs or uploaded_files:
            # Check if context is already set
            if not st.session_state.context:
                text_with_links = ""
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process default PDFs
                total_files = len(default_pdfs) + (len(uploaded_files) if uploaded_files else 0)
                processed = 0
                
                for pdf in default_pdfs:
                    status_text.text(f"Processing default PDF {processed+1}/{total_files}: {pdf['name']}")
                    pdf_text = extract_text_with_links(pdf["path"])
                    text_with_links += f"\n\n==== PDF: {pdf['name']} (Department: {pdf['department']}) ====\n"
                    text_with_links += pdf_text
                    processed += 1
                    progress_bar.progress(processed / total_files)
                
                # Process uploaded PDFs
                if uploaded_files:
                    for pdf in uploaded_files:
                        status_text.text(f"Processing uploaded PDF {processed+1}/{total_files}: {pdf.name}")
                        pdf_text = extract_text_with_links(pdf)
                        text_with_links += f"\n\n==== PDF: {pdf.name} (Uploaded) ====\n"
                        text_with_links += pdf_text
                        processed += 1
                        progress_bar.progress(processed / total_files)
                
                status_text.text("Processing complete!")
                progress_bar.empty()
                
                # Show a sample of the processed text
                with st.expander("Preview of extracted content (click to expand)"):
                    st.text(text_with_links[:1000] + "..." if len(text_with_links) > 1000 else text_with_links)
                    
                # Show number of tokens
                token_count = model_instance.count_tokens(text_with_links)
                st.write(f"Total tokens: {token_count.total_tokens}")
                
                # Check if token count exceeds model limits
                if token_count.total_tokens > 30000:  # Adjust this threshold based on model limits
                    st.warning("Warning: The extracted content exceeds recommended token limits. Consider using fewer PDFs.")
                
                # Save the context
                st.session_state.context = text_with_links
            
            # Handle the chat with the current context
            handle_chat(model_instance, st.session_state.context)
            
    elif typepdf == "Images":
        image_file_name = st.file_uploader("Upload your image file.",)
        if image_file_name:
            path3 = '/Users/....'
            fpath = image_file_name.name
            fpath2 = (os.path.join(path3, fpath))
            
            with st.spinner("Processing image..."):
                image_file = genai.upload_file(path=fpath2)
                
                while image_file.state.name == "PROCESSING":
                    time.sleep(10)
                    image_file = genai.get_file(image_file.name)
                if image_file.state.name == "FAILED":
                    st.error("Image processing failed.")
                    raise ValueError(image_file.state.name)
            
            # Store the image file in session state to maintain context
            if 'current_media' not in st.session_state:
                st.session_state.current_media = image_file
            
            # Handle the chat
            handle_chat(model_instance)
            
            # Cleanup when changing media
            if st.button("Remove Current Image"):
                if 'current_media' in st.session_state:
                    genai.delete_file(st.session_state.current_media.name)
                    del st.session_state.current_media
                    st.experimental_rerun()
           
    elif typepdf == "Video, mp4 file":
        video_file_name = st.file_uploader("Upload your video")
        if video_file_name:
            path3 = '/Users/....'
            fpath = video_file_name.name
            fpath2 = (os.path.join(path3, fpath))
            
            with st.spinner("Processing video..."):
                video_file = genai.upload_file(path=fpath2)
                
                while video_file.state.name == "PROCESSING":
                    time.sleep(10)
                    video_file = genai.get_file(video_file.name)
                if video_file.state.name == "FAILED":
                    st.error("Video processing failed.")
                    raise ValueError(video_file.state.name)
            
            # Store the video file in session state to maintain context
            if 'current_media' not in st.session_state:
                st.session_state.current_media = video_file
            
            # Handle the chat
            handle_chat(model_instance)
            
            # Cleanup when changing media
            if st.button("Remove Current Video"):
                if 'current_media' in st.session_state:
                    genai.delete_file(st.session_state.current_media.name)
                    del st.session_state.current_media
                    st.experimental_rerun()
      
    elif typepdf == "Audio files":
        audio_file_name = st.file_uploader("Upload your audio")
        if audio_file_name:
            path3 = '/Users/....'
            fpath = audio_file_name.name
            fpath2 = (os.path.join(path3, fpath))
            
            with st.spinner("Processing audio..."):
                audio_file = genai.upload_file(path=fpath2)

                while audio_file.state.name == "PROCESSING":
                    time.sleep(10)
                    audio_file = genai.get_file(audio_file.name)
                if audio_file.state.name == "FAILED":
                    st.error("Audio processing failed.")
                    raise ValueError(audio_file.state.name)

            # Store the audio file in session state to maintain context
            if 'current_media' not in st.session_state:
                st.session_state.current_media = audio_file
            
            # Handle the chat
            handle_chat(model_instance)
            
            # Cleanup when changing media
            if st.button("Remove Current Audio"):
                if 'current_media' in st.session_state:
                    genai.delete_file(st.session_state.current_media.name)
                    del st.session_state.current_media
                    st.experimental_rerun()

if __name__ == '__main__':
    main()