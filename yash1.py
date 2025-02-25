import streamlit as st
import google.generativeai as genai
import os, fitz
import re
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import numpy as np
import pickle

# Paths to your directories - replace with your actual paths
financial_inclusion_folder = 'Data for Financial Inclusion and Development Department'
financial_markets_folder = 'Data For Financial Markets Regulation Department'

# Directory to store vector database
vector_db_dir = 'vector_db'
if not os.path.exists(vector_db_dir):
    os.makedirs(vector_db_dir)

# Replace with your actual API key
GOOGLE_API_KEY = "AIzaSyBaVmGidt0Sb8oIr0TZJI6ly26zhK6wxNI"
genai.configure(api_key=GOOGLE_API_KEY)

def page_setup():
    st.header("Chat with PDF Documents", anchor=False, divider="blue")
    
    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

def get_llminfo():
    st.sidebar.header("Model Options", divider='rainbow')
    tip1 = "Select a model you want to use."
    model = st.sidebar.radio("Choose LLM:",
                           ("gemini-1.5-flash",
                            "gemini-1.5-pro"),
                           help=tip1)
    
    tip2 = "Lower temperatures for factual responses, higher for creative ones."
    temp = st.sidebar.slider("Temperature:", min_value=0.0,
                           max_value=2.0, value=1.0, step=0.25, help=tip2)
    
    tip3 = "Lower for focused responses, higher for more variety."
    topp = st.sidebar.slider("Top P:", min_value=0.0,
                          max_value=1.0, value=0.94, step=0.01, help=tip3)
    
    tip4 = "Number of response tokens, 8194 is limit."
    maxtokens = st.sidebar.slider("Maximum Tokens:", min_value=100,
                               max_value=5000, value=2000, step=100, help=tip4)
    
    return model, temp, topp, maxtokens

def get_rag_settings():
    """Get RAG-specific settings"""
    st.sidebar.header("RAG Settings", divider='blue')
    
    # Chunk size for text splitting
    chunk_size = st.sidebar.slider("Chunk Size:", min_value=100,
                               max_value=2000, value=500, step=100, 
                               help="Size of text chunks for retrieval")
    
    # Chunk overlap
    chunk_overlap = st.sidebar.slider("Chunk Overlap:", min_value=0,
                                 max_value=500, value=50, step=10,
                                 help="Overlap between chunks to maintain context")
    
    # Number of top chunks to retrieve
    top_k_chunks = st.sidebar.slider("Top K Chunks:", min_value=1,
                                max_value=10, value=3, step=1,
                                help="Number of most relevant chunks to retrieve")
    
    # Use RAG or raw content
    use_rag = st.sidebar.checkbox("Use RAG", value=True, 
                             help="If checked, uses RAG for retrieving relevant information. Otherwise, uses the entire document content.")
    
    return chunk_size, chunk_overlap, top_k_chunks, use_rag

def extract_text_with_links(pdf_file):
    """
    Extract text and hyperlinks from PDF files.
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
    for folder_name in [financial_inclusion_folder, financial_markets_folder]:
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

def split_text_into_chunks(text, chunk_size, chunk_overlap):
    """Split text into overlapping chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    # Create langchain Document objects with metadata
    chunks = text_splitter.create_documents([text])
    
    # Extract document name and section from the text if available
    for i, chunk in enumerate(chunks):
        # Extract PDF name if available (look for "==== PDF: " pattern)
        pdf_match = re.search(r'==== PDF: (.*?) \(Department: (.*?)\) ====', chunk.page_content)
        if pdf_match:
            chunk.metadata['pdf_name'] = pdf_match.group(1)
            chunk.metadata['department'] = pdf_match.group(2)
        else:
            chunk.metadata['pdf_name'] = "Unknown"
            chunk.metadata['department'] = "Unknown"
            
        # Include index for reference
        chunk.metadata['chunk_id'] = i
    
    return chunks

def create_embeddings_and_faiss(chunks):
    """Create embeddings and FAISS vector store from text chunks"""
    # Create a custom embedding function using Google's embedding model
    def get_embeddings(texts):
        # Create embeddings for each text
        embeddings = []
        for text in texts:
            # Use Gemini to create embeddings
            embedding = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            # Extract the values and convert to numpy array
            embedding_values = np.array(embedding["embedding"])
            embeddings.append(embedding_values)
        return embeddings
    
    # Extract texts from chunks
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    
    # Get embeddings using Google's embedding API
    embeddings = get_embeddings(texts)
    
    # Initialize FAISS index for similarity search
    # Fixed: Corrected parameters for FAISS.from_embeddings
    index = FAISS(embeddings, texts, metadatas)
    
    return index

def retrieve_similar_chunks(index, query, top_k=3):
    """Retrieve most similar chunks to the query"""
    # Create query embedding
    query_embedding = genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query"
    )
    query_embedding_values = np.array(query_embedding["embedding"])
    
    # Search for similar chunks
    similar_docs = index.similarity_search_by_vector(
        query_embedding_values, 
        k=top_k
    )
    
    return similar_docs

def save_faiss_index(index, filename):
    """Save FAISS index to disk"""
    with open(os.path.join(vector_db_dir, f"{filename}.pkl"), "wb") as f:
        pickle.dump(index, f)

def load_faiss_index(filename):
    """Load FAISS index from disk"""
    try:
        with open(os.path.join(vector_db_dir, f"{filename}.pkl"), "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return None

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
    if 'vector_index' not in st.session_state:
        st.session_state.vector_index = None
    if 'text_chunks' not in st.session_state:
        st.session_state.text_chunks = []

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
    # Don't clear the vector index as it's expensive to recreate

def handle_chat(model_instance, use_rag=True, top_k=3):
    """Handle chat interactions with RAG"""
    # Initialize message container for new messages
    message_placeholder = st.empty()
    
    # Get user input
    user_input = st.chat_input("Ask a question about your PDFs...")
    
    if user_input:
        # Add user message to chat
        add_to_chat("user", user_input)
        
        # Show updated chat
        message_placeholder = st.chat_message("assistant")
        
        # Build the conversation history for context
        conversation = []
        system_prompt = generate_system_prompt()
        conversation.append(system_prompt)
        
        # Get relevant content based on the query
        context_content = ""
        
        if use_rag and st.session_state.vector_index is not None:
            with st.spinner("Retrieving relevant information..."):
                # Get similar chunks
                similar_chunks = retrieve_similar_chunks(
                    st.session_state.vector_index, 
                    user_input, 
                    top_k=top_k
                )
                
                # Format the chunks for context
                context_content = "Here are the most relevant sections from the documents:\n\n"
                for i, chunk in enumerate(similar_chunks):
                    content = chunk.page_content
                    metadata = chunk.metadata
                    
                    # Add metadata details
                    pdf_name = metadata.get('pdf_name', 'Unknown')
                    department = metadata.get('department', 'Unknown')
                    context_content += f"--- Section {i+1} from {pdf_name} (Department: {department}) ---\n"
                    context_content += content + "\n\n"
        else:
            # Use the full context if RAG is disabled
            context_content = st.session_state.context if st.session_state.context else ""
        
        # Add context to conversation
        if context_content:
            conversation.append(context_content)
        
        # Add previous messages for context (limit to last few to avoid token limit)
        max_history = 4  # Adjust based on needs
        for message in st.session_state.chat_history[-max_history:-1]:  # Exclude the latest user message
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
    model, temperature, top_p, max_tokens = get_llminfo()
    chunk_size, chunk_overlap, top_k_chunks, use_rag = get_rag_settings()
    
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
        # Check if context is already set and RAG is not yet created
        if not st.session_state.context or st.session_state.vector_index is None:
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
            
            # Generate a unique identifier for this set of PDFs
            pdf_names = [pdf["name"] for pdf in default_pdfs]
            if uploaded_files:
                pdf_names.extend([pdf.name for pdf in uploaded_files])
            pdf_id = "_".join([name.split(".")[0] for name in pdf_names[:3]])
            if len(pdf_names) > 3:
                pdf_id += f"_and_{len(pdf_names)-3}_more"
            
            # Check if a FAISS index already exists for these PDFs
            vector_index = load_faiss_index(pdf_id)
            
            # If no existing index, create one
            if vector_index is None:
                status_text.text("Creating vector database...")
                
                # Split text into chunks
                chunks = split_text_into_chunks(text_with_links, chunk_size, chunk_overlap)
                st.session_state.text_chunks = chunks
                
                # Create FAISS index
                vector_index = create_embeddings_and_faiss(chunks)
                
                # Save the index
                save_faiss_index(vector_index, pdf_id)
            
            status_text.text("Processing complete!")
            progress_bar.empty()
            
            # Show a sample of the processed text
            with st.expander("Preview of extracted content (click to expand)"):
                st.text(text_with_links[:1000] + "..." if len(text_with_links) > 1000 else text_with_links)
                
            # Save the context and vector index
            st.session_state.context = text_with_links
            st.session_state.vector_index = vector_index
            
            # Show number of chunks
            st.write(f"Total text chunks: {len(st.session_state.text_chunks)}")
            
            # Show number of tokens in full context
            token_count = model_instance.count_tokens(text_with_links)
            st.write(f"Total tokens in full context: {token_count.total_tokens}")
            st.write(f"Using {'RAG with top-' + str(top_k_chunks) + ' chunks' if use_rag else 'full context'} for answering questions")
        
        # Handle the chat with RAG or full context
        handle_chat(model_instance, use_rag=use_rag, top_k=top_k_chunks)

if __name__ == '__main__':
    main()