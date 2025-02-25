import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader 
import os, fitz, PIL.Image, time
import re
import glob
import numpy as np
import faiss
import pickle
import hashlib
from datetime import datetime

# Paths to your directories
path2 = '/Users/....'  # Replace with your directory path
financial_inclusion_folder = 'Data for Financial Inclusion and Development Department'
financial_markets_folder = 'Data For Financial Markets Regulation Department'
FAISS_INDEX_DIR = 'faiss_indexes'  # Directory to store FAISS indexes

GOOGLE_API_KEY = "AIzaSyBaVmGidt0Sb8oIr0TZJI6ly26zhK6wxNI"  # Replace with your API key
genai.configure(api_key=GOOGLE_API_KEY)

# Create directory for FAISS indexes if it doesn't exist
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

# Vector database class for document storage and retrieval
class VectorDatabase:
    def __init__(self, model, dimension=768):
        self.model = model
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance for similarity search
        self.document_chunks = []
        self.document_metadata = []
        
    def _get_embedding(self, text):
        try:
            embedding_result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            return np.array(embedding_result["embedding"]).astype('float32')
        except Exception as e:
            st.error(f"Error generating embedding: {e}")
            try:
                from sentence_transformers import SentenceTransformer
                st.warning("Falling back to sentence-transformers for embeddings")
                if not hasattr(self, 'st_model'):
                    self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
                embedding = self.st_model.encode(text, normalize_embeddings=True)
                return embedding.astype('float32')
            except:
                return np.zeros(self.dimension).astype('float32')
    
    def add_document(self, document_text, chunk_size=1000, overlap=200, metadata=None):
        chunks = []
        for i in range(0, len(document_text), chunk_size - overlap):
            chunk = document_text[i:i + chunk_size]
            if len(chunk) < 100:
                continue
            chunks.append(chunk)
        
        embeddings = []
        for i, chunk in enumerate(chunks):
            with st.spinner(f"Processing chunk {i+1}/{len(chunks)}..."):
                embedding = self._get_embedding(chunk)
                embeddings.append(embedding)
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata["chunk_index"] = i
                chunk_metadata["text"] = chunk
                self.document_chunks.append(chunk)
                self.document_metadata.append(chunk_metadata)
        
        if embeddings:
            embeddings_array = np.array(embeddings).astype('float32')
            self.index.add(embeddings_array)
        
        return len(chunks)
    
    def search(self, query, top_k=5):
        query_embedding = self._get_embedding(query)
        query_embedding = np.array([query_embedding]).astype('float32')
        
        D, I = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(I[0]):
            if idx < len(self.document_chunks) and idx >= 0:
                results.append({
                    "text": self.document_chunks[idx],
                    "metadata": self.document_metadata[idx],
                    "score": float(D[0][i])
                })
        return results
    
    def save(self, filename):
        index_file = os.path.join(FAISS_INDEX_DIR, f"{filename}.index")
        faiss.write_index(self.index, index_file)
        data_file = os.path.join(FAISS_INDEX_DIR, f"{filename}.pkl")
        with open(data_file, 'wb') as f:
            pickle.dump({
                'chunks': self.document_chunks,
                'metadata': self.document_metadata,
                'dimension': self.dimension
            }, f)
    
    @classmethod
    def load(cls, filename, model):
        index_file = os.path.join(FAISS_INDEX_DIR, f"{filename}.index")
        data_file = os.path.join(FAISS_INDEX_DIR, f"{filename}.pkl")
        
        if not os.path.exists(index_file) or not os.path.exists(data_file):
            return None
        
        index = faiss.read_index(index_file)
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls(model, data['dimension'])
        instance.index = index
        instance.document_chunks = data['chunks']
        instance.document_metadata = data['metadata']
        return instance

def page_setup():
    st.set_page_config(page_title="RAG Document Chat", layout="wide")
    st.header("Chat with Your Documents using RAG", anchor=False, divider="blue")
    hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

def get_model_options():
    st.sidebar.header("Model Options", divider='rainbow')
    model = st.sidebar.radio(
        "Choose LLM:",
        ("gemini-1.5-flash", "gemini-1.5-pro"),
        help="Select a model for generating responses"
    )
    temp = st.sidebar.slider(
        "Temperature:", 
        min_value=0.0,
        max_value=1.0, 
        value=0.2, 
        step=0.1
    )
    top_p = st.sidebar.slider(
        "Top P:", 
        min_value=0.0,
        max_value=1.0, 
        value=0.95, 
        step=0.01
    )
    max_tokens = st.sidebar.slider(
        "Maximum Output Tokens:", 
        min_value=100,
        max_value=8000, 
        value=2000, 
        step=100
    )
    context_window = st.sidebar.slider(
        "Chat History Context Window:",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Number of previous messages to include as context"
    )
    return model, temp, top_p, max_tokens, context_window

def extract_text_with_links(pdf_file):
    if isinstance(pdf_file, str):
        doc = fitz.open(pdf_file)
        file_content = None
    else:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        file_content = pdf_file
        
    text_with_links = ""
    all_links = {}
    link_texts = {}
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text_with_links += f"\n--- Page {page_num + 1} ---\n"
        links = page.get_links()
        page_links = {}
        
        for link in links:
            if 'uri' in link and 'from' in link:
                rect = link["from"]
                page_links[rect] = link['uri']
        
        all_links[page_num] = page_links
        link_texts[page_num] = {}
        
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
                                found_link = False
                                for link_rect, uri in page_links.items():
                                    link_rect = fitz.Rect(link_rect)
                                    if rect.intersects(link_rect):
                                        line_text += text + " "
                                        if uri not in link_texts[page_num]:
                                            link_texts[page_num][uri] = []
                                        link_texts[page_num][uri].append(text)
                                        found_link = True
                                        break
                                if not found_link:
                                    line_text += text + " "
                        text_with_links += line_text + "\n"
    
    if any(links for links in all_links.values()):
        text_with_links += "\n\n=== DOCUMENT REFERENCES ===\n"
        for page_num, page_links in all_links.items():
            for link_rect, uri in page_links.items():
                if uri in link_texts[page_num]:
                    linked_text = " ".join(link_texts[page_num][uri])
                    if len(linked_text) > 100:
                        linked_text = linked_text[:97] + "..."
                    text_with_links += f"• Reference on page {page_num + 1}: \"{linked_text}\" - Link: {uri}\n"
    
    if file_content is not None:
        file_content.seek(0)
    return text_with_links

def get_pdf_files_from_folders():
    pdf_files = []
    for folder_name in [financial_inclusion_folder, financial_markets_folder]:
        folder_path = os.path.abspath(folder_name)
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
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

def generate_database_id(pdf_files):
    hasher = hashlib.md5()
    for pdf in pdf_files:
        name = pdf['name'] if isinstance(pdf, dict) else pdf.name
        hasher.update(name.encode())
        if isinstance(pdf, dict) and 'path' in pdf:
            mtime = os.path.getmtime(pdf['path'])
            hasher.update(str(mtime).encode())
    today = datetime.now().strftime("%Y%m%d")
    hasher.update(today.encode())
    return hasher.hexdigest()

def init_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'db_id' not in st.session_state:
        st.session_state.db_id = None
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []

def add_to_chat(role, content, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.chat_history.append({
        "role": role, 
        "content": content,
        "timestamp": timestamp
    })

def display_chat():
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(f"**[{message['timestamp']}]** {message['content']}")

def clear_chat():
    st.session_state.chat_history = []

def generate_rag_prompt(query, retrieved_contexts, chat_history, context_window):
    # Get relevant chat history (last 'context_window' exchanges)
    history_context = ""
    if chat_history:
        # Take the last context_window * 2 messages (user + assistant pairs)
        relevant_history = chat_history[-context_window*2:]
        history_context = "Previous conversation context:\n"
        for msg in relevant_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            history_context += f"{role} [{msg['timestamp']}]: {msg['content']}\n"
        history_context += "\n"

    prompt = f"""Answer the following question based on the provided document context and previous conversation history. 
If the information needed is not in the context or history, say "I don't have enough information to answer this question accurately."

{history_context}

Document context:
{retrieved_contexts}

Current Question: {query}

Answer:"""
    return prompt

def handle_rag_chat(model_instance, vector_db, context_window):
    user_input = st.chat_input("Ask a question about your documents...")
    
    if user_input:
        add_to_chat("user", user_input)
        st.chat_message("user").write(user_input)
        message_placeholder = st.chat_message("assistant")
        
        with st.spinner("Searching documents and history for relevant information..."):
            # Search with combined query including recent chat context
            recent_context = " ".join([msg["content"] for msg in st.session_state.chat_history[-context_window*2:]])
            augmented_query = f"{recent_context} {user_input}"
            search_results = vector_db.search(augmented_query, top_k=5)
            
            if not search_results:
                response_text = "I couldn't find any relevant information in the documents or chat history to answer your question."
                add_to_chat("assistant", response_text)
                message_placeholder.markdown(response_text)
                return
            
            retrieved_contexts = ""
            for i, result in enumerate(search_results):
                text = result["text"]
                metadata = result["metadata"]
                source = metadata.get("source", "Unknown document")
                retrieved_contexts += f"[Document {i+1}: {source}]\n{text}\n\n"
            
            with st.expander("View retrieved document sections"):
                st.markdown(retrieved_contexts)
        
        with st.spinner("Generating contextual response..."):
            rag_prompt = generate_rag_prompt(
                user_input, 
                retrieved_contexts,
                st.session_state.chat_history,
                context_window
            )
            
            response = model_instance.generate_content(rag_prompt)
            response_text = response.text
            
            url_pattern = r'(https?://[^\s\)]+)'
            response_text = re.sub(url_pattern, r'[\1](\1)', response_text)
            
            add_to_chat("assistant", response_text)
            message_placeholder.markdown(response_text)

def main():
    page_setup()
    init_session_state()
    
    st.sidebar.title("Document RAG Chat")
    model_name, temperature, top_p, max_tokens, context_window = get_model_options()
    
    embedding_selection = st.sidebar.selectbox(
        "Choose Embedding Model:",
        ["Google Embeddings", "Sentence Transformers (Fallback)"]
    )
    
    if embedding_selection == "Sentence Transformers (Fallback)":
        with st.sidebar.expander("Installation Instructions"):
            st.code("pip install sentence-transformers", language="bash")
    
    tab1, tab2, tab3 = st.tabs(["Document Processor", "Chat Interface", "Settings"])
    
    if st.sidebar.button("Clear Chat History"):
        clear_chat()
        st.sidebar.success("Chat history cleared!")
    
    generation_config = {
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_tokens,
    }
    
    model_instance = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
    )
    
    embed_model = "embedding-001"
    
    with tab1:
        st.header("Document Processor")
        default_pdfs = get_pdf_files_from_folders()
        
        if default_pdfs:
            st.info(f"Found {len(default_pdfs)} PDFs in default folders.")
            departments = {}
            for pdf in default_pdfs:
                if pdf["department"] not in departments:
                    departments[pdf["department"]] = []
                departments[pdf["department"]].append(pdf["name"])
            with st.expander("View default PDFs by department"):
                for dept, files in departments.items():
                    st.subheader(dept)
                    for file in files:
                        st.write(f"- {file}")
        else:
            st.warning("No PDF files found in the default folders.")
            
        uploaded_files = st.file_uploader("Upload additional PDFs", type='pdf', accept_multiple_files=True)
        
        st.subheader("Vector Database Options")
        chunk_size = st.slider("Chunk Size", min_value=500, max_value=3000, value=1000, step=100)
        chunk_overlap = st.slider("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)
        force_reindex = st.checkbox("Force reindexing", value=False)
        skip_empty_pdfs = st.checkbox("Skip PDFs with no text content", value=True)
        
        process_button = st.button("Process Documents", type="primary")
        
        if process_button and (default_pdfs or uploaded_files):
            combined_pdfs = default_pdfs + (list(uploaded_files) if uploaded_files else [])
            db_id = generate_database_id(combined_pdfs)
            
            if force_reindex or db_id != st.session_state.db_id:
                progress_bar = st.progress(0)
                status_text = st.empty()
                vector_db = VectorDatabase(embed_model)
                
                total_files = len(default_pdfs) + (len(uploaded_files) if uploaded_files else 0)
                processed = 0
                skipped = 0
                
                for pdf in default_pdfs:
                    status_text.text(f"Processing default PDF {processed+1}/{total_files}: {pdf['name']}")
                    pdf_text = extract_text_with_links(pdf["path"])
                    if pdf_text.strip():
                        metadata = {"source": f"{pdf['name']} (Department: {pdf['department']})", "type": "pdf"}
                        chunks_added = vector_db.add_document(pdf_text, chunk_size=chunk_size, overlap=chunk_overlap, metadata=metadata)
                        st.write(f"Added {chunks_added} chunks from {pdf['name']}")
                    else:
                        if skip_empty_pdfs:
                            st.warning(f"Skipping {pdf['name']} - no text extracted")
                            skipped += 1
                        else:
                            metadata = {"source": f"{pdf['name']} (Department: {pdf['department']})", "type": "pdf"}
                            vector_db.add_document("No extractable text.", chunk_size=chunk_size, overlap=0, metadata=metadata)
                    processed += 1
                    progress_bar.progress(processed / total_files)
                
                if uploaded_files:
                    for pdf in uploaded_files:
                        status_text.text(f"Processing uploaded PDF {processed+1}/{total_files}: {pdf.name}")
                        pdf_text = extract_text_with_links(pdf)
                        if pdf_text.strip():
                            metadata = {"source": f"{pdf.name} (Uploaded)", "type": "pdf"}
                            chunks_added = vector_db.add_document(pdf_text, chunk_size=chunk_size, overlap=chunk_overlap, metadata=metadata)
                            st.write(f"Added {chunks_added} chunks from {pdf.name}")
                        else:
                            if skip_empty_pdfs:
                                st.warning(f"Skipping {pdf.name} - no text extracted")
                                skipped += 1
                            else:
                                metadata = {"source": f"{pdf.name} (Uploaded)", "type": "pdf"}
                                vector_db.add_document("No extractable text.", chunk_size=chunk_size, overlap=0, metadata=metadata)
                        processed += 1
                        progress_bar.progress(processed / total_files)
                
                vector_db.save(db_id)
                st.session_state.vector_db = vector_db
                st.session_state.db_id = db_id
                st.session_state.processed_files = [pdf["name"] for pdf in default_pdfs] + ([pdf.name for pdf in uploaded_files] if uploaded_files else [])
                
                status_text.text("Processing complete!")
                progress_bar.empty()
                st.success(f"✅ Processed {total_files - skipped} documents " + 
                          (f"(skipped {skipped} empty PDFs) " if skipped > 0 else "") +
                          f"with ID: {db_id}")
            else:
                vector_db = VectorDatabase.load(db_id, embed_model)
                if vector_db:
                    st.session_state.vector_db = vector_db
                    st.session_state.db_id = db_id
                    st.success(f"✅ Loaded existing database with ID: {db_id}")
    
    with tab2:
        st.header("Chat with your Documents")
        if st.session_state.vector_db is None:
            st.warning("⚠ Please process documents first.")
        else:
            st.success(f"✅ {len(st.session_state.processed_files)} files indexed and ready.")
            display_chat()
            handle_rag_chat(model_instance, st.session_state.vector_db, context_window)
    
    with tab3:
        st.header("Settings")
        if st.session_state.vector_db:
            st.subheader("Vector Database Information")
            st.write(f"Database ID: {st.session_state.db_id}")
            st.write(f"Number of chunks: {len(st.session_state.vector_db.document_chunks)}")
            if st.session_state.processed_files:
                st.subheader("Indexed Documents")
                for file in st.session_state.processed_files:
                    st.write(f"- {file}")
            if st.button("Clear Vector Database"):
                st.session_state.vector_db = None
                st.session_state.db_id = None
                st.session_state.processed_files = []
                st.success("Vector database cleared!")
                st.experimental_rerun()
        else:
            st.info("No vector database created yet.")
        
        st.subheader("Installation Requirements")
        with st.expander("Required Python Packages"):
            st.code("""
pip install streamlit google-generativeai PyMuPDF PyPDF2 faiss-cpu numpy pillow
# Optional:
pip install sentence-transformers
            """, language="bash")

if __name__ == '__main__':
    main()