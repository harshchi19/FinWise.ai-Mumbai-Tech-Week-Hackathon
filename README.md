![FAISS](https://github.com/user-attachments/assets/602ffe9b-6ca5-4468-b82a-33062090a934)# 📌 RBI Circulars Chatbot - AI-Powered Response Retrieval System

## 🚀 Overview
An *AI-powered chatbot* that processes *Reserve Bank of India (RBI)* financial circulars, enabling users to retrieve *context-aware responses* from regulatory documents. Built with *Google Gemini AI*, it supports *text, images, videos, audio*, and includes *speech-to-text* and *text-to-speech* capabilities.

The system uses a *custom knowledge base* of *28 PDF circulars* from:
- *Financial Inclusion and Development Department*
- *Financial Markets Regulation Department*
- *Department of Regulation (Diagram Reports)*

---

## 🏗 System Architecture

### 🔹 *Core AI System (Gemini 1.5 Flash and Pro)*
- Powered by *Gemini 1.5 Flash & Pro* for query resolution.
- Extracts and processes *text, hyperlinks, and multimedia* from PDFs.
- Supports *multi-turn conversations* with context retention.

### 🔹 *Speech Processing*
- *Speech-to-Text*: Converts spoken queries to text using Google Speech API.
- *Text-to-Speech*: Reads responses aloud with Google TTS.

### 🔹 *Multimedia Processing*
- Extracts data from *PDFs, images, videos, and audio* via Google Generative AI.
- Handles uploads and integrates content into responses.

![image](https://github.com/user-attachments/assets/a594906e-46ac-4f8c-b6bd-843024de3bc9)

![image](https://github.com/user-attachments/assets/02912da1-512e-42cd-9fb4-76ab5ecf56af)

### ⚙️ Technical Architecture

![image](https://github.com/user-attachments/assets/0fdc4b77-357b-4ddc-8c3c-584a0166419e))

---

## 🎯 Features
✔ *Chat-Based Search* – Query RBI circulars in real-time.  
✔ *Multimedia Support* – Process *PDFs, images, videos, audio*.  
✔ *Speech Interaction* – Speak queries and hear responses.  
✔ *Hyperlink Extraction* – Provides clickable RBI circular references.  
✔ *Custom Knowledge Base* – Built on *28 RBI circulars*.  
✔ *Progress Tracking* – Displays circular processing status.

---

## ⚙ Tech Stack
- **Language**: Python  
- **Frontend**: Streamlit  
- **Backend**: FastAPI (planned)  
- **AI**: Google Gemini 1.5 Flash & Pro  
- **Speech**: SpeechRecognition, gTTS  
- **PDF**: PyMuPDF (fitz), PyPDF2  
- **File**: os, glob  
- **Image**: PIL (Pillow)  
- **Data**: io (BytesIO)  
- **Text**: re  
- **APIs**: Google (AI, Speech, TTS)  
- **Hardware**: Microphone  

---

## 🚀 Installation & Setup
### 1️⃣ *Clone the Repository*
```bash
git clone https://github.com/harshchi19/Fintech-Hackathon.git
```

### 2️⃣ *Install Dependencies*
```bash
pip install streamlit google-generativeai pypdf2 pymupdf pillow speechrecognition gtts pyaudio
```

### 3️⃣ *Set API Key*
Replace `GOOGLE_API_KEY` in the code with your Google API key.

### 4️⃣ *Run the App*
```bash
streamlit run responseretrieval.py
```


