from fastapi import FastAPI, Form, HTTPException, Query, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
from dotenv import load_dotenv
from services.gemini_game_flow import get_gemini_response
from rag_chatbot import process_pdf_and_generate_response, get_response, load_default_pdfs

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

default_pdf_content = load_default_pdfs()

@app.post("/ai-financial-path")
async def ai_financial_path(
    input: str = Form(...),
    risk: Optional[str] = Form("conservative")
):
    try:
        response = get_gemini_response(input, risk)  
        return JSONResponse(content=response, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {str(e)}")
    
@app.post("/rag-chatbot")
async def rag_chatbot_endpoint(
    user_input: str = Form(...),
    pdf_files: List[UploadFile] = File([]),
    model_name: str = Form("gemini-1.5-flash"),
    temperature: float = Form(0.5),
    top_p: float = Form(0.94),
    max_tokens: int = Form(2000)
):
    """
    Endpoint for a RAG (Retrieval-Augmented Generation) chatbot that leverages Gemini.

    Args:
        user_input (str): The user's query to the chatbot.
        pdf_files (List[UploadFile], optional): A list of PDF files to use as context for the chatbot.
            Defaults to an empty list.
        model_name (str, optional): The name of the Gemini model to use. Defaults to "gemini-1.5-pro".
        temperature (float, optional):  Controls the randomness of the response. Defaults to 1.0.
        top_p (float, optional): Controls the diversity of the response. Defaults to 0.94.
        max_tokens (int, optional): The maximum number of tokens in the response. Defaults to 2000.

    Returns:
        JSONResponse:  A JSON response containing the chatbot's generated response.
    """
    try:
        if pdf_files:
            response = process_pdf_and_generate_response(
                pdf_files=pdf_files,
                user_input=user_input,
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                default_pdf_text=default_pdf_content 
            )
        else:
            response = await get_response(
                user_input=user_input,
                text_content=default_pdf_content, 
                model_name=model_name,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )
            
        return JSONResponse(content={"response": response}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG Chatbot Error: {str(e)}")