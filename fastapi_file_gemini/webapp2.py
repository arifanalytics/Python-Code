from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import (
    PyPDFLoader, UnstructuredCSVLoader, UnstructuredExcelLoader,
    Docx2txtLoader, UnstructuredPowerPointLoader
)
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
import google.generativeai as genai
import re
import nest_asyncio

app = FastAPI()

if os.getenv("FASTAPI_ENV") == "development":
    nest_asyncio.apply()

# Initialize your model and other variables
uploaded_file_path = None
document_analyzed = False
summary = None
question_responses = []
api = None
llm = None

safety_settings = [
    {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

def format_text(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = text.replace('*', '<br>')
    return text

# Define Pydantic models for requests and responses
class AnalyzeDocumentRequest(BaseModel):
    api_key: str
    iam: str
    context: str
    output: str
    summary_length: str

class AnalyzeDocumentResponse(BaseModel):
    meta: dict
    summary: str

class AskRequest(BaseModel):
    question: str
    api_key: str

class AskResponse(BaseModel):
    meta: dict
    question: str
    result: str

# Route for analyzing documents
# Route for analyzing documents
@app.post("/", response_model=AnalyzeDocumentResponse)
async def analyze_document(
    api_key: str = Form(...),
    iam: str = Form(...),
    context: str = Form(...),
    output: str = Form(...),
    summary_length: str = Form(...),
    file: UploadFile = File(...)
):
    global uploaded_file_path, document_analyzed, summary, api, llm
    loader = None

    try:
        # Initialize or update API key and models
        api = api_key
        genai.configure(api_key=api)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api)

        # Save the uploaded file
        uploaded_file_path = "uploaded_file" + os.path.splitext(file.filename)[1]
        with open(uploaded_file_path, "wb") as f:
            f.write(file.file.read())

        # Determine the file type and load accordingly
        file_extension = os.path.splitext(uploaded_file_path)[1].lower()

        if file_extension == ".pdf":
            loader = PyPDFLoader(uploaded_file_path)
        elif file_extension == ".csv":
            loader = UnstructuredCSVLoader(uploaded_file_path, mode="elements", encoding="utf8")
        elif file_extension == ".xlsx":
            loader = UnstructuredExcelLoader(uploaded_file_path, mode="elements")
        elif file_extension == ".docx":
            loader = Docx2txtLoader(uploaded_file_path)
        elif file_extension == ".pptx":
            loader = UnstructuredPowerPointLoader(uploaded_file_path)
        elif file_extension == ".mp3":
            # Process audio files differently
            audio_file = genai.upload_file(path=uploaded_file_path)
            model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            prompt = f"I am an {iam}. This file is about {context}. Answer the question based on this file: {output}. Write a {summary_length} concise summary."
            response = model.generate_content([prompt, audio_file], safety_settings=safety_settings)
            summary = format_text(response.text)
            document_analyzed = True
            return AnalyzeDocumentResponse(meta={"status": "success", "code": 200}, summary=summary)

        # If no loader is set, raise an exception
        if loader is None:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")

        docs = loader.load()
        prompt_template = PromptTemplate.from_template(
            f"I am an {iam}. This file is about {context}. Answer the question based on this file: {output}. Write a {summary_length} concise summary of the following text: {{text}}"
        )
        llm_chain = LLMChain(llm=llm, prompt=prompt_template)
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
        response = stuff_chain.invoke(docs)
        summary = format_text(response["output_text"])
        document_analyzed = True

        return AnalyzeDocumentResponse(meta={"status": "success", "code": 200}, summary=summary)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


# Route for answering questions
@app.post("/ask", response_model=AskResponse)
async def ask_question(
    request: Request,
    api_key: str = Form(...),
    question: str = Form(...),
    file: UploadFile = File(...)
):
    global uploaded_file_path, document_analyzed, summary, api, llm
    loader = None

    try:
        # Initialize or update API key and models
        api = api_key
        genai.configure(api_key=api)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api)

        # Save the uploaded file
        uploaded_file_path = "uploaded_file" + os.path.splitext(file.filename)[1]
        with open(uploaded_file_path, "wb") as f:
            f.write(file.file.read())

        # Determine the file type and load accordingly
        file_extension = os.path.splitext(uploaded_file_path)[1].lower()

        if file_extension == ".pdf":
            loader = PyPDFLoader(uploaded_file_path)
        elif file_extension == ".csv":
            loader = UnstructuredCSVLoader(uploaded_file_path, mode="elements")
        elif file_extension == ".xlsx":
            loader = UnstructuredExcelLoader(uploaded_file_path, mode="elements")
        elif file_extension == ".docx":
            loader = Docx2txtLoader(uploaded_file_path)
        elif file_extension == ".pptx":
            loader = UnstructuredPowerPointLoader(uploaded_file_path)
        elif file_extension == ".mp3":
            audio_file = genai.upload_file(path=uploaded_file_path)
            model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            latest_conversation = request.cookies.get("latest_question_response", "")
            prompt = "Answer the question based on the speech: " + question + (f" Latest conversation: {latest_conversation}" if latest_conversation else "")
            response = model.generate_content([prompt, audio_file], safety_settings=safety_settings)
            current_response = response.text
            current_question = f"You asked: {question}"

            # Save the latest question and response to the session
            question_responses.append((current_question, current_response))

            # Perform vector embedding and search
            text = current_response  # Use the summary generated from the MP3 content
            os.environ["GOOGLE_API_KEY"] = api
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            summary_embedding = embeddings.embed_query(text)
            document_search = FAISS.from_texts([text], embeddings)

            if document_search:
                query_embedding = embeddings.embed_query(question)
                results = document_search.similarity_search_by_vector(query_embedding, k=1)

                if results:
                    current_response = format_text(results[0].page_content)
                else:
                    current_response = "No matching document found in the database."
            else:
                current_response = "Vector database not initialized."

            return AskResponse(meta={"status": "success", "code": 200}, question=question, result=current_response)

        # If no loader is set, raise an exception
        if loader is None:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_extension}")

        docs = loader.load()
        text = "\n".join([doc.page_content for doc in docs])
        os.environ["GOOGLE_API_KEY"] = api

        # Define the Summarize Chain for the question
        latest_conversation = request.cookies.get("latest_question_response", "")
        template1 = question + """answer the question based on the following:
                    "{text}" 
                    :""" + (f" Answer the Question with only 3 sentences. Latest conversation: {latest_conversation}" if latest_conversation else "")
        prompt1 = PromptTemplate.from_template(template1)

        # Initialize the LLMChain with the prompt
        llm_chain1 = LLMChain(llm=llm, prompt=prompt1)

        # Invoke the chain with the entire document text to get the summary
        response1 = llm_chain1.invoke({"text": text})
        summary1 = response1["text"]

        # Generate embeddings for the summary
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        summary_embedding = embeddings.embed_query(summary1)
        document_search = FAISS.from_texts([summary1], embeddings)

        # Perform a search on the FAISS vector database if it's initialized
        if document_search:
            query_embedding = embeddings.embed_query(question)
            results = document_search.similarity_search_by_vector(query_embedding, k=1)

            if results:
                current_response = format_text(results[0].page_content)
            else:
                current_response = "No matching document found in the database."
        else:
            current_response = "Vector database not initialized."

        return AskResponse(meta={"status": "success", "code": 200}, question=question, result=current_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
