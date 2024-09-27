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
from langchain.text_splitter import CharacterTextSplitter

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

        
        loader = PyPDFLoader(uploaded_file_path)
        docs = loader.load()
        prompt_template = PromptTemplate.from_template(
            """Explain it in simple and clear terms. Provide key findings and actionable insights based on the content:
                "{text}" 
                CONCISE SUMMARY:"""
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
    global uploaded_file_path, api, llm
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

        # Load the document and extract text
        loader = PyPDFLoader(uploaded_file_path)
        docs = loader.load()
        text = "\n".join([doc.page_content for doc in docs])
        os.environ["GOOGLE_API_KEY"] = api

        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)

        # Generate embeddings for the chunks
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        document_search = FAISS.from_texts(chunks, embeddings)

        # Generate query embedding and perform similarity search
        query_embedding = embeddings.embed_query(question)
        results = document_search.similarity_search_by_vector(query_embedding, k=3)

        if results:
            retrieved_texts = " ".join([result.page_content for result in results])

            # Define the Summarize Chain for the question
            latest_conversation = request.cookies.get("latest_question_response", "")
            template1 = (
                f"{question} Answer the question based on the following:\n\"{retrieved_texts}\"\n:" +
                (f" Answer the Question with only 3 sentences. Latest conversation: {latest_conversation}" if latest_conversation else "")
            )
            prompt1 = PromptTemplate.from_template(template1)

            # Initialize the LLMChain with the prompt
            llm_chain1 = LLMChain(llm=llm, prompt=prompt1)

            # Invoke the chain to get the summary
            response_chain = llm_chain1.invoke({"text": retrieved_texts})
            summary1 = response_chain["text"]

            # Return the response
            return AskResponse(meta={"status": "success", "code": 200}, question=question, result=summary1)

        else:
            current_response = "No relevant results found."
            return AskResponse(meta={"status": "success", "code": 200}, question=question, result=current_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
