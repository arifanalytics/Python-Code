from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader, UnstructuredCSVLoader, UnstructuredExcelLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import json
import os
import google.generativeai as genai
import nest_asyncio

nest_asyncio.apply()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Initialize your model and other variables
uploaded_file_path = None
document_analyzed = False
summary = None
question_responses = []
api = None
llm = None

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

# Route for main page
@app.get("/", response_class=HTMLResponse)
async def read_main(request: Request):
    return templates.TemplateResponse("analyze.html", {"request": request, "summary": summary, "show_conversation": document_analyzed, "question_responses": question_responses})

# Route for analyzing documents
@app.post("/", response_class=HTMLResponse)
async def analyze_document(request: Request, api_key: str = Form(...), file: UploadFile = File(...), summary_length: str = Form(...)):
    global uploaded_file_path, document_analyzed, summary, question_responses, api, llm

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
        # Use Gemini API for MP3 files
        audio_file = genai.upload_file(path=uploaded_file_path)
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        sum = f"Write a {summary_length} concise summary of the following text."
        prompt = sum + "Explain it in simple and clear terms. Provide key findings and actionable insights based on the content"
        response = model.generate_content([prompt, audio_file], safety_settings=safety_settings)
        summary = response.text
        document_analyzed = True
        # Create a dictionary to store the outputs
        outputs = {
            "summary": summary,
        }

        # Save the dictionary as a JSON file
        with open("output_summary.json", "w") as outfile:
            json.dump(outputs, outfile)

        return templates.TemplateResponse("analyze.html", {"request": request, "summary": summary, "show_conversation": document_analyzed, "question_responses": question_responses})

    docs = loader.load()

    sum = f"Write a {summary_length} concise summary of the following text."
    # Define the Summarize Chain
    template = sum + """Explain it in simple and clear terms. Provide key findings and actionable insights based on the content:
                "{text}" 
                CONCISE SUMMARY:"""

    prompt = PromptTemplate.from_template(template)

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

    # Invoke the chain to analyze the document
    response = stuff_chain.invoke(docs)
    summary = response["output_text"]
    document_analyzed = True
    # Create a dictionary to store the outputs
    outputs = {
        "summary": summary,
    }

    # Save the dictionary as a JSON file
    with open("output.json", "w") as outfile:
        json.dump(outputs, outfile)

    return templates.TemplateResponse("analyze.html", {"request": request, "summary": summary, "show_conversation": document_analyzed, "question_responses": question_responses})

# Route for asking questions
@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    global uploaded_file_path, question_responses, llm, api

    if uploaded_file_path:
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
                    current_response = results[0].page_content
                else:
                    current_response = "No matching document found in the database."
            else:
                current_response = "Vector database not initialized."

            # Append the question and response from FAISS search
            question_responses.append((current_question, current_response))

            # Save all results including FAISS response to output.json
            save_to_json(summary, question_responses)

            # Save the latest question and response to the session
            response = templates.TemplateResponse("analyze.html", {"request": request, "summary": summary, "show_conversation": document_analyzed, "question_responses": question_responses})
            response.set_cookie(key="latest_question_response", value=current_response)
            return response

        docs = loader.load()
        text = "\n".join([doc.page_content for doc in docs])
        os.environ["GOOGLE_API_KEY"] = api

        # Define the Summarize Chain for the question
        latest_conversation = request.cookies.get("latest_question_response", "")
        template1 = question + """answer the question based on the following:
                    "{text}" 
                    :""" + (f" Latest conversation: {latest_conversation}" if latest_conversation else "")
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
                current_response = results[0].page_content
            else:
                current_response = "No matching document found in the database."
        else:
            current_response = "Vector database not initialized."

        # Append the question and response from FAISS search
        current_question = f"You asked: {question}"
        question_responses.append((current_question, current_response))

        # Save all results to output.json
        save_to_json(summary, question_responses)

        # Save the latest question and response to the session
        response = templates.TemplateResponse("analyze.html", {"request": request, "summary": summary, "show_conversation": document_analyzed, "question_responses": question_responses})
        response.set_cookie(key="latest_question_response", value=current_response)
        return response

# Function to save to JSON
def save_to_json(summary, question_responses):
    output_data = {
        'summary': summary,
        'question_responses': question_responses
    }
    with open('output.json', 'w') as json_file:
        json.dump(output_data, json_file)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
