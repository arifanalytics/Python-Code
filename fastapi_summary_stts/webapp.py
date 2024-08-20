from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
import os
import json
import google.generativeai as genai
import nest_asyncio
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import re

app = FastAPI()
templates = Jinja2Templates(directory="templates")

if os.getenv("FASTAPI_ENV") == "development":
    nest_asyncio.apply()

# Initialize your model and other variables
document_analyzed = False
summary = None
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

def format_text(text):
    # Replace **text** with <b>text</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Replace any remaining * with <br>
    text = text.replace('*', '<br>')
    return text

# Route for main page
@app.get("/", response_class=HTMLResponse)
async def read_main(request: Request):
    return templates.TemplateResponse("analyze.html", {"request": request, "summary": summary, "show_conversation": document_analyzed})

# Route for analyzing documents
@app.post("/", response_class=HTMLResponse)
async def analyze_document(request: Request, api_key: str = Form(...)):
    global document_analyzed, summary, api, llm, uploaded_file_path

    # Initialize or update API key and models
    api = api_key
    genai.configure(api_key=api)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api)

    # Load the specified PDF file
    uploaded_file_path = "analysis_report (12).pdf"
    loader = PyPDFLoader(uploaded_file_path)
    docs = loader.load()

    # Define the Summarize Chain
    template = """Explain it in simple and clear terms. Provide key findings and actionable insights based on the content:
                "{text}" 
                CONCISE SUMMARY:"""

    prompt = PromptTemplate.from_template(template)

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

    # Invoke the chain to analyze the document
    response = stuff_chain.invoke(docs)
    summary = format_text(response["output_text"])
    document_analyzed = True

    # Create a dictionary to store the outputs
    outputs = {
        "summary": summary,
    }

    # Save the dictionary as a JSON file
    with open("output_summary.json", "w") as outfile:
        json.dump(outputs, outfile)

    return templates.TemplateResponse("analyze.html", {"request": request, "summary": summary, "show_conversation": document_analyzed})


# Route for asking questions
@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    global uploaded_file_path, question_responses, llm, api

    # Initialize question_responses if it doesn't exist
    if "question_responses" not in globals():
        question_responses = []

    loader = PyPDFLoader(uploaded_file_path)
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
