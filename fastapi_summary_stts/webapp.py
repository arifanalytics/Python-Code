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

# Route for main page
@app.get("/", response_class=HTMLResponse)
async def read_main(request: Request):
    return templates.TemplateResponse("analyze.html", {"request": request, "summary": summary, "show_conversation": document_analyzed})

# Route for analyzing documents
@app.post("/", response_class=HTMLResponse)
async def analyze_document(request: Request, api_key: str = Form(...)):
    global document_analyzed, summary, api, llm

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
    summary = response["output_text"]
    document_analyzed = True

    # Create a dictionary to store the outputs
    outputs = {
        "summary": summary,
    }

    # Save the dictionary as a JSON file
    with open("output.json", "w") as outfile:
        json.dump(outputs, outfile)

    return templates.TemplateResponse("analyze.html", {"request": request, "summary": summary, "show_conversation": document_analyzed})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
