from fastapi import FastAPI, Form, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import YoutubeLoader, WebBaseLoader
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import json
import nest_asyncio

app = FastAPI()
if os.getenv("FASTAPI_ENV") == "development":
    nest_asyncio.apply()

templates = Jinja2Templates(directory="templates")

# Global variables
file_link_global = None
document_analyzed = False
summary = None
question_responses = []
api = None
llm = None

@app.get("/", response_class=HTMLResponse)
async def read_main(request: Request):
    return templates.TemplateResponse("analyze.html", {"request": request, "summary": summary, "show_conversation": document_analyzed, "question_responses": question_responses})

@app.post("/", response_class=HTMLResponse)
async def analyze_document(request: Request, api_key: str = Form(...), file_link: str = Form(...), iam: str = Form(...), context: str = Form(...), output: str = Form(...), summary_length: str = Form(...)):
    global file_link_global, document_analyzed, summary, api, llm
    api = api_key
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api)


    file_link_global = file_link

    if "youtube.com" in file_link_global:
        loader = YoutubeLoader.from_youtube_url(file_link_global, add_video_info=True, language=["en", "id"], translation="en")
    else:
        loader = WebBaseLoader(file_link_global)

    docs = loader.load()

    # Define the Summarize Chain template
    who = f"I am an {iam}"
    con = f"This file is about {context}"
    out = f"Give me the answer based on this question : {output}"
    sum = f"Write a {summary_length} concise summary of the following text."
    # Define the Summarize Chain
    template = who + con + out + sum + """Explain it in simple and clear terms. Provide key findings and actionable insights based on the content:
                "{text}" 
                CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(template)

    # Invoke the chain to analyze the document
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
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


@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    global file_link_global, question_responses, api, llm


    if "youtube.com" in file_link_global:
        loader = YoutubeLoader.from_youtube_url(file_link_global, add_video_info=True, language=["en", "id"], translation="en")
    else:
        loader = WebBaseLoader(file_link_global)

    docs = loader.load()
    text = "\n".join([doc.page_content for doc in docs])
    os.environ["GOOGLE_API_KEY"] = api
    
    # Check if there's a latest conversation in the session
    latest_conversation = request.cookies.get("latest_question_response", "")
    template1 = f"{question}\nWrite a concise summary of the following:\n{text}\nCONCISE SUMMARY:" + (f" Latest conversation: {latest_conversation}" if latest_conversation else "")
    prompt1 = PromptTemplate.from_template(template1)

    # Initialize the LLMChain with the prompt
    llm_chain1 = LLMChain(llm=llm, prompt=prompt1)

    # Invoke the chain with the entire document text to get the summary
    response1 = llm_chain1.invoke({"text": text})  # Await the async operation
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

    # Format the question and response
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
