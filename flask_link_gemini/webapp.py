from flask import Flask, render_template, request, session
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
import json
import os
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import nest_asyncio
nest_asyncio.apply()
import asyncio

app = Flask(__name__)
app.secret_key = "link_gemini"



# Define the Summarize Chain
template = """Write a concise summary of the following:
"{text}" and give some key findings and actionable insights based on the content
CONCISE SUMMARY:"""

prompt = PromptTemplate.from_template(template)


# Variable to store the uploaded file path
file_link = None

# Flag to track if the document has been analyzed and conversation can start
document_analyzed = False
summary = None
question_responses = []  # Store multiple question responses

@app.route("/", methods=["GET", "POST"])
def analyze_document():
    global document_analyzed, summary, file_link, api, llm

    if request.method == "POST":
        api = request.form["api_key"]
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
        file_link = request.form["file_link"]
        if "youtube.com" in file_link:
            loader = YoutubeLoader.from_youtube_url(file_link,
                                                    add_video_info=True,
                                                    language=["en", "id"],
                                                    translation="en",)
        else:
            loader = WebBaseLoader(file_link)
            
        docs = loader.load()

        # Invoke the chain to analyze the document
        response = stuff_chain.invoke(docs)
        summary = response["output_text"]
        document_analyzed = True

    # Delete output.json if it exists to clear previous conversation history
    if os.path.exists("output.json"):
        os.remove("output.json")

    return render_template("analyze.html", summary=summary, show_conversation=document_analyzed)


@app.route("/ask", methods=["POST"])
def ask_question():
    global file_link, question_responses, api, llm

    if file_link:
        question = request.form["question"]
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

        if "youtube.com" in file_link:
            loader = YoutubeLoader.from_youtube_url(file_link,
                                                    add_video_info=True,
                                                    language=["en", "id"],
                                                    translation="en")
        else:
            loader = WebBaseLoader(file_link)

        docs = loader.load()
        text = "\n".join([doc.page_content for doc in docs])
        os.environ["GOOGLE_API_KEY"] = api

        # Check if there's a latest conversation in the session
        latest_conversation = session.get("latest_question_response", "")
        template1 = f"{question}\nWrite a concise summary of the following:\n{text}\nCONCISE SUMMARY:" + (f" Latest conversation: {latest_conversation}" if latest_conversation else "")
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

        # Format the question and response
        current_question = f"You asked: {question}"
        question_responses.append((current_question, current_response))

        # Save all results to output.json
        save_to_json(summary, question_responses)

        # Save the latest question and response to the session
        session["latest_question_response"] = current_response

    return render_template("analyze.html", summary=summary, show_conversation=True, question_responses=question_responses)

# Function to save to JSON
def save_to_json(summary, question_responses):
    output_data = {
        'summary': summary,
        'question_responses': question_responses
    }
    with open('output.json', 'w') as json_file:
        json.dump(output_data, json_file)

if __name__ == "__main__":
    app.run(debug=True)
