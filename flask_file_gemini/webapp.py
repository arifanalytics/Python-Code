from flask import Flask, render_template, request, redirect, session
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
import json
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
import os

app = Flask(__name__)
app.secret_key = "file_gemini"

# Initialize your model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key="AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA")
genai_api_key = "AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA"
genai.configure(api_key=genai_api_key)

# Variable to store the uploaded file path
uploaded_file_path = None

# Flag to track if the document has been analyzed and conversation can start
document_analyzed = False
summary = None
question_responses = []  # Store multiple question responses

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

@app.route("/", methods=["GET", "POST"])
def analyze_document():
    global document_analyzed, summary, uploaded_file_path
    import google.generativeai as genai

    if request.method == "POST":
        file = request.files["file"]
        # Save the uploaded file
        uploaded_file_path = "uploaded_file" + os.path.splitext(file.filename)[1]
        file.save(uploaded_file_path)
        summary_length = request.form["summary_length"]

        # Determine the file type and load accordingly
        file_extension = os.path.splitext(uploaded_file_path)[1].lower()
        if file_extension == ".pdf":
            loader = PyPDFLoader(uploaded_file_path)
        elif file_extension == ".csv":
            loader = UnstructuredCSVLoader(uploaded_file_path, mode="elements", encoding="utf8")
        elif file_extension == ".xlsx":
            loader = UnstructuredExcelLoader(uploaded_file_path, mode="elements")
        elif file_extension == ".docx":
            loader = loader = Docx2txtLoader(uploaded_file_path)
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
            return redirect("/")

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

       

    return render_template("analyze.html", summary=summary, show_conversation=document_analyzed)


@app.route("/ask", methods=["POST"])
def ask_question():
    global uploaded_file_path, question_responses

    if uploaded_file_path:
        question = request.form["question"]

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
            # Use Gemini API for MP3 files
            audio_file = genai.upload_file(path=uploaded_file_path)
            model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            latest_conversation = session.get("latest_question_response", "")
            prompt = "Answer the question based on the speech: " + question + (f" Latest conversation: {latest_conversation}" if latest_conversation else "")
            response = model.generate_content([prompt, audio_file], safety_settings=safety_settings)
            current_response = response.text
            current_question = f"You asked: {question}"
            question_responses.append((current_question, current_response))
            save_to_json(summary, question_responses)
            session["latest_question_response"] = current_response
            return render_template("analyze.html", summary=summary, show_conversation=True, question_responses=question_responses)

        docs = loader.load()
        text = "\n".join([doc.page_content for doc in docs])
        os.environ["GOOGLE_API_KEY"] = "AIzaSyCFI6cTqFdS-mpZBfi7kxwygewtnuF7PfA"

        # Define the Summarize Chain for the question
        latest_conversation = session.get("latest_question_response", "")
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
