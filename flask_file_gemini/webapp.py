from flask import Flask, render_template, request, redirect
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
import os

app = Flask(__name__)

# Initialize your model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key="AIzaSyAQLXJ6ROBzMycImPVp2jTlbB3zIpEWmhM")

# Define the Summarize Chain
template = """Write a concise summary of the following:
"{text}" and give some key findings and actionable insights based on the content
CONCISE SUMMARY:"""

prompt = PromptTemplate.from_template(template)

llm_chain = LLMChain(llm=llm, prompt=prompt)
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

# Variable to store the uploaded file path
uploaded_file_path = None

# Flag to track if the document has been analyzed and conversation can start
document_analyzed = False
summary = None
question_responses = []  # Store multiple question responses

@app.route("/", methods=["GET", "POST"])
def analyze_document():
    global document_analyzed, summary, uploaded_file_path

    if request.method == "POST":
        file = request.files["file"]
        # Save the uploaded file
        uploaded_file_path = "uploaded_file" + os.path.splitext(file.filename)[1]
        file.save(uploaded_file_path)

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
        else:
            return "Unsupported file type", 400

        docs = loader.load()

        # Invoke the chain to analyze the document
        response = stuff_chain.invoke(docs)
        summary = response["output_text"]
        document_analyzed = True

        # Delete output.json if it exists to clear previous conversation history
        if os.path.exists("output.json"):
            os.remove("output.json")
        # Redirect to the same route to clear the form data
        return redirect("/")

    # Delete output.json if it exists to clear previous conversation history
    if os.path.exists("output.json"):
        os.remove("output.json")

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
            loader = UnstructuredCSVLoader(uploaded_file_path, mode="elements", encoding="utf8")
        elif file_extension == ".xlsx":
            loader = UnstructuredExcelLoader(uploaded_file_path, mode="elements")
        elif file_extension == ".docx":
            loader = loader = Docx2txtLoader(uploaded_file_path)
        elif file_extension == ".pptx":
            loader = UnstructuredPowerPointLoader(uploaded_file_path)
        else:
            return "Unsupported file type", 400

        docs = loader.load()

        # Define the Summarize Chain
        template1 = question + """Write a concise summary of the following:
        "{text}"
        CONCISE SUMMARY:"""

        prompt1 = PromptTemplate.from_template(template1)

        llm_chain1 = LLMChain(llm=llm, prompt=prompt1)
        stuff_chain1 = StuffDocumentsChain(llm_chain=llm_chain1, document_variable_name="text")

        # Format the question and response
        current_question = f"You asked: {question}"
        response1 = stuff_chain1.invoke(docs)
        current_response = response1["output_text"]

        # Store the current question and response
        question_responses.append((current_question, current_response))

        # Save all results to output.json
        save_to_json(summary, question_responses)

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
