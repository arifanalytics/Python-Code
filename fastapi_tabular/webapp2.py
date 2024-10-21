import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
import google.generativeai as genai
from PIL import Image
from werkzeug.utils import secure_filename
import os
import json
from fpdf import FPDF
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import shutil
import re
from pydantic import BaseModel
from typing import List
from IPython.display import display, Markdown
import textwrap
from langchain.chains import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredCSVLoader, UnstructuredExcelLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi.middleware.cors import CORSMiddleware


safety_settings = [
    {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

sns.set_theme(color_codes=True)
uploaded_df = None
question_responses = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as necessary for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define Pydantic models for requests and responses
class AnalyzeDocumentRequest(BaseModel):
    api_key: str
    custom_question: str

class AnalyzeDocumentResponse(BaseModel):
    meta: dict
    plot1_path: str
    response1: str
    plot2_path: str
    response2: str
    pdf_file_path: str
    file_path: str
    columns: str

class MulticlassRequest(BaseModel):
    api_key: str
    custom_question: str
    target_variable: str
    columns_for_analysis: str  # Expecting comma-separated string

class MulticlassResponse(BaseModel):
    meta: dict
    plot3_path: str
    response3: str
    plot4_path: str
    response4: str
    pdf_file_path: str

class AskRequest(BaseModel):
    question: str
    api_key: str

class AskResponse(BaseModel):
    meta: dict
    question: str
    result: str

def format_text(text):
    # Replace **text** with <b>text</b>
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Replace any remaining * with <br>
    text = text.replace('*', '<br>')
    return text

def clean_data(df):
    # Step 1: Clean currency-related columns
    for col in df.columns:
        if any(x in col.lower() for x in ['value', 'price', 'cost', 'amount']):
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace('$', '').str.replace('£', '').str.replace('€', '').replace('[^\d.-]', '', regex=True).astype(float)
    
    # Step 2: Drop columns with more than 25% missing values
    null_percentage = df.isnull().sum() / len(df)
    columns_to_drop = null_percentage[null_percentage > 0.25].index
    df.drop(columns=columns_to_drop, inplace=True)
    
    # Step 3: Fill missing values for remaining columns
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if null_percentage[col] <= 0.25:
                if df[col].dtype in ['float64', 'int64']:
                    median_value = df[col].median()
                    df[col].fillna(median_value, inplace=True)
    
    # Step 4: Convert object-type columns to lowercase
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower()

    # Step 5: Drop columns with only one unique value
    unique_value_columns = [col for col in df.columns if df[col].nunique() == 1]
    df.drop(columns=unique_value_columns, inplace=True)

    return df




def clean_data2(df):
    for col in df.columns:
        if 'value' in col or 'price' in col or 'cost' in col or 'amount' in col or 'Value' in col or 'Price' in col or 'Cost' in col or 'Amount' in col:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace('$', '')
                df[col] = df[col].str.replace('£', '')
                df[col] = df[col].str.replace('€', '')
                df[col] = df[col].replace('[^\d.-]', '', regex=True).astype(float)


    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower()

    return df



def generate_plot(df, plot_path, plot_type):
    df = clean_data(df)
    excluded_words = ["name", "postal", "date", "phone", "address", "code", "id"]

    if plot_type == 'countplot':
        cat_vars = [col for col in df.select_dtypes(include='object').columns
                    if all(word not in col.lower() for word in excluded_words) and df[col].nunique() > 1]

        for col in cat_vars:
            if df[col].nunique() > 10:
                top_categories = df[col].value_counts().index[:10]
                df[col] = df[col].apply(lambda x: x if x in top_categories else 'Other')

        num_cols = len(cat_vars)
        num_rows = (num_cols + 1) // 2
        fig, axs = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, 5 * num_rows))
        axs = axs.flatten()

        for i, var in enumerate(cat_vars):
            category_counts = df[var].value_counts()
            top_values = category_counts.index[:10][::-1]
            filtered_df = df.copy()
            filtered_df[var] = pd.Categorical(filtered_df[var], categories=top_values, ordered=True)
            sns.countplot(x=var, data=filtered_df, order=top_values, ax=axs[i])
            axs[i].set_title(var)
            axs[i].tick_params(axis='x', rotation=30)

            total = len(filtered_df[var])
            for p in axs[i].patches:
                height = p.get_height()
                axs[i].annotate(f'{height / total:.1%}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')

            sample_size = filtered_df.shape[0]
            axs[i].annotate(f'Sample Size = {sample_size}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', va='center')

        for i in range(num_cols, len(axs)):
            fig.delaxes(axs[i])

    elif plot_type == 'histplot':
        num_vars = [col for col in df.select_dtypes(include=['int', 'float']).columns
                    if all(word not in col.lower() for word in excluded_words)]
        num_cols = len(num_vars)
        num_rows = (num_cols + 2) // 3
        fig, axs = plt.subplots(nrows=num_rows, ncols=min(3, num_cols), figsize=(15, 5 * num_rows))
        axs = axs.flatten()

        plot_index = 0

        for i, var in enumerate(num_vars):
            if len(df[var].unique()) == len(df):
                fig.delaxes(axs[plot_index])
            else:
                sns.histplot(df[var], ax=axs[plot_index], kde=True, stat="percent")
                axs[plot_index].set_title(var)
                axs[plot_index].set_xlabel('')

            sample_size = df.shape[0]
            axs[i].annotate(f'Sample Size = {sample_size}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', va='center')

            plot_index += 1

        for i in range(plot_index, len(axs)):
            fig.delaxes(axs[i])

    fig.tight_layout()
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


@app.post("/result", response_model=AnalyzeDocumentResponse)
async def result(api_key: str = Form(...), 
                 file: UploadFile = File(...), 
                 custom_question: str = Form(...)):
    global uploaded_df

    if file.filename == '':
        raise HTTPException(status_code=400, detail="No file selected")

    uploaded_filename = secure_filename(file.filename)
    file_path = os.path.join("static", uploaded_filename)

    with open(file_path, 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    if uploaded_filename.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif uploaded_filename.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    columns = df.columns.tolist()

    def generate_gemini_response(plot_path):
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        img = Image.open(plot_path)
        response = model.generate_content([custom_question + " Analyze the data insights from the chart.", img])
        response.resolve()
        return response.text

    try:
        plot1_path = generate_plot(df, 'static/plot1.png', 'countplot')
        plot2_path = generate_plot(df, 'static/plot2.png', 'histplot')

        response1 = generate_gemini_response(plot1_path)
        response2 = generate_gemini_response(plot2_path)

        uploaded_df = df

        outputs = {
            "barchart_visualization": plot1_path,
            "gemini_response1": response1,
            "histoplot_visualization": plot2_path,
            "gemini_response2": response2
        }

        with open("output.json", "w") as outfile:
            json.dump(outputs, outfile)

        pdf = FPDF()
        pdf.set_font("Arial", size=12)

        pdf.add_page()
        pdf.cell(200, 10, txt="Single Countplot Barchart", ln=True, align='C')
        pdf.image(plot1_path, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Single Countplot Barchart Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, response1)

        pdf.add_page()
        pdf.cell(200, 10, txt="Single Histoplot", ln=True, align='C')
        pdf.image(plot2_path, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Single Histoplot Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, response2)

        pdf_file_path = os.path.join("static", "output.pdf")
        pdf.output(pdf_file_path)

        pdf_file_path = os.path.join("static", "output.pdf")
        pdf_file_path = pdf_file_path.replace("\\", "/")

        return AnalyzeDocumentResponse(
            meta={"status": "success", "code": 200},
            plot1_path=plot1_path,
            response1=response1,
            plot2_path=plot2_path,
            response2=response2,
            pdf_file_path=pdf_file_path,
            file_path= file_path,
             columns=", ".join(columns)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/multiclass", response_model=MulticlassResponse)
async def multiclass(
    request: Request,
    target_variable: str = Form(...),
    custom_question: str = Form(...),
    api_key: str = Form(...),
    file: UploadFile = File(...),
    columns_for_analysis: str = Form(...),  # Changed to str to handle CSV string input
):
    global document_analyzed

    try:
        # Read the file content into a DataFrame
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file, encoding='utf-8')
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Process the columns_for_analysis
        columns_for_analysis_list = [col.strip() for col in columns_for_analysis.split(',')]

        # Ensure the columns exist in the DataFrame
        missing_cols = [col for col in columns_for_analysis_list if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Columns not found in the dataset: {', '.join(missing_cols)}")

        # Select the target variable and columns for analysis from the DataFrame
        if target_variable not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target variable '{target_variable}' not found in the dataset")

        target_variable_data = df[target_variable]
        columns_for_analysis_data = df[columns_for_analysis_list]

        # Concatenate target variable and columns for analysis into a single DataFrame
        df = pd.concat([target_variable_data, columns_for_analysis_data], axis=1)

        # Clean the data (if needed)
        df = clean_data2(df)

        # Generate visualizations

        # Multiclass Barplot
        excluded_words = ["name", "postal", "date", "phone", "address", "id"]

        # Get the names of all columns with data type 'object' (categorical variables)
        cat_vars = [col for col in df.select_dtypes(include=['object']).columns
                    if all(word not in col.lower() for word in excluded_words)]

        # Exclude the target variable from the list if it exists in cat_vars
        if target_variable in cat_vars:
            cat_vars.remove(target_variable)

        # Create a figure with subplots, but only include the required number of subplots
        num_cols = len(cat_vars)
        num_rows = (num_cols + 2) // 3  # To make sure there are enough rows for the subplots
        fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
        axs = axs.flatten()

        # Create a count plot for each categorical variable
        for i, var in enumerate(cat_vars):
            top_categories = df[var].value_counts().nlargest(5).index
            filtered_df = df[df[var].notnull() & df[var].isin(top_categories)]  # Exclude rows with NaN values in the variable

            # Replace less frequent categories with "Other" if there are more than 5 unique values
            if df[var].nunique() > 5:
                other_categories = df[var].value_counts().index[5:]
                filtered_df[var] = filtered_df[var].apply(lambda x: x if x in top_categories else 'Other')

            sns.countplot(x=var, hue=target_variable, data=filtered_df, ax=axs[i], stat="percent")
            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45)

            # Change y-axis label to represent percentage
            axs[i].set_ylabel('Percentage')

            # Annotate the subplot with sample size
            sample_size = df.shape[0]
            axs[i].annotate(f'Sample Size = {sample_size}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', va='center')

        # Remove any remaining blank subplots
        for i in range(num_cols, len(axs)):
            fig.delaxes(axs[i])

        plt.xticks(rotation=45)
        plt.tight_layout()
        plot3_path = "static/multiclass_barplot.png"
        plt.savefig(plot3_path)
        plt.close(fig)

        # Multiclass Histplot
        int_vars = df.select_dtypes(include=['int', 'float']).columns.tolist()
        int_vars = [col for col in int_vars if col != target_variable]

        # Create a figure with subplots
        num_cols = len(int_vars)
        num_rows = (num_cols + 2) // 3  # To make sure there are enough rows for the subplots
        fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
        axs = axs.flatten()

        # Create a histogram for each integer variable with hue=target_variable
        for i, var in enumerate(int_vars):
            sns.histplot(data=df, x=var, hue=target_variable, kde=True, ax=axs[i], stat="percent")
            axs[i].set_title(var)

            # Annotate the subplot with sample size
            sample_size = df.shape[0]
            axs[i].annotate(f'Sample Size = {sample_size}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', va='center')

        # Remove any extra empty subplots if needed
        if num_cols < len(axs):
            for i in range(num_cols, len(axs)):
                fig.delaxes(axs[i])

        fig.tight_layout()
        plt.xticks(rotation=45)
        plot4_path = "static/multiclass_histplot.png"
        plt.savefig(plot4_path)
        plt.close(fig)

        # Google Gemini Responses
        genai.configure(api_key=api_key)

        # Response for the barplot
        img_barplot = Image.open(plot3_path)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response3 = model.generate_content([custom_question, img_barplot]).text

        # Response for the histplot
        img_histplot = Image.open(plot4_path)
        response4 = model.generate_content([custom_question, img_histplot]).text

        document_analyzed = True

        # Create a dictionary to store the outputs
        outputs = {
            "multiBarchart_visualization": plot3_path,
            "gemini_response3": response3,
            "multiHistoplot_visualization": plot4_path,
            "gemini_response4": response4
        }

        # Save the dictionary as a JSON file
        with open("output1.json", "w") as outfile:
            json.dump(outputs, outfile)

        def safe_encode(text):
            try:
                return text.encode('latin1', errors='replace').decode('latin1')  # Replace invalid characters
            except Exception as e:
                return f"Error encoding text: {str(e)}"

        # Generate PDF with the results
        pdf = FPDF()
        pdf.set_font("Arial", size=12)

        # Add content to the PDF
        # Multiclass Countplot Barchart and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Multiclass Countplot Barchart", ln=True, align='C')
        pdf.image(plot3_path, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Multiclass Countplot Barchart Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(response3))

        # Multiclass Histplot and response
        pdf.add_page()
        pdf.cell(200, 10, txt="Multiclass Histplot", ln=True, align='C')
        pdf.image(plot4_path, x=10, y=30, w=190)
        pdf.add_page()
        pdf.cell(200, 10, txt="Multiclass Histplot Google Gemini Response", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, safe_encode(response4))

        pdf_output_path = 'static/analysis_report_complete.pdf'
        pdf.output(pdf_output_path)
        pdf_file_path = pdf_output_path.replace("\\", "/")

        return MulticlassResponse(
            meta={"status": "success", "code": 200},
            plot3_path=plot3_path,
            plot4_path=plot4_path,
            response3=response3,
            response4=response4,
            pdf_file_path=pdf_file_path
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



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

        # Initialize the LLM model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key)

        uploaded_file_path = "uploaded_file" + os.path.splitext(file.filename)[1]
        with open(uploaded_file_path, "wb") as f:
            f.write(file.file.read())
        # Determine the file extension and select the appropriate loader
        loader = None
        file_extension = os.path.splitext(uploaded_file_path)[1].lower()

        if file_extension == ".csv":
            loader = UnstructuredCSVLoader(uploaded_file_path, mode="elements")
        elif file_extension == ".xlsx":
            loader = UnstructuredExcelLoader(uploaded_file_path, mode="elements")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        # Load and process the document
        try:
            docs = loader.load()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading document: {str(e)}")

        # Combine document text
        text = "\n".join([doc.page_content for doc in docs])
        os.environ["GOOGLE_API_KEY"] = api_key

        # Initialize embeddings and create FAISS vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        document_search = FAISS.from_texts(chunks, embeddings)

        # Generate query embedding and perform similarity search
        query_embedding = embeddings.embed_query(question)
        results = document_search.similarity_search_by_vector(query_embedding, k=3)

        if results:
            retrieved_texts = " ".join([result.page_content for result in results])

            # Define the Summarize Chain for the question
            latest_conversation = request.cookies.get("latest_question_response", "")
            template1 = (
                f"{question} Answer the question based on the following:\n\"{text}\"\n:" +
                (f" Answer the Question with only 3 sentences. Latest conversation: {latest_conversation}" if latest_conversation else "")
            )
            prompt1 = PromptTemplate.from_template(template1)

            # Initialize the LLMChain with the prompt
            llm_chain1 = LLMChain(llm=llm, prompt=prompt1)

            # Invoke the chain to get the summary
            try:
                response_chain = llm_chain1.invoke({"text": text})
                summary1 = response_chain["text"]
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error invoking LLMChain: {str(e)}")

            # Generate embeddings for the summary
            try:
                summary_embedding = embeddings.embed_query(summary1)
                document_search = FAISS.from_texts([summary1], embeddings)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error generating embeddings: {str(e)}")

            # Perform a search on the FAISS vector database
            try:
                if document_search:
                    query_embedding = embeddings.embed_query(question)
                    results = document_search.similarity_search_by_vector(query_embedding, k=1)

                    if results:
                        current_response = format_text(results[0].page_content)
                    else:
                        current_response = "No matching document found in the database."
                else:
                    current_response = "Vector database not initialized."
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error during similarity search: {str(e)}")
        else:
            current_response = "No relevant results found."

        # Append the question and response from FAISS search
        current_question = f"You asked: {question}"
        question_responses.append((current_question, current_response))

        # Save all results to output_summary.json
        save_to_json(question_responses)

        return AskResponse(meta={"status": "success", "code": 200}, question=question, result=current_response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")



def save_to_json(question_responses):
    outputs = {
        "question_responses": question_responses
    }
    with open("output_summary.json", "w") as outfile:
        json.dump(outputs, outfile)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
