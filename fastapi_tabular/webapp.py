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
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from typing import List
import textwrap
from IPython.display import display, Markdown
from PIL import Image
import shutil
from werkzeug.utils import secure_filename
import urllib.parse

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

sns.set_theme(color_codes=True)
uploaded_df = None

def clean_data(df):
    for col in df.columns:
        if any(x in col.lower() for x in ['value', 'price', 'cost', 'amount']):
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace('$', '').str.replace('£', '').str.replace('€', '').replace('[^\d.-]', '', regex=True).astype(float)

    null_percentage = df.isnull().sum() / len(df)
    columns_to_drop = null_percentage[null_percentage > 0.25].index
    df.drop(columns=columns_to_drop, inplace=True)

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if null_percentage[col] <= 0.25:
                if df[col].dtype in ['float64', 'int64']:
                    median_value = df[col].median()
                    df[col].fillna(median_value, inplace=True)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower()

    return df



def clean_data2(df):
    for col in df.columns:
        if 'value' in col or 'price' in col or 'cost' in col or 'amount' in col or 'Value' in col or 'Price' in col or 'Cost' in col or 'Amount' in col:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace('$', '')
                df[col] = df[col].str.replace('£', '')
                df[col] = df[col].str.replace('€', '')
                df[col] = df[col].replace('[^\d.-]', '', regex=True).astype(float)

    null_percentage = df.isnull().sum() / len(df)

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if null_percentage[col] <= 0.25:
                if df[col].dtype in ['float64', 'int64']:
                    median_value = df[col].median()
                    df[col].fillna(median_value, inplace=True)

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
        fig, axs = plt.subplots(nrows=num_rows, ncols=2, figsize=(15, 5*num_rows))
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
                axs[i].annotate(f'{height/total:.1%}', (p.get_x() + p.get_width() / 2., height), ha='center', va='bottom')

            sample_size = filtered_df.shape[0]
            axs[i].annotate(f'Sample Size = {sample_size}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', va='center')

        for i in range(num_cols, len(axs)):
            fig.delaxes(axs[i])

    elif plot_type == 'histplot':
        num_vars = [col for col in df.select_dtypes(include=['int', 'float']).columns
                    if all(word not in col.lower() for word in excluded_words)]
        num_cols = len(num_vars)
        num_rows = (num_cols + 2) // 3
        fig, axs = plt.subplots(nrows=num_rows, ncols=min(3, num_cols), figsize=(15, 5*num_rows))
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

@app.get("/", response_class=HTMLResponse)
async def upload_file(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/result")
async def result(request: Request, 
                 api_key: str = Form(...), 
                 file: UploadFile = File(...), 
                 custom_question: str = Form(...)):
    global uploaded_df, uploaded_filename, plot1_path, plot2_path, response1, response2, api, question

    api = api_key
    uploaded_file = file
    
    if file.filename == '':
        raise HTTPException(status_code=400, detail="No file selected")

    # Secure and validate the file name
    uploaded_filename = secure_filename(file.filename)

    # Determine file path based on file type
    if uploaded_filename.endswith('.csv'):
        file_path = 'dataset.csv'
        # Save the file
        with open(file_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Read the file into a DataFrame
        df = pd.read_csv(file_path, encoding='utf-8')
    
    elif uploaded_filename.endswith('.xlsx'):
        file_path = 'dataset.xlsx'
        # Save the file
        with open(file_path, 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        # Read the file into a DataFrame
        df = pd.read_excel(file_path)
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    columns = df.columns.tolist()

    def generate_gemini_response(plot_path):
        global question
        question = custom_question
        genai.configure(api_key=api)
        img = Image.open(plot_path)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content([
            question + " As a marketing consultant, I want to understand consumer insights based on the chart and the market context so I can use the key findings to formulate actionable insights", 
            img
        ])
        response.resolve()
        return response.text

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

    def safe_encode(text):
        try:
            return text.encode('latin1', errors='replace').decode('latin1')
        except Exception as e:
            return f"Error encoding text: {str(e)}"

    pdf = FPDF()
    pdf.set_font("Arial", size=12)

    # Single Countplot Barchart and response
    pdf.add_page()
    pdf.cell(200, 10, txt="Single Countplot Barchart", ln=True, align='C')
    pdf.image(plot1_path, x=10, y=30, w=190)
    pdf.add_page()
    pdf.cell(200, 10, txt="Single Countplot Barchart Google Gemini Response", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, safe_encode(response1))

    # Single Histplot and response
    pdf.add_page()
    pdf.cell(200, 10, txt="Single Histplot", ln=True, align='C')
    pdf.image(plot2_path, x=10, y=30, w=190)
    pdf.add_page()
    pdf.cell(200, 10, txt="Single Histplot Google Gemini Response", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, safe_encode(response2))

    pdf_output_path = 'static/analysis_report.pdf'
    pdf.output(pdf_output_path)

    return templates.TemplateResponse("upload.html", {
        "request": request,
        "response1": response1,
        "response2": response2,
        "plot1_path": plot1_path,
        "plot2_path": plot2_path,
        "columns": columns})

@app.get("/download_pdf")
async def download_pdf():
    pdf_output_path = 'static/analysis_report.pdf'
    return FileResponse(pdf_output_path, media_type='application/pdf', filename=os.path.basename(pdf_output_path))





@app.post("/streamlit")
async def streamlit(request: Request,
                    target_variable: str = Form(...),
                    columns_for_analysis: List[str] = Form(...)):
    global uploaded_df, uploaded_filename, plot1_path, plot2_path, response1, response2, api, question
    target_variable_html = None
    columns_for_analysis_html = None
    response3 = None
    response4 = None
    plot3_path = None
    plot4_path = None


    if uploaded_df is None:
        raise HTTPException(status_code=400, detail="No CSV file uploaded")


    df = uploaded_df

    # Process the uploaded file
    if uploaded_filename.endswith('.csv'):
        df = pd.read_csv('dataset.csv', encoding='utf-8')
    elif uploaded_filename.endswith('.xlsx'):
        df = pd.read_excel('dataset.xlsx')

    # Select the target variable and columns for analysis from the original DataFrame
    target_variable_data = df[target_variable]
    columns_for_analysis_data = df[columns_for_analysis]

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
                
        sns.countplot(x=var, hue=target_variable, stat="percent", data=filtered_df, ax=axs[i])
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
    # Get the names of all columns with data type 'object' (categorical columns)
    cat_cols = df.columns.tolist()

    # Get the names of all columns with data type 'int'
    int_vars = df.select_dtypes(include=['int', 'float']).columns.tolist()
    int_vars = [col for col in int_vars if col != target_variable]

    # Create a figure with subplots
    num_cols = len(int_vars)
    num_rows = (num_cols + 2) // 3  # To make sure there are enough rows for the subplots
    fig, axs = plt.subplots(nrows=num_rows, ncols=3, figsize=(15, 5*num_rows))
    axs = axs.flatten()

    # Create a histogram for each integer variable with hue='Attrition'
    for i, var in enumerate(int_vars):
        top_categories = df[var].value_counts().nlargest(10).index
        filtered_df = df[df[var].notnull() & df[var].isin(top_categories)]
        sns.histplot(data=df, x=var, hue=target_variable, kde=True, ax=axs[i], stat="percent")
        axs[i].set_title(var)

        # Annotate the subplot with sample size
        sample_size = df.shape[0]
        axs[i].annotate(f'Sample Size = {sample_size}', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', va='center')

    # Remove any extra empty subplots if needed
    if num_cols < len(axs):
        for i in range(num_cols, len(axs)):
            fig.delaxes(axs[i])

    # Adjust spacing between subplots
    fig.tight_layout()
    plt.xticks(rotation=45)
    plot4_path = "static/multiclass_histplot.png"
    plt.savefig(plot4_path)
    plt.close(fig)

    #response 3
    def to_markdown(text):
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

    genai.configure(api_key=api)

    import PIL.Image

    img = PIL.Image.open("static/multiclass_barplot.png")
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content(img)
    response = model.generate_content([question + "As a marketing consulant, I want to understand consumer insighst based on the chart and the market context so I can use the key findings to formulate actionable insights", img])
    response.resolve()



    #response 4
    def to_markdown(text):
        text = text.replace('•', '  *')
        return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

    genai.configure(api_key=api)

    import PIL.Image

    img = PIL.Image.open("static/multiclass_histplot.png")
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response5 = model.generate_content(img)
    response5 = model.generate_content([question + "As a marketing consulant, I want to understand consumer insighst based on the chart and the market context so I can use the key findings to formulate actionable insights", img])
    response5.resolve()

    # Generate Google Gemini responses
    response3 = response.text
    response4 = response5.text


    # Create a dictionary to store the outputs
    outputs = {
        "barchart_visualization": plot1_path,
        "gemini_response1": response1,
        "histoplot_visualization": plot2_path,
        "gemini_response2": response2,
        "multiBarchart_visualization": plot3_path,
        "gemini_response3": response3,
        "multiHistoplot_visualization": plot4_path,
        "gemini_response4": response4
    }

    # Save the dictionary as a JSON file
    with open("output1.json", "w") as outfile:
        json.dump(outputs, outfile)


    
    # Function to handle encoding to latin1
    def safe_encode(text):
        try:
            return text.encode('latin1', errors='replace').decode('latin1')  # Replace invalid characters
        except Exception as e:
            return f"Error encoding text: {str(e)}"



    # Generate PDF with the results
    pdf = FPDF()
    pdf.set_font("Arial", size=12)

    # Single Countplot Barchart and response
    pdf.add_page()
    pdf.cell(200, 10, txt="Single Countplot Barchart", ln=True, align='C')
    pdf.image(plot1_path, x=10, y=30, w=190)
    pdf.add_page()
    pdf.cell(200, 10, txt="Single Countplot Barchart Google Gemini Response", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, safe_encode(response1))

    # Single Histplot and response
    pdf.add_page()
    pdf.cell(200, 10, txt="Single Histplot", ln=True, align='C')
    pdf.image(plot2_path, x=10, y=30, w=190)
    pdf.add_page()
    pdf.cell(200, 10, txt="Single Histplot Google Gemini Response", ln=True, align='C')
    pdf.ln(10)
    pdf.multi_cell(0, 10, safe_encode(response2))

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



    return templates.TemplateResponse("upload.html", {
        "request": request,
        "plot1_path": plot1_path,
        "response1": response1,
        "plot2_path": plot2_path,
        "response2": response2,
        "plot3_path": plot3_path,
        "response3": response3,
        "plot4_path": plot4_path,
        "response4": response4
    })




@app.get('/download_pdf2')
async def download_pdf2():
    pdf_output_path2 = 'static/analysis_report_complete.pdf'
    return FileResponse(pdf_output_path2, media_type='application/pdf', filename='analysis_report_complete.pdf')



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
