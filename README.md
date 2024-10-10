# Celebrity-Search
Introduction

The Celebrity Search Application is an interactive web application built using Streamlit that allows users to search for information about celebrities. By leveraging the power of LangChain and Hugging Face's FLAN-T5 model, the application can generate detailed responses about a celebrity's background, including their date of birth and significant historical events related to their birthdate.

Features

-User-friendly interface for searching celebrity information.
-Generates responses based on the input query about the celebrity's name.
-Sequential processing of queries to gather comprehensive information.
-Memory integration to maintain conversation history.

Technologies Used

-Python
-Streamlit
-LangChain
-Transformers (Hugging Face)
-Hugging Face API

Installation
To run this project, ensure you have Python installed on your machine and then follow these steps:

1.Clone repository 
git clone https://github.com/yourusername/celebrity-search-app.git
cd celebrity-search-app

2.Create Virtual Environment
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate

3.Install the required packages:
pip install -r requirements.txt

4.Set your Hugging Face API key:
export HUGGINGFACEHUB_API_TOKEN='your_hugging_face_api_key'

5.Run the Streamlit app:
streamlit run main.py

6.Open your browser and navigate to http://localhost:8501.



