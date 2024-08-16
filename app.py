import os
import tempfile
import shutil
import uuid
from flask import Flask, request, jsonify, abort, Response
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from langchain_community.document_loaders import TextLoader, UnstructuredPowerPointLoader, Docx2txtLoader, PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from werkzeug.exceptions import RequestEntityTooLarge
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_chroma import Chroma
import json
import re
import logging
import sys
import nltk


# Use /tmp or another writable directory
nltk_data_dir = '/tmp/nltk_data_dir'

# Ensure the directory exists
os.makedirs(nltk_data_dir, exist_ok=True)

# Download the NLTK data
nltk.download('punkt_tab', download_dir=nltk_data_dir)

# Set the path for NLTK to find the data
nltk.data.path.append(nltk_data_dir)


logging.basicConfig(level=logging.DEBUG,  # Change to DEBUG to get detailed logs
                    format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

models = "gpt-4o"

load_dotenv()

ALLOWED_EXTENSIONS = {'pdf', 'pptx', 'docx', 'txt', 'ppt'}
ALLOWED_IP_ADDRESSES = {"127.0.0.1","171.50.226.59"}  # Add allowed IP addresses here

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB


@app.route('/check-nltk-data', methods=['GET'])
def check_nltk_data():
    nltk_data_dir = '/tmp/nltk_data/tokenizers/punkt'
    logging.debug(f"Checking NLTK data directory: {nltk_data_dir}")
    try:
        # List contents of the directory
        files = os.listdir(nltk_data_dir)
        return jsonify({'files': files}), 200
    except FileNotFoundError:
        return jsonify({'Message': 'Directory not found'}), 404
    except Exception as e:
        return jsonify({'Message': str(e)}), 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(loader, file_path):
    logging.debug(f"Extracting text from file: {file_path} using loader: {loader}")
    try:
        doc = loader(file_path).load()
        logging.debug(f"Document loaded successfully: {file_path}")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50, add_start_index=True)
        all_split = text_splitter.split_documents(doc)
        logging.debug(f"Text split into {len(all_split)} chunks")
        return all_split, doc
    except Exception as e:
        logging.error(f"Error extracting text from {file_path}: {e}")
        return None, f"Error extracting text: {e}"
   
    
    

def handle_input_file(file):
    file_extension = os.path.splitext(file.filename)[1].lower()
    loaders = {
        '.pdf': PyPDFLoader,
        '.pptx': UnstructuredPowerPointLoader,
        '.ppt': UnstructuredPowerPointLoader,
        '.txt': TextLoader,
        '.docx': Docx2txtLoader
    }
    if file_extension in loaders:
        temp_dir = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))  # Unique temporary directory
        os.makedirs(temp_dir, exist_ok=True)  # Ensure directory exists
        file_path = os.path.join(temp_dir, secure_filename(file.filename))  # Ensure a secure file name
        file.save(file_path)
        try:
            all_splits, docs = extract_text(loaders[file_extension], file_path)
            combined_text = " ".join([split.page_content for split in all_splits])

            # Check for blank document
            if not combined_text.strip():
                return None, "The document is blank", temp_dir

            # Check for repetitive characters
            if re.search(r'(.)\1{10,}', combined_text):
                return None, "The document contains repetitive characters", temp_dir

            # Check for high numeric data
            numeric_count = sum(c.isdigit() for c in combined_text)
            if numeric_count / len(combined_text) > 0.5:
                return None, "The document contains too much numeric data", temp_dir
            return all_splits, docs, temp_dir
        except Exception as e:
            return None, f"Error extracting text: {e}", temp_dir
    return None, "Unsupported file format", None
def extract_number(key):
    match = re.search(r'\d+', key)
    return int(match.group()) if match else 0

# @app.before_request
# def limit_remote_addr():
#     if request.remote_addr not in ALLOWED_IP_ADDRESSES:
#         abort(403, description="Access denied: Your IP address is not allowed.")  # Forbidden

@app.errorhandler(RequestEntityTooLarge)
def handle_file_size_error(e):
    current_file_size = request.content_length
    max_file_size = app.config['MAX_CONTENT_LENGTH']
    return jsonify({
        'Message': 'File size exceeds the 20MB limit',
        'file_size': f'{current_file_size / (1024 * 1024):.2f} MB',
        'max_size': f'{max_file_size / (1024 * 1024):.2f} MB'
    }), 400

@app.route('/upload-document', methods=['POST'])
def upload_document():
    logging.debug("Starting upload_document function.")
    # Ensure OPENAI_API_KEY is set
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("The OPENAI_API_KEY environment variable is not set. Please set it in your environment.")
    logging.debug(f"OPENAI_API_KEY: {OPENAI_API_KEY}")

    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    if 'file' not in request.files:
        logging.error('No file part in request.')
        return jsonify({'Message': 'No file part'}), 400

    file = request.files['file']
    question = request.args.get('question', '')

    if file.filename == '':
        return jsonify({'Message': 'Invalid file'}), 400
    if not allowed_file(file.filename):
        return jsonify({'Message': 'File type not allowed'}), 400

    try:
        # Handle file input
        all_splits, docs, temp_dir = handle_input_file(file)
        if all_splits is None:
            if temp_dir:
                shutil.rmtree(temp_dir)
            return jsonify({'Message': docs}), 400

        # Combine text for context
        combined_text = " ".join([split.page_content for split in all_splits])

        # Initialize components for a fresh session
        selected_model = models
        llm = ChatOpenAI(model=selected_model)
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

        template_with_question_for_insights = """Analyze the content of the provided file and generate up to 15 insights. Each insight should include a summary (200 characters) and a detailed description (1500 characters). Output the response in JSON format without 'json' heading, with each insight structured as follows and {input}:

- Insight1:
  - Summary: Insight summary here
  - Description: Detailed insight description here
- Insight2:
  - Summary: Insight summary here
  - Description: Detailed insight description here
...
- Insight15:
  - Summary: Insight summary here
  - Description: Detailed insight description here

Instructions:
1. Base your response solely on the content within the provided context.
2. Do not introduce new elements or information not present in the context.
3. If there is no insight, generate the response without JSON header with the message: "Message": "There is no insight found. Please upload a different document."
4. Ensure the response does not mention ChatGPT or OpenAI.
5. The insights can be up to 15. For example, if there are only two insights available in the document, then generate two insights. If there are ten insights, generate ten insights. The insights should be in order: Insight1, Insight2......Insight15.
<context>
{context}
</context>
"""

        template_without_question_for_insights = """
Analyze the content of the provided file  and generate up to 15 insights. Each insight should include a summary (200 characters) and a detailed description (1500 characters). Output the response in JSON format without 'json' heading, with each insight structured as follows:

- Insight1:
  - Summary: Insight summary here
  - Description: Detailed insight description here
- Insight2:
  - Summary: Insight summary here
  - Description: Detailed insight description here
...
- Insight15:
  - Summary: Insight summary here
  - Description: Detailed insight description here

Instructions:
1. Base your response solely on the content within the provided context.
2. Do not introduce new elements or information not present in the context.
3. If there is no insight, generate the response without JSON header with the message: "Message": "There is no insight found. Please upload a different document."
4. Ensure the response does not mention ChatGPT or OpenAI.
5. The insights can be up to 15. For example, if there are only two insights available in the document, then generate two insights. If there are ten insights, generate ten insights. The insights should be in order: Insight1, Insight2......Insight15..

<context>
{context}
</context>
"""
        # Construct prompt template
        logging.debug('Constructing prompt template.')
        if question:
            custom_rag_prompt = PromptTemplate.from_template(template_with_question_for_insights)
        else:
            custom_rag_prompt = PromptTemplate.from_template(template_without_question_for_insights)
        
        document_chain = create_stuff_documents_chain(llm, custom_rag_prompt)
        retriever = vectorstore.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({"input": question, "context": combined_text})
        vectorstore.delete_collection()
        
        try:
            # Load the data from the JSON response
            data = json.loads(response["answer"])

            # Order the dictionary by its keys
            ordered_data = dict(sorted(data.items(), key=lambda item: extract_number(item[0])))

            # Convert the ordered dictionary back to JSON format
            ordered_data = json.dumps(ordered_data, indent=4)
            #ordered_json = json.loads(ordered_data)
        except json.JSONDecodeError:
            return jsonify({"Message": "Failed to decode JSON response from LLM"}), 500

        if not data:
            return jsonify({"Message": "There is no insight found. Please upload a different document"}), 200

        # Cleanup
        if temp_dir:
            shutil.rmtree(temp_dir)

        # Clear variables to avoid retention of previous data
        del llm
        del vectorstore
        del document_chain
        del retriever
        del retrieval_chain
        del response
        del all_splits
        del docs
        del combined_text
        del selected_model

        # Return response
        return Response(ordered_data, content_type='application/json'), 200
    except Exception as e:
        return jsonify({"Message": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0")
