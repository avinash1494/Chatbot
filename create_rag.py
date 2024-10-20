import re
import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from PyPDF2 import PdfReader
from constants import TWILIO_SID, TWILIO_TOKEN, FROM, DB_DIR, OUTPUT_DIR

# Function to clean text by removing unwanted symbols, extra spaces, etc.
def clean_text(text: str) -> str:
    # Remove unwanted symbols/patterns
    text = re.sub(r'[^\w\s,.?!:;\'\"()-]', '', text)  # Keep basic punctuation and alphanumeric characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = text.strip()  # Trim leading and trailing spaces
    return text

def create_index(file_path: str) -> None:
    all_texts=[]
    for file_name in file_path:
        reader = PdfReader("input_data_path/"+file_name)
        text = ''
        
        # Extract and clean text from each page
        for page in reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += clean_text(extracted_text)  # Clean the extracted text

        # Save the cleaned text to a file
        with open(f'{OUTPUT_DIR}/{file_name}.txt', 'w') as file:
            file.write(text)

        loader = DirectoryLoader(
            OUTPUT_DIR,
            glob='**/*.txt',
            loader_cls=TextLoader
        )

        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=25,
        length_function=len
        )

        texts = text_splitter.split_documents(documents)

        all_texts.extend(texts)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    persist_directory = DB_DIR

    vectordb = Chroma.from_documents(
        documents=all_texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    vectordb.persist()

def inital_run():
    print("Welcome to the RAG Chatbot over your Files")

    input_dir="/home/avinash/Meta_Hackthon/Personalised_Knowledge_Bot/input_data_path/"
    
    input_files=os.listdir(input_dir)

    print("input data files:",input_files)

    #file_name = input("Enter Filename/ File path that you want to enable bot service:")

    print("File entered By the User:", input_files)

    create_index(input_files)

    print("Vector store creation is completed !!!!")
    print("Now going to build the retriever !!!")

inital_run()