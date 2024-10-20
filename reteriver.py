import os
import random
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
#from llama_model import Load_llama_model
import requests
from constants import OPENAI_API_KEY,TWILIO_SID,TWILIO_TOKEN,FROM,DB_DIR,OUTPUT_DIR,ENV_KEY




def grade_answer(question, answer, actual_answer, max_marks=10):
    # Prompt template
    prompt_re = f"""
    You are a grading assistant. Your task is to evaluate the following answer based on the actual answer provided.

    Question: {question}
    Given Answer: {answer}
    Actual Answer: {actual_answer}

    Please provide a score from 0 to {max_marks} based on the following criteria:
    - Relevance to the question
    - Accuracy of the information
    - Completeness of the answer
    """
    response=llm.invoke(prompt_re)
    return response.content


import string 
def  process_string_sec(input_str):
    # Remove newlines and specified punctuation except space and period
    input_str = input_str.replace('\n', ' ')  # Replace newlines with space
    input_str = ''.join([char for char in input_str if char not in string.punctuation or char in ['.', ' ','?']])

    # If the string exceeds 1500 characters, truncate it
    if len(input_str) > 1500:
        input_str = input_str[:1500]
    
    return input_str

def generate_question_based_on_context_for_questionarie(context, marks):
    if marks == 2:
        prompt_template = (
            f'''Based on the following context: "{context}", generate a single 2-mark question that tests understanding of the content. 
            Generate only the question. Do not generate any unnecessary content.'''
        )
    elif marks == 5:
        prompt_template = (
            f'''Based on the following context: "{context}", generate a single 5-mark question that tests understanding of the content. 
            Generate only the question. Do not generate any unnecessary content.'''
        )
    elif marks == 10:
        prompt_template = (
            f'''Based on the following context: "{context}", generate a single 10-mark question that tests understanding of the content. 
            Generate only the question. Do not generate any unnecessary content.'''
        )
    else:
        raise ValueError("Marks should be one of the following: 2, 5, or 10.")
    
    print("prompt_template:", prompt_template)
    response = llm.invoke(prompt_template)
    return process_string_sec(response.content)


def generate_questionaire():
    questioniare_ques=[]

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

    texts=[jk.page_content for jk in texts]

    random_index = random.randint(3, len(texts) - 3)

    concatenated_text = " ".join(texts[random_index:random_index+3])

    questioniare_ques.append(generate_question_based_on_context_for_questionarie(concatenated_text, 2))

    questioniare_ques

    random_index = random.randint(3, len(texts) - 3)

    concatenated_text = " ".join(texts[random_index:random_index+3])

    questioniare_ques.append(generate_question_based_on_context_for_questionarie(concatenated_text, 2))

    random_index = random.randint(3, len(texts) - 3)

    concatenated_text = " ".join(texts[random_index:random_index+3])

    questioniare_ques.append(generate_question_based_on_context_for_questionarie(concatenated_text, 5))

    random_index = random.randint(3, len(texts) - 3)

    concatenated_text = " ".join(texts[random_index:random_index+3])

    questioniare_ques.append(generate_question_based_on_context_for_questionarie(concatenated_text, 5))

    random_index = random.randint(3, len(texts) - 3)

    concatenated_text = " ".join(texts[random_index:random_index+3])

    questioniare_ques.append(generate_question_based_on_context_for_questionarie(concatenated_text, 10))

    random_index = random.randint(3, len(texts) - 3)

    concatenated_text = " ".join(texts[random_index:random_index+3])

    questioniare_ques.append(generate_question_based_on_context_for_questionarie(concatenated_text, 10))

    return questioniare_ques



def generate_question_based_on_context(context):
    prompt_template = (
        f'''Based on the following context: "{context}", generate a single question that could test understanding of the content.Generate only Question . 
        Dont genrate any unneccesary content'''
    )
    print("prompt_template:",prompt_template)
    response=llm.invoke(prompt_template)
    return response.content


def generate_question():

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

    texts=[jk.page_content for jk in texts]

    random_index = random.randint(3, len(texts) - 3)

    concatenated_text = " ".join(texts[random_index:random_index+3])

    print("random texts:",concatenated_text)

    single_question=generate_question_based_on_context(concatenated_text)

    print("single Question:",single_question)

    return single_question












os.environ["GROQ_API_KEY"] = ENV_KEY

llm = ChatGroq(temperature=0.4, model_name="llama3-8b-8192")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_conversation() -> ConversationalRetrievalChain:

    persist_directory = DB_DIR

    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=False
    )
    

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        get_chat_history=lambda h: h,
        verbose=True

    )
    # Retrieve relevant documents from the vector store
    

    return qa
    




























































































































