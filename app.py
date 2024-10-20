from flask import Flask, request
import requests
from requests.auth import HTTPBasicAuth
from pydub import AudioSegment
import os
from langchain.vectorstores import Chroma
from twilio.rest import Client
from reteriver import create_conversation,generate_question,generate_questionaire,grade_answer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from requests.auth import HTTPBasicAuth
from twilio.twiml.messaging_response import MessagingResponse
from constants import TWILIO_SID, TWILIO_TOKEN, FROM,DB_DIR,ENV_KEY
from PIL import Image
import pytesseract
import re
import uuid
# Initialize Twilio client
account_sid = TWILIO_SID
auth_token = TWILIO_TOKEN
client = Client(account_sid, auth_token)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

import json
current_User_option_Stage=""

learning_questions=[]

learning_responses=[]

test_questionaries=[]
qa=create_conversation()
# next_to_answer=""
var_name="next_to_answer"
globals()[var_name]= ""
app = Flask(__name__)

# #send_to_this_number='whatsapp:+919711456735'
# send_to_this_number='whatsapp:+919398577967'

import string
def process_string(input_str):
    # Remove newlines and specified punctuation except space and period
    input_str = input_str.replace('\n', ' ')  # Replace newlines with space
    input_str = ''.join([char for char in input_str if char not in string.punctuation or char in ['.', ' ']])

    # If the string exceeds 1500 characters, truncate it
    if len(input_str) > 1500:
        input_str = input_str[:1500]
    
    return input_str

@app.route('/', methods=['GET', 'POST'])
def home():
    return 'OK', 200

def respond(message):
    response = MessagingResponse()
    response.message(message)
    return str(response)

def separate_questions_and_answers(text):
    # Regular expression to match questions and their answers
    pattern = r'(\d+\..+?)(?=Answer:-|Answer:~|‘Answer:-)(.+?)(?=\n\d+\.|\Z)'
    matches = re.findall(pattern, text, re.DOTALL)
    
    # Create a dictionary to hold the results
    qa_dict = {}
    
    for match in matches:
        question = match[0].strip()
        answer = match[1].strip().replace("Answer:-", "").replace("Answer:~", "").replace("‘", "").strip()
        qa_dict[question] = answer
    
    return qa_dict


def extract_score(text):
    # Define the regex pattern
    pattern = r'(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)'
    
    # Search for the pattern in the text
    match = re.search(pattern, text)

    if match:
        score = float(match.group(1))  # Extract the score as a float
        total = float(match.group(2))   # Extract the total possible points as a float
        return [score, total]
    else:
        return None  # Return None if no score is found
    

def calculate_score(score_list):
    total_score = 0
    
    # Iterate through the list, starting from index 1
    for index, score in enumerate(score_list[1:], start=1):
        if score is None:
            if index in (1, 2):  # Index 1 (second element)
                total_score += 2
            elif index in (3, 4):  # Index 3 (fourth element)
                total_score += 5
            elif index in (5, 6):  # Index 5 (sixth element)
                total_score += 10
        else:
            total_score += score[0]  # Add the first value of the sublist
    
    return total_score



@app.route('/twilio', methods=['POST'])
def twilio():
    query = request.form['Body']
    sender_id = request.form['From']
    sender = request.form.get('From')
    message = request.form.get('Body')
    media_url = request.form.get('MediaUrl0') 
    print(sender_id, query)
    global next_to_answer
    print("next_to_answer:",next_to_answer)
    print(media_url)
    print(f'{sender} sent {message}')
    if media_url:
        random_uuid = uuid.uuid4()
        print("entered 70 ")
        #r = requests.get(media_url)
        r = requests.get(media_url, auth=HTTPBasicAuth(TWILIO_SID, TWILIO_TOKEN))
        print("r value:",r)
        content_type = r.headers['Content-Type']
        print("contetn type:",content_type)
        username = sender.split(':')[1]  # remove the whatsapp: prefix from the number
        print("username:",username)
        if content_type == 'image/jpeg':
            filename = f'uploads/{str(random_uuid)}.jpg'
        elif content_type == 'image/png':
            filename = f'uploads/{str(random_uuid)}.png'
        elif content_type == 'image/gif':
            filename = f'uploads/{str(random_uuid)}.gif'
        else:
            filename = None
        if filename:
            print("entered 83")
            print("filename:",filename)
            if not os.path.exists(f'uploads/'):
                os.mkdir(f'uploads/')
            with open(filename, 'wb') as f:
                f.write(r.content)
            message = client.messages.create(
            from_='whatsapp:+14155238886',
            content_sid='HXc9dcda84ea4b585585c3340d4d84f9f2',
            to=send_to_this_number,

            )
            return 'OK', 200
    elif query.lower() in ["hi", "hello","how can you help me ?"]:
        print("Welcome Msg GOt from the User !")
        # Creating the message
        message = client.messages.create(
        from_='whatsapp:+14155238886',
        content_sid='HX1461046430b84d6ccac397e7faaa927b',
        to=send_to_this_number
        )

        print(f"Message sent with SID: {message.sid}")
        return 'OK', 200
    
    elif query.lower()=="ask me anything." or query.lower()=="ask another question":
        print("user selected the Lerning option")
        current_User_option_Stage="learning"
        print("user stage:",current_User_option_Stage)
        ques=generate_question()
        print("question in flask:",ques)
        learning_questions.append(ques)
        message = client.messages.create(
        from_='whatsapp:+14155238886',
        content_sid='HXf37ff7fb88fa883a382a7b4b3c3ca669',
        content_variables=f'{{"1":"{ques}"}}',
        to=send_to_this_number,

        )
        print(f"Message sent with SID: {message.sid}")
        return 'OK', 200
    elif query.lower()=="let me ask you":
        print("User selected the let me ask you  option")
        current_User_option_Stage="clarifying"
        print("user stage:",current_User_option_Stage)
        learning_questions.append(query)
        var_name="next_to_answer"
        if var_name not in globals():
            globals()[var_name]= True
        next_to_answer=True
        message = client.messages.create(
            from_='whatsapp:+14155238886',
            content_sid='HX3375c62a0708ae634992dfb972faef66',
            to=send_to_this_number,
        )
        print("Message sent successfully:", message.sid)   
        return 'OK', 200


    elif query.lower()=="take a test" or query.lower()=="take a test." or query.lower()=="go for test":
        print("User selected the Test option")
        current_User_option_Stage="test"
        print("user stage:",current_User_option_Stage)
        questionarire_list=generate_questionaire()
        print("questionarie list:",questionarire_list)
        test_questionaries.append(questionarire_list)
        content_variables = {
        str(i + 1): question for i, question in enumerate(questionarire_list)
        }
        # Create a JSON-like string for Twilio content variables
        content_variables_str = f'{{' + ', '.join([f'"{key}":"{value}"' for key, value in content_variables.items()]) + '}}'
        # Now you can use content_variables_str in your Twilio message
        print("contente vairbales string:",content_variables_str[:-1])
        message = client.messages.create(
        from_='whatsapp:+14155238886',
        content_sid='HX03c3dafaddf37c2d89b76ff2ae578564',
        content_variables=content_variables_str[:-1],
        to=send_to_this_number)
        print(f"Message sent with SID: {message.sid}")
        return 'OK', 200
    elif query.lower()=="evaluate":
        print("evalaute started !!!!")
        image_dir = "uploads"
        # Variable to store all extracted text
        extracted_text = ""

        # Loop through all files in the directory
        for filename in os.listdir(image_dir):
            # Check if the file is an image (jpeg, png, etc.)
            if filename.endswith((".png", ".jpg", ".jpeg", ".gif", ".tiff")):
                file_path = os.path.join(image_dir, filename)
                
                # Open the image
                img = Image.open(file_path)
                print("189:",file_path,img)
                # Perform OCR on the image
                text = pytesseract.image_to_string(img)
                print("************************************")
                print("text:",text)
                # Append the extracted text to the variable
                extracted_text += f"\n{text}\n\n"
        formated_text=separate_questions_and_answers(extracted_text)
        print("questions:",formated_text)
        fe_questions = []
        fe_answers = []
        fel_actual_res=[]
        marks=[2,2,5,5,10,10]
        scores_eval=[]
        for questr, answ in formated_text.items():
            res = qa({
                    'question': questr,
                    'chat_history': {}
                })
            fel_actual_res.append(process_string(res["answer"]))
            fe_questions.append(questr)
            fe_answers.append(answ)
        print("evalaution satrted !")
        for klj in range(len(fe_questions)):
            eval_res=grade_answer(fe_questions[klj], fe_answers[klj], fel_actual_res[klj], marks[klj])
            print("response:",eval_res)
            sc=extract_score(eval_res)
            scores_eval.append(sc)
        print("Seeting things for Evaluation")
        print("************************************************************************")
        print("Questions:",fe_questions)
        print("fe_answers:",fe_answers)
        print("fel_actual_res:",fel_actual_res)
        print("marks:",marks)
        print("scores :",scores_eval)
        print("Evalauted scroe:",calculate_score(scores_eval))
        message = client.messages.create(
            from_='whatsapp:+14155238886',
            content_sid='HX1f11ff2555090036ab6ccb1db623b882',
            content_variables=f'{{"1":"{calculate_score(scores_eval)}"}}',  # Correctly formatted JSON string
            to=send_to_this_number,
        )

        return 'OK', 200
    elif query.lower() == "show answer" or next_to_answer == True:
        print("User selected the show answer response:")
        if next_to_answer == True:
            learning_questions.append(query)
        print("learning_questions:", learning_questions)
        if len(learning_questions) > 0:
            ques_to_gen_res = learning_questions[-1]
            print("ques_to_gen_res:", ques_to_gen_res)
            try:
                res = qa({
                    'question': ques_to_gen_res,
                    'chat_history': {}
                })
                print("responsse:", res)
                dupl = res['answer']
                print("response:", dupl)
                learning_responses.append(dupl)
                dupl=process_string(dupl)
                print("*********************************************************")
                # Send message via Twilio
                print("dupl:", type(dupl), dupl)
                # print("content varaibles:",content_variables_json)
                message = client.messages.create(
                    from_='whatsapp:+14155238886',
                    content_sid='HXca78d96d7c1b0abd2acccc67b18124d0',
                    content_variables=f'{{"1":"{dupl}"}}',  # Correctly formatted JSON string
                    to=send_to_this_number,
                )
                print("Message sent successfully:", message.sid)
                if next_to_answer == True:
                    next_to_answer = False
            except Exception as e:
                print("An error occurred while generating the response or sending the message:", str(e))
        return 'OK', 200

        pass
        return 'OK', 200
        

if __name__ == '__main__':
    app.run(debug=True)
