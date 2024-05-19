from modules import Chat
from modules import ingest_new_file
import os
from dotenv import load_dotenv

load_dotenv()

def ingest_a_folder(folder_path):
    print("Ingesting PDFs now this may take a while")
    for i in os.walk(folder_path):
        if i.endswith(".pdf"):
            ingest_new_file(i)
            print("Done with", i)
    print("Data stored successfully")
    
    


if __name__ == "__main__":
    # input a folder name
    folder = os.getenv("default_pdf_folder_path")
    ingest_a_folder(folder)
    chat = Chat()
    while True:
        question = input("Question: ")
        answer, pgs = chat.chat(question = question )
        print(answer)


    