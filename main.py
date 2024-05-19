from src.modules import Chat
from src.modules import ingest_new_file
import os
from dotenv import load_dotenv

load_dotenv()

def ingest_a_folder(folder_path):
    print("Ingesting PDFs now this may take a while")
    all_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    print(all_files)
    for i in all_files:
        print(i)
        if i.endswith(".pdf"):
            ingest_new_file(os.path.join(folder, i))
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


    