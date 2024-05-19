
# # Silence all warnings
# warnings.filterwarnings("ignore")
from openai import OpenAI
import os
import requests
from io import BytesIO
from dotenv import load_dotenv
import os
import re

load_dotenv()

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=os.getenv("OPENAI_API_KEY")
)

class Utils:
    def __init__(self) -> None:
        pass
    def validate_intern(self, api_token):
        url = os.getenv("Validate_intern_api_url")+ api_token

        try:
            response = requests.get(url)
            data = response.json()

            if not data["error"] and data["message"] == "success":
                user_check = data.get("user_check", {})
                is_intern = user_check.get("is_intern", 0)
                return bool(is_intern)
            else:
                print(f"Error: {data['message']}")
                return False

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return False


    def call_openai(self, openai_params):

            try:
                response = client.chat.completions.create(**openai_params)
                return response.choices[0].message.content.strip()
            except Exception as e:
                return "Exception: " + str(e)

    def extract_file_id(self, url):
        try: 
            file_id = url.split("/file/d/")[1].split("/view")[0] # to get the code of google drive
            return file_id
        except IndexError:
            raise Exception("Invalid Google Drive link.")

    def download_file_from_google_drive(self, file_id):
        URL = os.getenv("default_drive_url") + file_id # downloading the pdf from gdrive
        response = requests.get(URL, stream=True)

        if response.status_code == 200:
            return BytesIO(response.content) # loading the file in memory
        else:
            raise Exception("Failed to download the file from Google Drive.")
    