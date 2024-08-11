import requests
import os

class FileSaver:

    def __init__(self) -> None:
        pass

    def download_pdf(self,url):
        response = requests.get(url)
        if response.status_code == 200:
            file_path = os.path.join("uploaded_files", "downloaded.pdf")
            with open(file_path, "wb") as f:
                f.write(response.content)
            return file_path
        else:
            return None