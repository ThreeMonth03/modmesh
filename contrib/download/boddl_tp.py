import json
import os
import ssl
import urllib.request
import zipfile

class BoddLTP:
    url = "https://techport.nasa.gov/api/file/presignedUrl/380503"
    download_dir = "./download"
    filename = "DDL-F1_Dataset-20201013.zip"

    @classmethod
    def download(cls):
        response = urllib.request.urlopen(cls.url)
        presigned_url = json.loads(response.read())["presignedUrl"]
        os.makedirs(cls.download_dir, exist_ok=True)
        urllib.request.urlretrieve(presigned_url, f"{cls.download_dir}/{cls.filename}")

    @classmethod
    def extract(cls):
        with zipfile.ZipFile(f"{cls.download_dir}/{cls.filename}", "r") as file:
            file.extractall(cls.download_dir)

def main():
    ssl._create_default_https_context = ssl._create_stdlib_context
    BoddLTP.download()
    BoddLTP.extract()

if __name__ == "__main__":
    main()