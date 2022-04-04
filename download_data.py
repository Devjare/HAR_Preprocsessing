# For zip extraction.
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile
import os 

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00344/Activity%20recognition%20exp.zip'
    
def download_and_unzip(url, extract_to='.'):
    isdir = os.path.isdir(extract_to) 
    if(isdir != True):
        print("Downloading dataset...")
        http_response = urlopen(url)
        zipfile = ZipFile(BytesIO(http_response.read()))
        zipfile.extractall(path=extract_to)

download_and_unzip(url, "./dataset")
