from base64 import b64encode

import googleapiclient.discovery
from oauth2client.client import GoogleCredentials

# Settings
IMAGE_FILE = "text.png"
CREDENTIALS_FILE = "credentials-googleapi.json"

# Connect to Google cloud service
credentials = GoogleCredentials.from_stream(CREDENTIALS_FILE)
service = googleapiclient.discovery.build('vision', 'v1', credentials=credentials)

# read file and covert to base64 encoding
with open(IMAGE_FILE, "rb") as f:
    image_data = f.read()
    encoded_image_data = b64encode(image_data).decode('UTF-8')

# Create the request object for Google Vision API
batch_request = [{
    'image':{'content': encoded_image_data}, 
    'features':[{'type': 'TEXT_DETECTION'}] # (asking for extracted text from the image back)
}]
request = service.images().annotate(body={'requests': batch_request})

# Send request to Google
response = request.execute()
if 'error' in response:
    raise RuntimeError(response['error'])

# Print results
extracted_texts = response['responses'][0]['textAnnotations']
# Print the first piece of text found
extracted_text = extracted_texts[0]
print(extracted_text['description'])
# Print the pixel location where the text was detected
print(extracted_text['boundingPoly'])

