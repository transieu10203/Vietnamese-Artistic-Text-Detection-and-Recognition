import cv2
import time
import requests
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from collections import OrderedDict,namedtuple
from fastapi import FastAPI, UploadFile, File
import numpy as np
import io
import uvicorn

from Processing import Processing

# Khởi tạo FastAPI
app = FastAPI()

# Định nghĩa endpoint để nhận hình ảnh và xử lý
@app.post("/process_image")
async def process_image(image: UploadFile = File(...)):
    # Read the image data
    image = await image.read()

    # Convert image data to OpenCV format
    image = np.frombuffer(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # Now you can process the image using cv2
    image=Processing(image,w='')
    success, encoded_image = cv2.imencode('.jpg', image)
    image = encoded_image.tobytes()
    
    # Tạo kết quả trả về dưới dạng JSON
    result = {
        'filename': image.filename,
        'resized_image_data': image
    }
    
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
