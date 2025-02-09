from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
import io

app = FastAPI()

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

interpreter = tflite.Interpreter(model_path="1.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to 128x128
    image = np.array(image).astype(np.float32)
    image = (image - 127.5) / 127.5  # Normalize to [-1, 1]
    img_data = np.zeros((1, 128, 128, 3), dtype=np.float32)
    img_data[0] = image
    return img_data

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    processed_image = preprocess_image(image)

    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get top 5 predictions
    top_5_indices = np.argsort(output_data[0])[-5:][::-1]
    top_5_predictions = [ 
        {
            "class": labels[i],
            "confidence": float(output_data[0][i])
        }
        for i in top_5_indices
    ]
    
    return {"predictions": top_5_predictions}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
