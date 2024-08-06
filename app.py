from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import io
from PIL import Image

app = FastAPI()

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define class names
class_names = [
    'Anthracnose', 'algal leaf', 'bird eye spot', 'brown blight',
    'gray light', 'healthy', 'red leaf spot', 'white spot'
]

# Absolute path to the model file
model_path = 'tea_modelv3.h5'

# Load the entire model
model = tf.keras.models.load_model(model_path)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    # Preprocess the image
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, 0)  # Create batch axis
    
    # Predict
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    # Get the highest confidence label
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)
    
    return JSONResponse(content={
        "predicted_class": predicted_class,
        "confidence": confidence
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
