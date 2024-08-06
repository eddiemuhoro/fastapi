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

# Define detailed information for each disease
disease_info = {
    'Anthracnose': {
        'description': 'Anthracnose is a group of fungal diseases that affect various plants.',
        'symptoms': 'Dark, sunken lesions on leaves, stems, flowers, or fruit.',
        'treatment': 'Use fungicides such as chlorothalonil, mancozeb, or thiophanate-methyl.',
        'prevention': 'Remove infected plant debris, avoid overhead watering, and ensure proper spacing for air circulation.',
        'ideal_conditions': 'Warm, wet conditions favor the development of anthracnose.'
    },
    'algal leaf': {
        'description': 'Algal leaf spots are caused by algae that thrive in moist conditions.',
        'symptoms': 'Greenish, circular spots on leaves.',
        'treatment': 'Apply copper-based fungicides and improve air circulation around plants.',
        'prevention': 'Reduce moisture on leaves, water at the base of plants, and ensure good drainage.',
        'ideal_conditions': 'High humidity and poor air circulation.'
    },
    'bird eye spot': {
        'description': 'Bird eye spot is a fungal disease characterized by small, circular spots with dark borders.',
        'symptoms': 'Small, circular spots with dark borders on leaves.',
        'treatment': 'Apply fungicides and remove affected leaves.',
        'prevention': 'Ensure good air circulation, avoid overhead watering, and maintain plant health.',
        'ideal_conditions': 'Warm and humid environments.'
    },
    'brown blight': {
        'description': 'Brown blight is a fungal disease that causes brown lesions on leaves.',
        'symptoms': 'Brown, irregularly shaped lesions on leaves.',
        'treatment': 'Use appropriate fungicides and remove infected plant parts.',
        'prevention': 'Keep the area clean of debris, avoid overhead irrigation, and improve air circulation.',
        'ideal_conditions': 'Moist conditions and moderate temperatures.'
    },
    'gray light': {
        'description': 'Gray light is a disease that causes grayish spots on leaves.',
        'symptoms': 'Grayish spots on leaves, often with a yellow halo.',
        'treatment': 'Apply fungicides and prune infected parts.',
        'prevention': 'Ensure proper plant spacing and avoid excessive moisture on leaves.',
        'ideal_conditions': 'Cool and wet conditions.'
    },
    'healthy': {
        'description': 'The plant is healthy with no signs of disease.',
        'symptoms': 'No symptoms present.',
        'treatment': 'No treatment needed.',
        'prevention': 'Maintain good cultural practices to keep plants healthy.',
        'ideal_conditions': 'Varies depending on the plant species.'
    },
    'red leaf spot': {
        'description': 'Red leaf spot is a fungal disease causing red spots on leaves.',
        'symptoms': 'Red, circular spots on leaves.',
        'treatment': 'Use fungicides and remove affected leaves.',
        'prevention': 'Improve air circulation and avoid wetting leaves.',
        'ideal_conditions': 'Humid conditions and poor air circulation.'
    },
    'white spot': {
        'description': 'White spot is a fungal disease causing white spots on leaves.',
        'symptoms': 'White spots on leaves, often with a powdery appearance.',
        'treatment': 'Apply fungicides and remove infected leaves.',
        'prevention': 'Ensure good air circulation and avoid overhead watering.',
        'ideal_conditions': 'Humid and shaded environments.'
    }
}

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
    
    # Get disease information
    info = disease_info[predicted_class]
    
    return JSONResponse(content={
        "predicted_class": predicted_class,
        "confidence": confidence,
        "description": info['description'],
        "symptoms": info['symptoms'],
        "treatment": info['treatment'],
        "prevention": info['prevention'],
        "ideal_conditions": info['ideal_conditions']
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
