# gradio_app.py
import gradio as gr
from keras.models import load_model
import numpy as np
import cv2

# Load the model
model = load_model('app/mnist_model.h5')

def predict_image(image):
    # Convert the PIL image to a numpy array
    img = np.array(image.convert('L'))
    
    # Resize the image to 28x28 pixels (same size as the MNIST images)
    img = cv2.resize(img, (28, 28))
    
    # Reshape the image to match the input shape for the model
    img = img.reshape(1, 28, 28)
    
    # Normalize the pixel values to the range [0, 1]
    img = img / 255.0
    
    # Make a prediction using the trained model
    prediction = model.predict(img)
    
    # Get the predicted label (the digit)
    predicted_label = np.argmax(prediction, axis=1)[0]
    
    return predicted_label

# Define the Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.inputs.Image(shape=(28, 28), image_mode='L', invert_colors=False, source="upload"),
    outputs="label",
    live=True,
    title="Handwriting Analysis"
)

if __name__ == "__main__":
    iface.launch()
