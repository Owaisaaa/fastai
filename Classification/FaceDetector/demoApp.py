import gradio as gr
from fastai.vision.all import *

# Load your pre-trained model
learn_inf = load_learner('export.pkl')

def classify_image(img):
    """Classify an uploaded image."""
    pred, pred_idx, probs = learn_inf.predict(img)
    return {str(pred): float(probs[pred_idx])}

# Define the Gradio interface
image_input = gr.Image(type="pil", label="Upload Image")
label_output = gr.Label(label="Prediction")

demo = gr.Interface(
    fn=classify_image,
    inputs=image_input,
    outputs=label_output,
    title="Image Classifier",
    description="Upload an image to get a prediction from the model."
)

# Launch the Gradio app
demo.launch(share=True)
