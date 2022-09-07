import os
env_var = os.environ.get('env')
import torch
import time 
import numpy as np
import gradio as gr
from PIL import Image
from torchvision import transforms

device = 'cpu'
model = torch.load('model.pkl').to(device).eval()
transform = transforms.Resize(size=500)
labels = ['Cat', 'Dog']

def predict(image):
  start = time.time()
  with torch.no_grad():
    image = Image.fromarray(np.uint8(image)).convert('RGB')
    image = transform(image)
    image = np.array(image)
    image = torch.from_numpy(image).permute(2,0,1).float()
    image = image.unsqueeze(0)
    prediction = model(image.to(device))
    pred_idx = np.argmax(prediction.to(device))
    pred_label = "Cat" if pred_idx == 0 else "Dog"
    label = [l for l in labels if l!=pred_label]
    confidences = {pred_label: float(prediction[0][pred_idx])/100, label[len(label)-1]: 1-(float(prediction[0][pred_idx]))/100 }
    infer = time.time()-start  
  return confidences, infer

gr.Interface(fn=predict,
             inputs=gr.inputs.Image(shape=(512, 512)), 
             outputs=[gr.outputs.Label(num_top_classes=3), gr.outputs.Textbox('infer',label='Inference Time')],
             examples='1.jpg 2.jpg 3.jpg 4.jpg 5.jpg'.split(' ')).launch()