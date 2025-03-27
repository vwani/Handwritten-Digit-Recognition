from flask import Flask, request, render_template
from models import CNN
import torch
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if 'file' not in request.files:
        return "No file part"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"
    
    if file and file.content_type.startswith("image/"):
        image = Image.open(io.BytesIO(file.read()))

        image = image.convert("L").resize((28,28))
        image = torch.tensor(np.array(image) / 255., dtype=torch.float32)

        image = image.unsqueeze(0).unsqueeze(0).to(device)

        output = model(image)
        prediction = output.argmax(dim=1, keepdim=True).item()

        return render_template("index.html", prediction=prediction)

    else:
        return "Invalid file type! Only images are allowed"
    
if __name__=="__main__":
    app.run(debug=True)
