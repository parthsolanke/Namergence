from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
from utils import loader
from pydantic import BaseModel
import numpy as np
from model import RNN

app = FastAPI()

# Enable CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and labels
save_path = './model'
loaded_model = RNN(loader.N_LETTERS, 128, 19)
loaded_model.load_state_dict(torch.load(f"{save_path}/model.pth"))
loaded_model.eval()

loaded_labels = np.load(f'{save_path}/labels.npy', allow_pickle=True).tolist()

class InputName(BaseModel):
    name: str
    

def category_from_output(output):
    catagory_idx = torch.argmax(output).item()
    return loaded_labels[catagory_idx]


@app.post("/predict")
def predict(input_name: InputName):
    name = input_name.name
    with torch.no_grad():
        name_tensor = loader.line_to_tensor(name)
        hidden = loaded_model.init_hidden()
        for i in range(name_tensor.size()[0]):
            output, hidden = loaded_model(name_tensor[i], hidden)
        label = category_from_output(output)
        confidence = torch.exp(output).max().item()
    return {"label": label, "confidence": confidence}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
