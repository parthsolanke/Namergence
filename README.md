# Namergence

Name checker to find the origin of names 📛

Namergence is a small web application that classifies a name into a language/origin category using a character-level PyTorch RNN. It is made of two parts:

- **Backend** — a FastAPI server that loads the trained model and serves predictions over HTTP.
- **Frontend** — a Streamlit app where a user types a name and sees the ranked category probabilities.

## Prerequisites

- Python 3.8 or later
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/parthsolanke/Namergence.git
   cd Namergence
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure a trained model is available. The server expects the following files inside `model/` at the project root:

   - `model/model.pth` — the saved RNN weights
   - `model/labels.npy` — the ordered list of category labels

   You can produce them by running the training script (see below).

## Usage

1. Start the FastAPI backend:

   ```bash
   python src/server.py
   ```

   The server runs on `http://127.0.0.1:8000`.

2. In a new terminal, start the Streamlit app:

   ```bash
   streamlit run src/app.py
   ```

   Visit `http://localhost:8501` in your browser, enter a name and click **Predict**.

## API

`POST /predict`

Request body:

```json
{ "name": "alice" }
```

Response:

```json
{
  "name": "alice",
  "label": "<top category>",
  "confidences": [[...]],
  "categories": ["..."]
}
```

`confidences` is the probability distribution over `categories` (aligned by index); the frontend sorts them highest first.

## Training the model

The model is trained on the name lists stored in `data/names/`, one file per language/category.

```bash
python src/train.py
```

The script trains the RNN with negative log-likelihood loss, periodically prints accuracy, plots the loss curve, and reports overall accuracy on the training data. Save the resulting state dict and label list to `model/model.pth` and `model/labels.npy` so the server can load them.

## Project structure

```
data/names/        # one text file of names per category
src/app.py         # Streamlit frontend
src/server.py      # FastAPI backend exposing POST /predict
src/train.py       # training script
src/model.py       # RNN definition
src/utils/loader.py# data loading and preprocessing helpers
requirements.txt
```

## Notes

- Input names are normalized to ASCII and lowercased before being converted to tensors; characters outside `a-z` (plus a few separators) are dropped.
- Feel free to retrain or extend the model with additional categories for your own use case.
