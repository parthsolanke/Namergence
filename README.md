# Namergence

Name checker to find the likely origin of names 📛

Namergence is a small web application that uses a PyTorch recurrent neural network to classify a name into one of several language categories. It consists of a FastAPI backend that serves predictions and a Streamlit frontend for user interaction.

## Project structure

```
src/
  app.py          Streamlit frontend
  server.py       FastAPI backend exposing POST /predict
  train.py        Training script for the RNN model
  model.py        PyTorch RNN definition
  utils/loader.py Data loading and preprocessing helpers
data/names/       Name datasets grouped by language
```

## Prerequisites

- Python 3.8 or later
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository and enter the project directory:

   ```bash
   git clone https://github.com/parthsolanke/Namergence.git
   cd Namergence
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Training the model

The backend loads its weights from `./model/model.pth` and the label mapping from `./model/labels.npy`, so train the model first:

```bash
python src/train.py
```

The script reads the datasets in `data/names/`, trains the RNN, plots the loss curve, and prints an evaluation accuracy. Make sure the `./model` directory exists and save the trained weights (`model.pth`) and labels (`labels.npy`) there before starting the server.

## Usage

1. Start the FastAPI backend from the project root (so the relative `./model` path resolves):

   ```bash
   python src/server.py
   ```

   This runs the API on `http://127.0.0.1:8000`.

2. In a new terminal, start the Streamlit app:

   ```bash
   streamlit run src/app.py
   ```

   Visit `http://localhost:8501` in your browser, enter a name, and click **Predict** to see the categories ranked by confidence.

## API

### `POST /predict`

Request body:

```json
{ "name": "Alice" }
```

Response returns the predicted `label`, the full list of `categories`, and the per-category `confidences`, for example:

```json
{
  "name": "Alice",
  "label": "English",
  "confidences": [[0.01, 0.02, 0.97]],
  "categories": ["Chinese", "French", "English"]
}
```

## Additional information

- The model was trained on a dataset of names and their corresponding languages.
- Feel free to customize and improve the model for your own use case.
