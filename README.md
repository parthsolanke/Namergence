# Namergence

Name checker to find the likely origin (language) of a name.

Namergence is a small web application built around a character-level PyTorch RNN. A FastAPI backend loads the trained model and serves predictions, and a Streamlit frontend provides a simple interface for entering a name and viewing ranked predictions.

## How it works

- Names are normalised to ASCII and encoded one character at a time as one-hot vectors over the allowed alphabet (letters plus space, `.`, `,`, `;`, `'`).
- A recurrent layer processes the character sequence and produces a score for each supported language.
- The model output is passed through `LogSoftmax`, so the served values are log-probabilities that the frontend converts to percentages.

Supported origins: Arabic, Chinese, Czech, Dutch, English, French, German, Greek, Indian, Irish, Italian, Japanese, Korean, Polish, Portuguese, Russian, Scottish, Spanish, Vietnamese.

## Project structure

```
data/names/          # One .txt file of names per language
src/app.py           # Streamlit frontend
src/server.py        # FastAPI backend exposing POST /predict
src/model.py         # RNN definition (nn.Module)
src/train.py         # Training and evaluation script
src/utils/loader.py  # Dataset loading, normalisation and tensor helpers
src/model/           # Trained weights: model.pth and labels.npy
src/notebook/        # Exploratory notebook
requirements.txt
```

## Prerequisites

- Python 3.8 or later
- The dependencies listed in `requirements.txt`

The pinned versions target Python 3.8. On newer Python releases you may need to relax the pins (for example for `torch` and `numpy`) so that prebuilt wheels are available.

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

## Usage

The backend resolves its weights from `./model` relative to the working directory, and imports `model` and `utils` as sibling modules. Start it from the `src` directory:

```bash
cd src
python server.py
```

This starts the FastAPI server on `http://127.0.0.1:8000`.

In a new terminal, launch the Streamlit frontend from the repository root:

```bash
streamlit run src/app.py
```

Open `http://localhost:8501`, enter a name and select **Predict** to see the predictions ranked by confidence. The frontend calls the backend at `http://127.0.0.1:8000/predict`, so the server must be running.

## API

### `POST /predict`

Request body:

```json
{ "name": "Ada" }
```

Response:

```json
{
  "name": "Ada",
  "label": "English",
  "confidences": [[-2.31, -5.02, "..."]],
  "categories": ["Arabic", "Chinese", "..."]
}
```

`confidences` is a list containing one list of per-category log-probabilities, aligned with `categories`. The frontend exponentiates these values to display percentages.

## Training

`src/train.py` trains the RNN from scratch and then reports accuracy over the dataset. The dataset is loaded by `src/utils/loader.py`, which resolves names from `../../data/names/*.txt` relative to the working directory, so run the trainer from a location where that path points at `data/names/` (or adjust the path in the loader).

Training uses SGD with a learning rate of `0.001` over a long loop, plots the loss curve, and finishes with an accuracy evaluation. The script does not write the trained weights; you will need to save `model.pth` and `labels.npy` to `src/model/` yourself if you want the server to use a newly trained model.

## Additional information

- The model was trained on the name files in `data/names/`, one file per language.
- Feel free to customise the model, dataset or frontend for your own use case.
