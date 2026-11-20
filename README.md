# Namergence

Name-origin checker: given a name, guess which language/culture it most likely comes from.

Namergence is a character-level PyTorch RNN classifier served by a FastAPI backend, with a Streamlit frontend for entering names and browsing the ranked predictions. It recognizes 19 categories taken from the bundled `data/names/*.txt` datasets: Arabic, Chinese, Czech, Dutch, English, French, German, Greek, Indian, Irish, Italian, Japanese, Korean, Polish, Portuguese, Russian, Scottish, Spanish and Vietnamese.

## How it works

- `src/utils/loader.py` normalizes each name to ASCII, builds a one-hot tensor over the character vocabulary `string.ascii_letters + " .,;'"` (`N_LETTERS`), and can draw random training examples from the name files.
- `src/model.py` defines `RNN`, a simple recurrent classifier that concatenates the current character with the hidden state, projects it through two linear layers, and returns `LogSoftmax` scores over the categories.
- `src/train.py` trains the model with `NLLLoss` and SGD, prints progress, plots the loss with matplotlib, and reports accuracy.
- `src/server.py` loads the trained weights from `src/model/model.pth` and labels from `src/model/labels.npy`, then exposes `POST /predict`.
- `src/app.py` calls the API and displays every category with its confidence, sorted highest first.

## Requirements

Dependencies are pinned in `requirements.txt`:

```
streamlit==0.91.0
requests==2.26.0
torch==1.10.0
numpy==1.21.0
matplotlib==3.4.2
fastapi==0.68.0
```

Python 3.8+ is recommended. The server runs through `uvicorn`, which is not pinned in `requirements.txt`; install it separately if it is not already available:

```bash
pip install uvicorn
```

## Installation

```bash
git clone https://github.com/parthsolanke/Namergence.git
cd Namergence
pip install -r requirements.txt
```

## Running the app

The server resolves its weights as `./model`, so start it from the `src` directory:

```bash
cd src
python server.py
```

It listens on `http://127.0.0.1:8000`. Then, from another terminal, start the frontend from the repository root:

```bash
streamlit run src/app.py
```

Open `http://localhost:8501`, enter a name, and click **Predict** to see the categories ranked by confidence.

## API usage

`POST /predict` accepts JSON with a single `name` field and returns the name, the top label, the per-category probabilities, and the category list:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice"}'
```

```json
{
  "name": "Alice",
  "label": "English",
  "confidences": [[0.01, 0.02, "..."]],
  "categories": ["Arabic", "Chinese", "..."]
}
```

`confidences` is a `1 x 19` list in the same order as `categories`, so pair them by index to rank the results.

## Training

`src/train.py` loads the dataset through the relative path `../../data/names/*.txt`, so run it from `src/notebook`, which also keeps `model` and `utils` importable:

```bash
cd src/notebook
python ../train.py
```

This runs 100,000 iterations by default (SGD, learning rate 0.001, hidden size 128), shows a loss plot, and prints final accuracy. The exploration notebook is at `src/notebook/RNN.ipynb`.

## Dataset

The `data/names/` directory holds one text file per category, one name per line. Add new `.txt` files there to introduce additional origins; the category name is the file name without its extension.
