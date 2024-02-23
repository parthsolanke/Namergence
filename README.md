# Namergence
Name checker to find origin of names 📛

This is a simple web application that uses a PyTorch model for classifying names into different categories (languages). It consists of a FastAPI backend serving predictions, and a Streamlit frontend for user interaction.

### Prerequisites

- Python 3.8 or later
- Dependencies listed in `requirements.txt`

### Installation
1. Clone the repository:

   ```bash
   https://github.com/parthsolanke/Namergence.git
   cd Namergence
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Start the FastAPI backend:

   ```bash
   python src/server.py
   ```

   This will run the FastAPI server on `http://127.0.0.1:8000`.

2. In a new terminal, run the Streamlit app:

   ```bash
   streamlit run src/app.py
   ```

   Visit `http://localhost:8501` in your web browser to use the app.

## Additional Information

- The model was trained on a dataset of names and their corresponding languages.
- Feel free to customize and improve the model as needed for your specific use case.
