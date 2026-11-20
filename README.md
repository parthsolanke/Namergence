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

### JavaScript client

A small ESM client with TypeScript typings lives in `js-client/` (see `js-client/README.md`). It calls `POST /predict` and returns the server response unchanged.

```js
import { predict, createClient } from "./js-client/index.js";

const result = await predict("Alice", { baseUrl: "http://127.0.0.1:8000" });
console.log(result.label, result.confidences);

const client = createClient({ baseUrl: "http://127.0.0.1:8000" });
console.log(await client.predict("Alice"));
```

Non-2xx responses throw a `NamergenceError` with `status` and `body`.

### Go client

A Go client lives in `go-client/` (see `go-client/README.md`). It calls `POST /predict` and returns the server response unchanged.

```go
result, err := namergence.Predict("Alice", &namergence.RequestOptions{
	BaseURL: "http://127.0.0.1:8000",
})
if err != nil {
	log.Fatal(err)
}
fmt.Println(result.Label, result.Confidences)
```

Non-2xx responses return a `*NamergenceError` with `Status` and `Body`.

## Additional Information

- The model was trained on a dataset of names and their corresponding languages.
- Feel free to customize and improve the model as needed for your specific use case.
