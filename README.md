# Namergence
Name checker to find origin of names 📛

This is a simple web application that uses a PyTorch model for classifying names into different categories (languages). It consists of a FastAPI backend serving predictions, and a Streamlit frontend for user interaction.

### Prerequisites

- Python 3.8 or later
- Dependencies listed in `requirements.txt`
- Go 1.20 or later for the Go client in `namergence-go/`

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

## Go client

A Go client for the `/predict` endpoint lives in `namergence-go/` (module `namergence`, standard library only). It works with the FastAPI server started above.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"namergence"
)

func main() {
	client := namergence.NewClient("http://127.0.0.1:8000")

	prediction, err := client.Predict(context.Background(), "Sofia")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(prediction.Label)
	for _, score := range prediction.Scores() {
		fmt.Printf("%s: %.2f%%\n", score.Category, score.Confidence*100)
	}
}
```

`Prediction` mirrors the API response (`name`, `label`, `confidences` and `categories`), and `Scores` returns the categories sorted from highest to lowest confidence. Run the client tests with `cd namergence-go && go test ./...`.

## Ping/pong connectivity

Namergence and ScalarFlow can confirm they are reachable from each other over HTTP. The Namergence FastAPI server listens on `http://127.0.0.1:8000` and answers `GET /ping` with:

```json
{"message": "pong", "service": "namergence"}
```

ScalarFlow serves `GET /pong` on `http://127.0.0.1:8001`, which answers with:

```json
{"message": "pong", "service": "scalarflow"}
```

Start both sides (from the ScalarFlow repository) to try the round trip:

```bash
# Terminal 1: Namergence API on 127.0.0.1:8000
python src/server.py

# Terminal 2: ScalarFlow pong server on 127.0.0.1:8001
cd scalarflow-go && go run ./cmd/pong
```

The Go client in `namergence-go/` can then call the peer's `/pong` endpoint with `NewPongClient`/`Pong`:

```go
client := namergence.NewPongClient("http://127.0.0.1:8001")
pong, err := client.Pong(context.Background())
if err != nil {
	log.Fatal(err)
}
fmt.Println(pong.Service, pong.Message)
```

An empty base URL uses the default `http://127.0.0.1:8001`. A Go 1.20 or later toolchain is required; run the client tests with `cd namergence-go && go test ./...`.

## Additional Information

- The model was trained on a dataset of names and their corresponding languages.
- Feel free to customize and improve the model as needed for your specific use case.
