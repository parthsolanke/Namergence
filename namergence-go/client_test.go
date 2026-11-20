package namergence

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestPredict(t *testing.T) {
	received := make(chan PredictRequest, 1)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("method = %s, want POST", r.Method)
		}
		if r.URL.Path != "/predict" {
			t.Errorf("path = %s, want /predict", r.URL.Path)
		}

		var request PredictRequest
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Errorf("decode request: %v", err)
		}
		received <- request

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(Prediction{
			Name:        "Ada",
			Label:       "English",
			Confidences: [][]float64{{0.1, 0.7, 0.2}},
			Categories:  []string{"French", "English", "German"},
		})
	}))
	defer server.Close()

	client := NewClient(server.URL + "/")
	prediction, err := client.Predict(context.Background(), "Ada")
	if err != nil {
		t.Fatalf("Predict() error = %v", err)
	}
	if request := <-received; request.Name != "Ada" {
		t.Errorf("request name = %q, want Ada", request.Name)
	}
	if prediction.Label != "English" {
		t.Errorf("label = %q, want English", prediction.Label)
	}

	scores := prediction.Scores()
	if len(scores) != 3 {
		t.Fatalf("len(scores) = %d, want 3", len(scores))
	}
	if scores[0].Category != "English" || scores[0].Confidence != 0.7 {
		t.Errorf("top score = %+v, want {English 0.7}", scores[0])
	}
}

func TestPredictServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "bad name", http.StatusBadRequest)
	}))
	defer server.Close()

	client := NewClient(server.URL)
	if _, err := client.Predict(context.Background(), ""); err == nil {
		t.Fatal("Predict() error = nil, want server error")
	}
}
