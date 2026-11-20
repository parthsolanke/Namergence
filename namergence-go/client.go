// Package namergence provides a Go client for the Namergence HTTP API.
//
// Start the Python server with "python src/server.py", then use the client to
// send names to the POST /predict endpoint.
package namergence

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
)

// DefaultBaseURL is the address used by src/server.py when it starts.
const DefaultBaseURL = "http://127.0.0.1:8000"

// Client talks to a running Namergence server.
type Client struct {
	BaseURL    string
	HTTPClient *http.Client
}

// PredictRequest is the JSON body expected by POST /predict.
type PredictRequest struct {
	Name string `json:"name"`
}

// Prediction is the JSON response returned by POST /predict. Confidences is a
// single row whose entries line up with Categories.
type Prediction struct {
	Name        string      `json:"name"`
	Label       string      `json:"label"`
	Confidences [][]float64 `json:"confidences"`
	Categories  []string    `json:"categories"`
}

// Score is one category and its confidence value.
type Score struct {
	Category   string
	Confidence float64
}

// NewClient creates a client for baseURL. An empty baseURL uses DefaultBaseURL.
func NewClient(baseURL string) *Client {
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	return &Client{
		BaseURL: baseURL,
		HTTPClient: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

// Predict sends name to the server and returns its origin prediction.
func (c *Client) Predict(ctx context.Context, name string) (*Prediction, error) {
	body, err := json.Marshal(PredictRequest{Name: name})
	if err != nil {
		return nil, fmt.Errorf("namergence: encode request: %w", err)
	}

	baseURL := c.BaseURL
	if baseURL == "" {
		baseURL = DefaultBaseURL
	}
	url := strings.TrimRight(baseURL, "/") + "/predict"

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("namergence: create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	httpClient := c.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("namergence: send request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		message, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("namergence: server returned %s: %s", resp.Status, strings.TrimSpace(string(message)))
	}

	var prediction Prediction
	if err := json.NewDecoder(resp.Body).Decode(&prediction); err != nil {
		return nil, fmt.Errorf("namergence: decode response: %w", err)
	}
	return &prediction, nil
}

// Scores returns the categories sorted from highest to lowest confidence.
func (p *Prediction) Scores() []Score {
	if len(p.Categories) == 0 || len(p.Confidences) == 0 {
		return nil
	}

	row := p.Confidences[0]
	n := len(p.Categories)
	if len(row) < n {
		n = len(row)
	}

	scores := make([]Score, n)
	for i := 0; i < n; i++ {
		scores[i] = Score{
			Category:   p.Categories[i],
			Confidence: row[i],
		}
	}

	sort.SliceStable(scores, func(i, j int) bool {
		return scores[i].Confidence > scores[j].Confidence
	})
	return scores
}
