package namergence

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// DefaultPongBaseURL points at the ScalarFlow pong server, whose GET /pong
// endpoint is used to check that the two services can reach each other.
const DefaultPongBaseURL = "http://127.0.0.1:8001"

// PingResponse is the JSON body returned by the /ping and /pong endpoints.
type PingResponse struct {
	Message string `json:"message"`
	Service string `json:"service"`
}

// PongClient talks to a peer service's /pong endpoint.
type PongClient struct {
	BaseURL    string
	HTTPClient *http.Client
}

// NewPongClient creates a client for baseURL. An empty baseURL uses
// DefaultPongBaseURL.
func NewPongClient(baseURL string) *PongClient {
	if baseURL == "" {
		baseURL = DefaultPongBaseURL
	}
	return &PongClient{
		BaseURL: baseURL,
		HTTPClient: &http.Client{
			Timeout: 10 * time.Second,
		},
	}
}

// Pong calls GET /pong on the peer and returns its response.
func (c *PongClient) Pong(ctx context.Context) (*PingResponse, error) {
	baseURL := c.BaseURL
	if baseURL == "" {
		baseURL = DefaultPongBaseURL
	}
	url := strings.TrimRight(baseURL, "/") + "/pong"

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("namergence: create pong request: %w", err)
	}
	req.Header.Set("Accept", "application/json")

	httpClient := c.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("namergence: send pong request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		message, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("namergence: peer returned %s: %s", resp.Status, strings.TrimSpace(string(message)))
	}

	var pong PingResponse
	if err := json.NewDecoder(resp.Body).Decode(&pong); err != nil {
		return nil, fmt.Errorf("namergence: decode pong response: %w", err)
	}
	return &pong, nil
}
