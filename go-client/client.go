package namergence

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

const DEFAULT_BASE_URL = "http://127.0.0.1:8000"

const DefaultBaseURL = DEFAULT_BASE_URL

type PredictionResult struct {
	Name        string    `json:"name"`
	Label       string    `json:"label"`
	Confidences []float64 `json:"confidences"`
	Categories  []string  `json:"categories"`
}

type RequestOptions struct {
	BaseURL string
	Timeout time.Duration
	Headers map[string]string
	Client  *http.Client
}

type NamergenceError struct {
	Message string
	Status  int
	Body    []byte
}

func (e *NamergenceError) Error() string {
	return e.Message
}

type predictRequest struct {
	Name string `json:"name"`
}

func NormalizeBaseURL(baseURL string) string {
	return strings.TrimRight(baseURL, "/")
}

func Predict(name string, options *RequestOptions) (*PredictionResult, error) {
	opts := RequestOptions{}
	if options != nil {
		opts = *options
	}

	if opts.BaseURL == "" {
		opts.BaseURL = DEFAULT_BASE_URL
	}

	client := opts.Client
	if client == nil {
		client = &http.Client{}
	}
	if opts.Timeout > 0 {
		timeoutClient := *client
		timeoutClient.Timeout = opts.Timeout
		client = &timeoutClient
	}

	payload, err := json.Marshal(predictRequest{Name: name})
	if err != nil {
		return nil, err
	}

	url := NormalizeBaseURL(opts.BaseURL) + "/predict"
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	for key, value := range opts.Headers {
		req.Header.Set(key, value)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, &NamergenceError{
			Message: fmt.Sprintf("Namergence request failed with status %d", resp.StatusCode),
			Status:  resp.StatusCode,
			Body:    body,
		}
	}

	var result PredictionResult
	if err := json.Unmarshal(body, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

type Client struct {
	BaseURL string
	Client  *http.Client
	Headers map[string]string
}

func createClient(options *RequestOptions) *Client {
	opts := RequestOptions{}
	if options != nil {
		opts = *options
	}

	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = DEFAULT_BASE_URL
	}

	return &Client{
		BaseURL: NormalizeBaseURL(baseURL),
		Client:  opts.Client,
		Headers: opts.Headers,
	}
}

func CreateClient(options *RequestOptions) *Client {
	return createClient(options)
}

func NewClient(options *RequestOptions) *Client {
	return createClient(options)
}

func (c *Client) Predict(name string, requestOptions *RequestOptions) (*PredictionResult, error) {
	opts := RequestOptions{}
	if requestOptions != nil {
		opts = *requestOptions
	}

	opts.BaseURL = c.BaseURL
	if opts.Client == nil {
		opts.Client = c.Client
	}
	if opts.Headers == nil {
		opts.Headers = c.Headers
	}

	return Predict(name, &opts)
}
