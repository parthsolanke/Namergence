package namergence

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestPong(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Errorf("method = %s, want GET", r.Method)
		}
		if r.URL.Path != "/pong" {
			t.Errorf("path = %s, want /pong", r.URL.Path)
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(PingResponse{Message: "pong", Service: "scalarflow"})
	}))
	defer server.Close()

	client := NewPongClient(server.URL + "/")
	pong, err := client.Pong(context.Background())
	if err != nil {
		t.Fatalf("Pong() error = %v", err)
	}
	if pong.Message != "pong" {
		t.Errorf("message = %q, want pong", pong.Message)
	}
	if pong.Service != "scalarflow" {
		t.Errorf("service = %q, want scalarflow", pong.Service)
	}
}

func TestPongServerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "not available", http.StatusServiceUnavailable)
	}))
	defer server.Close()

	client := NewPongClient(server.URL)
	if _, err := client.Pong(context.Background()); err == nil {
		t.Fatal("Pong() error = nil, want server error")
	}
}
