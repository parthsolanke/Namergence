# namergence-client (Go)

Go client for the Namergence FastAPI server. It sends `{ "name": string }` to `POST /predict` and returns the server response unchanged.

```go
package main

import (
	"fmt"

	namergence "namergence-client"
)

func main() {
	result, err := namergence.Predict("Alice", &namergence.RequestOptions{
		BaseURL: "http://127.0.0.1:8000",
	})
	if err != nil {
		panic(err)
	}
	fmt.Println(result.Label, result.Confidences)
}
```

You can also create a configured client:

```go
client := namergence.NewClient(&namergence.RequestOptions{
	BaseURL: "http://127.0.0.1:8000",
	Headers: map[string]string{"X-Request-ID": "demo"},
})

result, err := client.Predict("Alice", nil)
```

`RequestOptions` supports `BaseURL`, `Timeout`, `Headers` and `Client`. Non-2xx responses return a `*NamergenceError` exposing `Status` and `Body`. `DefaultBaseURL` (also exported as `DEFAULT_BASE_URL`) is `http://127.0.0.1:8000`, and `NormalizeBaseURL` strips trailing slashes.
