# namergence-client

JavaScript client for the Namergence FastAPI server. It sends `{ "name": string }` to `POST /predict` and returns the server response unchanged.

```js
import { predict } from "namergence-client";

const result = await predict("Alice", { baseUrl: "http://127.0.0.1:8000" });
console.log(result.label, result.confidences);
```

You can also create a configured client:

```js
import { createClient } from "namergence-client";

const client = createClient({ baseUrl: "http://127.0.0.1:8000" });
const result = await client.predict("Alice");
```

Non-2xx responses throw a `NamergenceError` exposing `status` and `body`.
