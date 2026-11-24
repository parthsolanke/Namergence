# Namergence JS Client

A small, dependency-free JavaScript client for the Namergence FastAPI
`POST /predict` endpoint. Works in modern browsers and Node.js 18+ (both ship
`fetch`).

## Install

Copy the `js-client` folder into your project, or import `index.js` directly.
The package is published as an ES module (`"type": "module"`).

## Usage

```js
import { NamergenceClient } from "./js-client/index.js";

const client = new NamergenceClient({ baseUrl: "http://127.0.0.1:8000" });
const result = await client.predict("Charlotte");

console.log(result.label);
for (const { category, confidence } of result.predictions) {
  console.log(`${category}: ${(confidence * 100).toFixed(2)}%`);
}
```

A one-shot helper is also available:

```js
import { predictName } from "./js-client/index.js";

const result = await predictName("Charlotte");
```

## API

`new NamergenceClient({ baseUrl, fetch })`

- `baseUrl` — backend URL, defaults to `http://127.0.0.1:8000`.
- `fetch` — optional custom fetch implementation (defaults to `globalThis.fetch`).

`client.predict(name, { signal })` returns a `Promise` resolving to:

```ts
{
  name: string;
  label: string;
  predictions: { category: string; confidence: number }[]; // sorted, highest first
  raw: { name, label, confidences, categories };           // untouched server response
}
```

Failed HTTP responses reject with a `NamergenceError` exposing `status` and
`body`. An optional `AbortSignal` cancels an in-flight request.

Run the bundled example (with the backend running):

```bash
node js-client/example.js
```
