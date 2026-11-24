import { NamergenceClient } from "./index.js";

const client = new NamergenceClient({ baseUrl: "http://127.0.0.1:8000" });

const result = await client.predict("Charlotte");

console.log(`Predictions for ${result.name}:`);
for (const { category, confidence } of result.predictions) {
  console.log(`  ${category}: ${(confidence * 100).toFixed(2)}%`);
}
