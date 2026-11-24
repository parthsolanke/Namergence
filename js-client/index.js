export const DEFAULT_BASE_URL = "http://127.0.0.1:8000";

export class NamergenceError extends Error {
  constructor(message, status, body) {
    super(message);
    this.name = "NamergenceError";
    this.status = status;
    this.body = body;
  }
}

export class NamergenceClient {
  constructor({ baseUrl = DEFAULT_BASE_URL, fetch: fetchImpl } = {}) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.fetch = fetchImpl || globalThis.fetch;

    if (typeof this.fetch !== "function") {
      throw new Error(
        "No fetch implementation found. Use Node.js 18+ or pass a fetch polyfill."
      );
    }
  }

  async predict(name, { signal } = {}) {
    if (typeof name !== "string" || name.trim() === "") {
      throw new TypeError("name must be a non-empty string");
    }

    const response = await this.fetch(`${this.baseUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
      signal,
    });

    if (!response.ok) {
      let body = null;
      try {
        body = await response.json();
      } catch {
        body = await response.text().catch(() => null);
      }
      throw new NamergenceError(
        `Prediction failed with status ${response.status}`,
        response.status,
        body
      );
    }

    const data = await response.json();
    return normalizePrediction(data);
  }
}

export function normalizePrediction(data) {
  const categories = data.categories || [];
  const scores = Array.isArray(data.confidences?.[0])
    ? data.confidences[0]
    : data.confidences || [];

  const predictions = categories
    .map((category, index) => ({
      category,
      confidence: typeof scores[index] === "number" ? scores[index] : 0,
    }))
    .sort((a, b) => b.confidence - a.confidence);

  return {
    name: data.name,
    label: data.label,
    predictions,
    raw: data,
  };
}

export async function predictName(name, options = {}) {
  return new NamergenceClient(options).predict(name);
}

export default NamergenceClient;
