const DEFAULT_BASE_URL = "http://127.0.0.1:8000";

export class NamergenceError extends Error {
  constructor(message, status, body) {
    super(message);
    this.name = "NamergenceError";
    this.status = status;
    this.body = body;
  }
}

function toRankedPredictions(data) {
  const categories = Array.isArray(data.categories) ? data.categories : [];
  const raw = Array.isArray(data.confidences) ? data.confidences : [];
  const scores = raw.length === 1 && Array.isArray(raw[0]) ? raw[0] : raw;

  return categories
    .map((category, index) => {
      const score = Number(scores[index] ?? 0);
      const probability = Math.exp(score);
      return {
        category,
        logProbability: score,
        probability,
        percent: probability * 100,
      };
    })
    .sort((a, b) => b.probability - a.probability);
}

function normalizeResponse(data) {
  return {
    name: data.name,
    label: data.label,
    categories: Array.isArray(data.categories) ? data.categories : [],
    predictions: toRankedPredictions(data),
    raw: data,
  };
}

export class NamergenceClient {
  constructor({ baseUrl = DEFAULT_BASE_URL, fetchImpl } = {}) {
    if (typeof baseUrl !== "string" || baseUrl.length === 0) {
      throw new TypeError("baseUrl must be a non-empty string");
    }

    const fetchFn = fetchImpl || globalThis.fetch;
    if (typeof fetchFn !== "function") {
      throw new TypeError(
        "No fetch implementation found. Use Node.js 18+ or pass fetchImpl."
      );
    }

    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.fetch = fetchFn;
  }

  async predict(name) {
    if (typeof name !== "string") {
      throw new TypeError("name must be a string");
    }

    const response = await this.fetch(`${this.baseUrl}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });

    const text = await response.text();
    let body;
    try {
      body = text ? JSON.parse(text) : null;
    } catch (error) {
      body = text;
    }

    if (!response.ok) {
      throw new NamergenceError(
        `Request failed with status ${response.status}`,
        response.status,
        body
      );
    }

    if (!body || typeof body !== "object") {
      throw new NamergenceError("Unexpected response from /predict", response.status, body);
    }

    return normalizeResponse(body);
  }
}

export function createClient(options) {
  return new NamergenceClient(options);
}

export async function predictName(name, options) {
  const client = new NamergenceClient(options);
  return client.predict(name);
}
