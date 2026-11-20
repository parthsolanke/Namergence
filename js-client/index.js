const DEFAULT_BASE_URL = "http://127.0.0.1:8000";

export class NamergenceError extends Error {
  constructor(message, status, body) {
    super(message);
    this.name = "NamergenceError";
    this.status = status;
    this.body = body;
  }
}

function normalizeBaseUrl(baseUrl) {
  return String(baseUrl).replace(/\/+$/, "");
}

export async function predict(name, options = {}) {
  if (typeof name !== "string") {
    throw new TypeError("predict expects the name to be a string");
  }

  const {
    baseUrl = DEFAULT_BASE_URL,
    fetch: fetchImpl = globalThis.fetch,
    signal
  } = options;

  if (typeof fetchImpl !== "function") {
    throw new TypeError("No fetch implementation available; pass one via options.fetch");
  }

  const url = `${normalizeBaseUrl(baseUrl)}/predict`;
  const response = await fetchImpl(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "application/json"
    },
    body: JSON.stringify({ name }),
    signal
  });

  if (!response.ok) {
    let body;
    try {
      body = await response.json();
    } catch (error) {
      body = await response.text();
    }
    throw new NamergenceError(
      `Namergence request failed with status ${response.status}`,
      response.status,
      body
    );
  }

  return response.json();
}

export function createClient(options = {}) {
  const baseUrl = normalizeBaseUrl(options.baseUrl ?? DEFAULT_BASE_URL);
  const fetchImpl = options.fetch;

  return {
    baseUrl,
    predict(name, requestOptions = {}) {
      return predict(name, { ...requestOptions, baseUrl, fetch: fetchImpl });
    }
  };
}

export default { predict, createClient, NamergenceError, DEFAULT_BASE_URL };
