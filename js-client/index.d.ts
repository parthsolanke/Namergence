export interface NamergenceClientOptions {
  baseUrl?: string;
  fetch?: typeof fetch;
}

export interface Prediction {
  category: string;
  confidence: number;
}

export interface PredictionResult {
  name: string;
  label: string;
  predictions: Prediction[];
  raw: {
    name: string;
    label: string;
    confidences: number[][];
    categories: string[];
  };
}

export class NamergenceError extends Error {
  status: number;
  body: unknown;
  constructor(message: string, status: number, body: unknown);
}

export class NamergenceClient {
  constructor(options?: NamergenceClientOptions);
  predict(
    name: string,
    options?: { signal?: AbortSignal }
  ): Promise<PredictionResult>;
}

export function normalizePrediction(data: PredictionResult["raw"]): PredictionResult;
export function predictName(
  name: string,
  options?: NamergenceClientOptions
): Promise<PredictionResult>;
export default NamergenceClient;
