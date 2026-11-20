export interface PredictionResult {
  name: string;
  label: string;
  confidences: number[];
  categories: string[];
}

export interface RequestOptions {
  baseUrl?: string;
  fetch?: typeof fetch;
  signal?: AbortSignal;
}

export class NamergenceError extends Error {
  readonly status: number;
  readonly body: unknown;
  constructor(message: string, status: number, body: unknown);
}

export const DEFAULT_BASE_URL: string;

export function predict(name: string, options?: RequestOptions): Promise<PredictionResult>;

export interface Client {
  baseUrl: string;
  predict(name: string, options?: Omit<RequestOptions, "baseUrl">): Promise<PredictionResult>;
}

export function createClient(options?: RequestOptions): Client;

declare const namergence: {
  predict: typeof predict;
  createClient: typeof createClient;
  NamergenceError: typeof NamergenceError;
  DEFAULT_BASE_URL: string;
};

export default namergence;
