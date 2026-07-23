/**
 * Browser-side API client. Calls the same-origin Next.js proxy under /api/*,
 * which forwards to the FastAPI service. Non-2xx responses become a thrown
 * ApiError carrying the HTTP status and the FastAPI `detail`, so callers can
 * branch on 503 (model/backend unavailable), 422 (bad input) and 502 (Claude
 * upstream failure).
 *
 * Educational prototype — NOT a medical device.
 */
import type {
  HealthResponse,
  PredictionRequest,
  PredictionResponse,
  ReportResponse,
} from "@/lib/types";

export class ApiError extends Error {
  readonly status: number;
  readonly detail: string;
  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, init);
  } catch {
    // The Next proxy is same-origin; a throw here means the dev server itself
    // is unreachable. Surface it like an offline backend.
    throw new ApiError(0, "Network error — the dashboard server is unreachable.");
  }

  const text = await res.text();
  let data: unknown = undefined;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = undefined;
    }
  }

  if (!res.ok) {
    const detail =
      data && typeof data === "object" && "detail" in data && typeof data.detail === "string"
        ? data.detail
        : res.statusText || `Request failed (${res.status})`;
    throw new ApiError(res.status, detail);
  }

  return data as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/api/health", { cache: "no-store" });
}

export function postPredict(body: PredictionRequest): Promise<PredictionResponse> {
  return request<PredictionResponse>("/api/predict", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function postReport(body: PredictionRequest): Promise<ReportResponse> {
  return request<ReportResponse>("/api/report", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}
