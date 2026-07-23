/**
 * TypeScript mirrors of the FastAPI Pydantic schemas
 * (api/src/biosignal_api/schemas.py). Keep these 1:1 with the backend contract.
 *
 * Educational prototype — NOT a medical device.
 */

/** Mirrors PredictionRequest — one window of raw samples per modality. */
export interface PredictionRequest {
  /** ECG samples for one window (flat list). */
  ecg: number[];
  /** PPG (BVP) samples for one window (flat list). */
  ppg: number[];
  /** Accelerometer samples for one window, each a [x, y, z] triple. */
  acc: [number, number, number][];
}

/** Mirrors PredictionResponse. */
export interface PredictionResponse {
  /** One of ACTIVITIES. */
  predicted_class: string;
  /** Softmax max, 0..1. */
  confidence: number;
  /** [start, end] sample indices in the resampled (512) grid. */
  relevant_segment: [number, number];
  disclaimer: string;
}

/** Mirrors ReportResponse (extends PredictionResponse with the report text). */
export interface ReportResponse extends PredictionResponse {
  /** Claude-generated natural-language report, always DISCLAIMER-prefixed. */
  report: string;
}

/** Mirrors HealthResponse. */
export interface HealthResponse {
  status: string;
  version: string;
}

/** FastAPI error envelope ({"detail": "..."}). */
export interface ApiErrorBody {
  detail: string;
}

/**
 * Provenance wrapper so the UI can never confuse a live model output with a
 * canned demo one. Every result carries where it came from; demo results also
 * carry the reason the app fell back (503, offline, ...).
 */
export type Sourced<T> =
  | { source: "live"; data: T }
  | { source: "demo"; data: T; reason: string };
