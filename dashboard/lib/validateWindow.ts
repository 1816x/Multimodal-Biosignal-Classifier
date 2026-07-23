/**
 * Pure validator for user-supplied windows (uploaded or pasted JSON). Mirrors
 * the API's 422 contract but is stricter on length: the dashboard plots and the
 * relevant-segment overlay assume the canonical 512-sample grid, so a window
 * must be exactly WINDOW_SAMPLES long per modality.
 *
 * Educational prototype — NOT a medical device.
 */
import { WINDOW_SAMPLES } from "@/lib/constants";
import type { PredictionRequest } from "@/lib/types";

export interface WindowValidation {
  ok: boolean;
  errors: string[];
  input?: PredictionRequest;
}

function isNumberArray(v: unknown, len: number): v is number[] {
  return (
    Array.isArray(v) &&
    v.length === len &&
    v.every((x) => typeof x === "number" && Number.isFinite(x))
  );
}

function isTripleArray(v: unknown, len: number): v is [number, number, number][] {
  return (
    Array.isArray(v) &&
    v.length === len &&
    v.every(
      (t) =>
        Array.isArray(t) &&
        t.length === 3 &&
        t.every((x) => typeof x === "number" && Number.isFinite(x)),
    )
  );
}

export function validateWindow(raw: unknown): WindowValidation {
  if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
    return {
      ok: false,
      errors: ["Expected a JSON object with numeric `ecg`, `ppg` and `acc` fields."],
    };
  }
  const obj = raw as Record<string, unknown>;
  const errors: string[] = [];

  if (!isNumberArray(obj.ecg, WINDOW_SAMPLES)) {
    errors.push(`\`ecg\` must be an array of exactly ${WINDOW_SAMPLES} finite numbers.`);
  }
  if (!isNumberArray(obj.ppg, WINDOW_SAMPLES)) {
    errors.push(`\`ppg\` must be an array of exactly ${WINDOW_SAMPLES} finite numbers.`);
  }
  if (!isTripleArray(obj.acc, WINDOW_SAMPLES)) {
    errors.push(`\`acc\` must be an array of exactly ${WINDOW_SAMPLES} [x, y, z] number triples.`);
  }

  if (errors.length > 0) {
    return { ok: false, errors };
  }

  return {
    ok: true,
    errors: [],
    input: {
      ecg: obj.ecg as number[],
      ppg: obj.ppg as number[],
      acc: obj.acc as [number, number, number][],
    },
  };
}
