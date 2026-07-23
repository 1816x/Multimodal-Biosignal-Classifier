import { describe, expect, it } from "vitest";
import { WINDOW_SAMPLES } from "@/lib/constants";
import { validateWindow } from "@/lib/validateWindow";

function validWindow() {
  return {
    ecg: Array(WINDOW_SAMPLES).fill(0),
    ppg: Array(WINDOW_SAMPLES).fill(0),
    acc: Array.from({ length: WINDOW_SAMPLES }, () => [0, 0, 0]),
  };
}

describe("validateWindow", () => {
  it("accepts a well-formed 512-sample window", () => {
    const result = validateWindow(validWindow());
    expect(result.ok).toBe(true);
    expect(result.input?.ecg).toHaveLength(WINDOW_SAMPLES);
    expect(result.input?.acc).toHaveLength(WINDOW_SAMPLES);
  });

  it("rejects non-object input", () => {
    expect(validateWindow(null).ok).toBe(false);
    expect(validateWindow([1, 2, 3]).ok).toBe(false);
  });

  it("rejects wrong lengths", () => {
    const bad = { ecg: [1, 2], ppg: [], acc: [] };
    const result = validateWindow(bad);
    expect(result.ok).toBe(false);
    expect(result.errors.length).toBeGreaterThan(0);
  });

  it("rejects accelerometer samples that are not [x, y, z] triples", () => {
    const bad = {
      ...validWindow(),
      acc: Array.from({ length: WINDOW_SAMPLES }, () => [0, 0]), // pairs, not triples
    };
    expect(validateWindow(bad).ok).toBe(false);
  });
});
