/**
 * Canned prediction + report per bundled sample, so the whole UI is viewable
 * with zero backend. These are illustrative, NOT live model outputs — the UI
 * always badges them as demo. The canned report carries the DISCLAIMER prefix,
 * mirroring the real API contract (generate_report returns `${DISCLAIMER}\n\n${body}`).
 *
 * Confidences intentionally echo the project's honest metrics: cycling is the
 * model's strongest class, while walking is one of its weakest.
 *
 * Educational prototype — NOT a medical device.
 */
import { DISCLAIMER } from "@/lib/constants";
import type { PredictionResponse, ReportResponse } from "@/lib/types";

export interface DemoResult {
  sampleId: string;
  prediction: PredictionResponse;
  report: ReportResponse;
}

function make(
  sampleId: string,
  predicted_class: string,
  confidence: number,
  relevant_segment: [number, number],
  body: string,
): DemoResult {
  const prediction: PredictionResponse = {
    predicted_class,
    confidence,
    relevant_segment,
    disclaimer: DISCLAIMER,
  };
  const report: ReportResponse = {
    ...prediction,
    report: `${DISCLAIMER}\n\n${body}`,
  };
  return { sampleId, prediction, report };
}

export const DEMO_RESULTS: Record<string, DemoResult> = {
  cycling: make(
    "cycling",
    "cycling",
    0.98,
    [160, 352],
    "This window was classified as cycling with very high confidence (98%). The " +
      "combination of an elevated, steady heart rate and strong rhythmic wrist " +
      "motion is a pattern the model separates well. The highlighted segment marks " +
      "the stretch of signal the model focused on most.",
  ),
  sitting: make(
    "sitting",
    "sitting",
    0.8,
    [96, 288],
    "This window was classified as sitting with fairly high confidence (80%). A low, " +
      "steady heart rate together with almost no wrist movement points to a sedentary " +
      "activity. The highlighted segment is where the model's attention was strongest.",
  ),
  walking: make(
    "walking",
    "walking",
    0.36,
    [128, 320],
    "This window was classified as walking, but only with low confidence (36%). " +
      "Walking is one of this model's weakest classes — it is easily confused with " +
      "stairs and other light activities — so treat this label with caution. The " +
      "highlighted segment shows where the model focused.",
  ),
};

export function getDemoResult(sampleId: string): DemoResult | undefined {
  return DEMO_RESULTS[sampleId];
}
