/**
 * Headline model-card metrics surfaced in the dashboard, mirrored from the honest
 * numbers in model/metrics/*.json and MODEL_CARD.md. Keep in sync with those.
 *
 * Educational prototype — NOT a medical device.
 */
export const MODEL_CARD = {
  version: "0.1.0",
  dataset: "PPG-DaLiA (UCI #495, CC BY 4.0)",
  split: "subject-wise · train S1–S11 · val S12–S13 · test S14–S15",
  chanceLevel: 0.125, // 8 balanced classes
  ecgOnly: {
    label: "ECG-only (Phase 1)",
    params: 53_256,
    testAccuracy: 0.371,
    testMacroF1: 0.385,
  },
  multimodal: {
    label: "Multimodal (Phase 2)",
    params: 157_896,
    testAccuracy: 0.65,
    testMacroF1: 0.676,
    valAccuracy: 0.712,
  },
  strongestClasses: ["cycling", "table_soccer", "sitting"],
  weakestClasses: ["walking", "working"],
} as const;
