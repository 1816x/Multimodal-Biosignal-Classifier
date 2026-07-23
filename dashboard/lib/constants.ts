/**
 * Constants mirrored from the backend so the UI stays in lock-step with the model
 * and the API contract. Keep these in sync with their canonical sources:
 *   - DISCLAIMER            -> api/src/biosignal_api/__init__.py
 *   - ACTIVITIES            -> model/src/biosignal_model/config.py (PPG_DALIA_ACTIVITIES)
 *   - WINDOW_SAMPLES/TARGET_HZ/WINDOW_SECONDS -> model ModelConfig (8 s @ 64 Hz)
 *
 * Educational prototype — NOT a medical device.
 */

/** Canonical educational disclaimer (verbatim copy of the API's DISCLAIMER). */
export const DISCLAIMER =
  "EDUCATIONAL PROTOTYPE — NOT an approved medical or diagnostic tool. " +
  "This service must not be used for clinical decisions. Model outputs are " +
  "illustrative only.";

/** One window = 8 s at 64 Hz -> 512 samples per modality (the resampled grid). */
export const WINDOW_SECONDS = 8;
export const TARGET_HZ = 64;
export const WINDOW_SAMPLES = WINDOW_SECONDS * TARGET_HZ; // 512

/** PPG-DaLiA activity labels, in the model's class order. */
export const ACTIVITIES = [
  "sitting",
  "stairs",
  "table_soccer",
  "cycling",
  "driving",
  "lunch_break",
  "walking",
  "working",
] as const;

/** Human-readable labels for the activity classes. */
export const ACTIVITY_LABELS: Record<string, string> = {
  sitting: "Sitting",
  stairs: "Stairs",
  table_soccer: "Table soccer",
  cycling: "Cycling",
  driving: "Driving",
  lunch_break: "Lunch break",
  walking: "Walking",
  working: "Working",
};

/** Display metadata for each modality plotted in the dashboard. */
export const MODALITIES = [
  {
    key: "ecg" as const,
    label: "ECG",
    description: "Electrocardiogram (chest, resampled to 64 Hz)",
    channels: ["ecg"],
  },
  {
    key: "ppg" as const,
    label: "PPG",
    description: "Photoplethysmogram / BVP (wrist)",
    channels: ["ppg"],
  },
  {
    key: "acc" as const,
    label: "Accelerometer",
    description: "3-axis wrist accelerometer",
    channels: ["x", "y", "z"],
  },
];
