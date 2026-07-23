/**
 * Bundled synthetic input windows for the demo. Each is a canonical 512-sample
 * window per modality, clearly marked synthetic. These let the dashboard render
 * real-looking signal plots on a fresh clone with no backend and no dataset.
 *
 * Educational prototype — NOT a medical device.
 */
import { synthWindow } from "@/lib/fixtures/synth";
import type { PredictionRequest } from "@/lib/types";

export interface SampleWindow {
  id: string;
  label: string;
  synthetic: true;
  note: string;
  input: PredictionRequest;
}

export const SAMPLES: SampleWindow[] = [
  {
    id: "cycling",
    label: "Cycling-like",
    synthetic: true,
    note: "Synthetic — elevated heart rate with strong, steady rhythmic wrist motion.",
    input: synthWindow({ seed: 101, bpm: 132, accFreqHz: 1.4, accAmp: 0.7, gravityAxis: 2 }),
  },
  {
    id: "sitting",
    label: "Sitting-like",
    synthetic: true,
    note: "Synthetic — low, steady heart rate with almost no wrist motion.",
    input: synthWindow({ seed: 202, bpm: 66, accFreqHz: 0.2, accAmp: 0.03, gravityAxis: 2 }),
  },
  {
    id: "walking",
    label: "Walking-like",
    synthetic: true,
    note: "Synthetic — moderate heart rate with periodic ~2 Hz wrist motion.",
    input: synthWindow({ seed: 303, bpm: 102, accFreqHz: 2.0, accAmp: 0.35, gravityAxis: 2 }),
  },
];

export const DEFAULT_SAMPLE_ID = SAMPLES[0].id;

export function getSample(id: string): SampleWindow | undefined {
  return SAMPLES.find((s) => s.id === id);
}
