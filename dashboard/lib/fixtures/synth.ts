/**
 * Deterministic synthetic biosignal generators for the bundled demo samples.
 *
 * These are NOT real recordings. The project deliberately never commits the
 * PPG-DaLiA dataset (2.6 GB, CC BY 4.0), and a fresh clone has no trained model
 * anyway — so the demo ships loudly-labeled synthetic windows instead of
 * dataset-derived ones. Generation is seeded (mulberry32) so the same sample is
 * byte-for-byte reproducible across runs and tests.
 *
 * Educational prototype — NOT a medical device.
 */
import { TARGET_HZ, WINDOW_SAMPLES } from "@/lib/constants";
import type { PredictionRequest } from "@/lib/types";

/** Small seeded PRNG so fixtures are deterministic (no Math.random). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Crude PQRST morphology as a function of beat phase in [0, 1). */
function ecgWave(phase: number): number {
  const bump = (center: number, width: number, amp: number) =>
    amp * Math.exp(-((phase - center) ** 2) / (2 * width * width));
  return (
    bump(0.2, 0.025, 0.12) - // P wave (positive)
    bump(0.38, 0.012, 0.16) + // Q trough (negative)
    bump(0.4, 0.011, 1.0) - // R spike (positive)
    bump(0.42, 0.013, 0.28) + // S trough (negative)
    bump(0.62, 0.045, 0.3) // T wave (positive)
  );
}

/** Smooth pulsatile PPG/BVP-ish wave with a dicrotic-notch approximation. */
function ppgWave(phase: number): number {
  return (
    0.6 * Math.sin(2 * Math.PI * phase) +
    0.25 * Math.sin(4 * Math.PI * phase + 0.9)
  );
}

export interface SynthOpts {
  /** PRNG seed — fixes the whole window. */
  seed: number;
  /** Heart rate in beats per minute (drives ECG/PPG periodicity). */
  bpm: number;
  /** Dominant wrist-motion frequency in Hz (drives the accelerometer). */
  accFreqHz: number;
  /** Motion amplitude for the accelerometer (0 ≈ still, ~0.7 ≈ vigorous). */
  accAmp: number;
  /** Axis (0=x,1=y,2=z) carrying the ~1 g gravity baseline. */
  gravityAxis?: 0 | 1 | 2;
}

/** Build one canonical 512-sample window (ECG, PPG, tri-axial ACC). */
export function synthWindow(opts: SynthOpts): PredictionRequest {
  const rnd = mulberry32(opts.seed);
  const n = WINDOW_SAMPLES;
  const g = opts.gravityAxis ?? 2;
  const beatT = 60 / opts.bpm; // seconds per beat

  const ecg: number[] = new Array(n);
  const ppg: number[] = new Array(n);
  const acc: [number, number, number][] = new Array(n);

  for (let i = 0; i < n; i++) {
    const t = i / TARGET_HZ;
    const phase = (t % beatT) / beatT;
    ecg[i] = ecgWave(phase) + (rnd() - 0.5) * 0.05;
    ppg[i] = ppgWave(phase) + (rnd() - 0.5) * 0.04;

    const motion = (axisPhase: number) =>
      Math.sin(2 * Math.PI * opts.accFreqHz * t + axisPhase) * opts.accAmp +
      (rnd() - 0.5) * opts.accAmp * 0.3;
    const triple: [number, number, number] = [motion(0), motion(2.1), motion(4.2)];
    triple[g] += 1.0; // gravity baseline
    acc[i] = triple;
  }

  return { ecg, ppg, acc };
}
