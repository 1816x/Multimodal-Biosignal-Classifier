"use client";

import { useMemo } from "react";
import SignalChart from "@/components/SignalChart";
import { WINDOW_SAMPLES } from "@/lib/constants";
import type { PredictionRequest } from "@/lib/types";

/**
 * The three stacked modality plots (ECG, PPG, tri-axial ACC). All share one
 * relevant-segment band on the same 512-sample / 8 s base, so the shaded region
 * lines up across the charts.
 */
export default function SignalPanel({
  input,
  segment,
}: {
  input: PredictionRequest;
  segment: [number, number] | null;
}) {
  const ecgData = useMemo(
    () => input.ecg.map((v, i) => ({ i, ecg: v })),
    [input.ecg],
  );
  const ppgData = useMemo(
    () => input.ppg.map((v, i) => ({ i, ppg: v })),
    [input.ppg],
  );
  const accData = useMemo(
    () => input.acc.map((t, i) => ({ i, x: t[0], y: t[1], z: t[2] })),
    [input.acc],
  );

  return (
    <div className="flex flex-col gap-3">
      <SignalChart
        title="ECG"
        subtitle="chest → 64 Hz"
        data={ecgData}
        series={[{ key: "ecg", label: "ECG", color: "#ef4444" }]}
        segment={segment}
        xMax={WINDOW_SAMPLES}
      />
      <SignalChart
        title="PPG"
        subtitle="wrist BVP"
        data={ppgData}
        series={[{ key: "ppg", label: "PPG", color: "#2563eb" }]}
        segment={segment}
        xMax={WINDOW_SAMPLES}
      />
      <SignalChart
        title="Accelerometer"
        subtitle="wrist, 3-axis"
        data={accData}
        series={[
          { key: "x", label: "x", color: "#f59e0b" },
          { key: "y", label: "y", color: "#10b981" },
          { key: "z", label: "z", color: "#8b5cf6" },
        ]}
        segment={segment}
        xMax={WINDOW_SAMPLES}
      />
    </div>
  );
}
