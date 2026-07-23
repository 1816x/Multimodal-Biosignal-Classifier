"use client";

import DemoBadge from "@/components/DemoBadge";
import { ACTIVITY_LABELS } from "@/lib/constants";
import type { PredictionResponse, Sourced } from "@/lib/types";

/** The model's numeric output: activity, confidence, and relevant segment. */
export default function PredictionCard({
  result,
}: {
  result: Sourced<PredictionResponse>;
}) {
  const { data } = result;
  const isDemo = result.source === "demo";
  const label = ACTIVITY_LABELS[data.predicted_class] ?? data.predicted_class;
  const pct = Math.round(data.confidence * 100);
  const [start, end] = data.relevant_segment;

  return (
    <section className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">
          Prediction
        </h2>
        {isDemo ? <DemoBadge title={result.reason}>Demo</DemoBadge> : null}
      </div>

      <div className="flex items-end justify-between gap-4">
        <div>
          <div className="text-2xl font-bold">{label}</div>
          <div className="text-xs text-zinc-500 dark:text-zinc-400">
            relevant segment: samples {start}–{end}
          </div>
        </div>
        <div className="text-right">
          <div className="text-2xl font-bold tabular-nums">{pct}%</div>
          <div className="text-xs text-zinc-500 dark:text-zinc-400">confidence</div>
        </div>
      </div>

      <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-zinc-200 dark:bg-zinc-800">
        <div
          className="h-full rounded-full bg-emerald-500"
          style={{ width: `${pct}%` }}
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
    </section>
  );
}
