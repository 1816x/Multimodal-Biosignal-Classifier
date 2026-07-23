"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useMounted } from "@/lib/useMounted";

export interface ChartSeries {
  key: string;
  label: string;
  color: string;
}

interface SignalChartProps {
  title: string;
  subtitle?: string;
  /** Each row is `{ i, <seriesKey>: value, ... }`. */
  data: Array<Record<string, number>>;
  series: ChartSeries[];
  /** [start, end] sample indices to shade, or null. */
  segment: [number, number] | null;
  /** Number of samples on the x axis (512). */
  xMax: number;
}

/**
 * One modality's signal plot: a line per series plus a shaded ReferenceArea for
 * the model's relevant segment. Rendered client-side only (after mount) so the
 * SSR pass emits a stable-height skeleton and Recharts never causes a hydration
 * mismatch or a zero-size warning.
 */
export default function SignalChart({
  title,
  subtitle,
  data,
  series,
  segment,
  xMax,
}: SignalChartProps) {
  const mounted = useMounted();

  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-3 dark:border-zinc-800 dark:bg-zinc-900">
      <div className="mb-1 flex items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold">{title}</h3>
        {subtitle ? (
          <span className="text-xs text-zinc-500 dark:text-zinc-400">{subtitle}</span>
        ) : null}
      </div>
      <div className="h-40 w-full">
        {mounted ? (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: -12 }}>
              <CartesianGrid strokeDasharray="3 3" className="stroke-zinc-200 dark:stroke-zinc-800" />
              <XAxis
                dataKey="i"
                type="number"
                domain={[0, xMax - 1]}
                tickCount={5}
                allowDecimals={false}
                tick={{ fontSize: 11 }}
                stroke="currentColor"
                className="text-zinc-400"
              />
              <YAxis
                tick={{ fontSize: 11 }}
                width={44}
                stroke="currentColor"
                className="text-zinc-400"
              />
              <Tooltip
                isAnimationActive={false}
                labelFormatter={(v) => `sample ${v}`}
                contentStyle={{ fontSize: 12 }}
              />
              {segment ? (
                <ReferenceArea
                  x1={segment[0]}
                  x2={segment[1]}
                  fill="#f59e0b"
                  fillOpacity={0.15}
                  stroke="#f59e0b"
                  strokeOpacity={0.4}
                />
              ) : null}
              {series.map((s) => (
                <Line
                  key={s.key}
                  type="monotone"
                  dataKey={s.key}
                  name={s.label}
                  stroke={s.color}
                  strokeWidth={1.4}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <div className="h-full w-full animate-pulse rounded bg-zinc-100 dark:bg-zinc-800" />
        )}
      </div>
    </div>
  );
}
