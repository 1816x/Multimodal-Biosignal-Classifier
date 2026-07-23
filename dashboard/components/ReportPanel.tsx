"use client";

import DemoBadge from "@/components/DemoBadge";
import { DISCLAIMER } from "@/lib/constants";
import type { ReportResponse, Sourced } from "@/lib/types";

/**
 * Renders the Claude-generated report. The API prepends the exact DISCLAIMER +
 * "\n\n" to every report body; we split that known prefix into a styled callout
 * and render the remaining body as paragraphs (no Markdown dependency needed).
 */
export default function ReportPanel({
  result,
}: {
  result: Sourced<ReportResponse>;
}) {
  const { data } = result;
  const isDemo = result.source === "demo";

  const full = data.report ?? "";
  const body = full.startsWith(DISCLAIMER)
    ? full.slice(DISCLAIMER.length).trimStart()
    : full;
  const paragraphs = body.split(/\n{2,}/).filter((p) => p.trim().length > 0);

  return (
    <section className="rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
      <div className="mb-3 flex items-center justify-between gap-2">
        <h2 className="text-sm font-semibold text-zinc-700 dark:text-zinc-300">
          Report
        </h2>
        {isDemo ? <DemoBadge title={result.reason}>Demo</DemoBadge> : null}
      </div>

      <div className="mb-3 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-800/60 dark:bg-amber-950/50 dark:text-amber-200">
        {data.disclaimer || DISCLAIMER}
      </div>

      <div className="flex flex-col gap-2 text-sm leading-relaxed text-zinc-800 dark:text-zinc-200">
        {paragraphs.map((p, idx) => (
          <p key={idx}>{p}</p>
        ))}
      </div>
    </section>
  );
}
