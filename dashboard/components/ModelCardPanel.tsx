import { MODEL_CARD as M } from "@/lib/modelCard";

const pct = (x: number) => `${Math.round(x * 100)}%`;
const times = (x: number) => `${(x / M.chanceLevel).toFixed(1)}×`;

/**
 * Collapsible "About this model" panel surfacing the honest headline metrics, so
 * predictions are read with appropriate skepticism. Mirrors MODEL_CARD.md.
 */
export default function ModelCardPanel() {
  return (
    <details className="group rounded-lg border border-zinc-200 bg-white text-sm dark:border-zinc-800 dark:bg-zinc-900">
      <summary className="cursor-pointer list-none px-4 py-3 font-semibold text-zinc-700 marker:content-none dark:text-zinc-300">
        <span className="mr-1 inline-block transition-transform group-open:rotate-90">▸</span>
        About this model — honest metrics
      </summary>
      <div className="flex flex-col gap-3 border-t border-zinc-200 px-4 py-3 text-zinc-700 dark:border-zinc-800 dark:text-zinc-300">
        <p>
          Test accuracy on held-out subjects:{" "}
          <strong className="text-zinc-900 dark:text-zinc-100">
            {pct(M.multimodal.testAccuracy)}
          </strong>{" "}
          multimodal ({times(M.multimodal.testAccuracy)} chance) vs{" "}
          <strong className="text-zinc-900 dark:text-zinc-100">
            {pct(M.ecgOnly.testAccuracy)}
          </strong>{" "}
          ECG-only. Chance for 8 classes is {(M.chanceLevel * 100).toFixed(1)}%.
        </p>
        <p>
          A genuine result, <strong>not</strong> a solved task: weakest classes are{" "}
          <strong>{M.weakestClasses.join(" & ")}</strong> (easily confused with
          look-alike activities). Strongest: {M.strongestClasses.join(", ")}.
        </p>
        <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
          <dt className="text-zinc-500 dark:text-zinc-400">Dataset</dt>
          <dd>{M.dataset}</dd>
          <dt className="text-zinc-500 dark:text-zinc-400">Split</dt>
          <dd>{M.split}</dd>
          <dt className="text-zinc-500 dark:text-zinc-400">Model size</dt>
          <dd>{M.multimodal.params.toLocaleString()} params (multimodal)</dd>
        </dl>
        <p className="text-xs text-zinc-500 dark:text-zinc-400">
          Metrics are illustrative and educational. Full breakdown, per-class
          numbers, and limitations: <code>MODEL_CARD.md</code>.
        </p>
      </div>
    </details>
  );
}
