"use client";

import { useState } from "react";
import type { SampleWindow } from "@/lib/fixtures/samples";
import type { PredictionRequest } from "@/lib/types";
import { validateWindow } from "@/lib/validateWindow";

interface InputControlsProps {
  samples: SampleWindow[];
  /** Selected bundled sample id, or null when a custom window is loaded. */
  selectedSampleId: string | null;
  onSelectSample: (id: string) => void;
  onLoadCustom: (input: PredictionRequest, label: string) => void;
  onError: (message: string) => void;
  onRunPrediction: () => void;
  onExplain: () => void;
  canExplain: boolean;
  predicting: boolean;
  reporting: boolean;
}

/**
 * Window input: pick a bundled synthetic sample (default), or upload / paste a
 * JSON window. Two separate actions mirror the API split — a fast "Run
 * prediction" (/predict) and an explicit, opt-in "Explain with Claude" (/report).
 */
export default function InputControls({
  samples,
  selectedSampleId,
  onSelectSample,
  onLoadCustom,
  onError,
  onRunPrediction,
  onExplain,
  canExplain,
  predicting,
  reporting,
}: InputControlsProps) {
  const [pasted, setPasted] = useState("");

  function ingest(text: string, label: string) {
    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch {
      onError("Could not parse JSON — check the file/paste is valid JSON.");
      return;
    }
    const result = validateWindow(parsed);
    if (!result.ok || !result.input) {
      onError(`Invalid window: ${result.errors.join(" ")}`);
      return;
    }
    onLoadCustom(result.input, label);
  }

  async function onFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const text = await file.text();
      ingest(text, file.name);
    } catch {
      onError("Could not read the selected file.");
    } finally {
      e.target.value = ""; // allow re-selecting the same file
    }
  }

  const selectValue = selectedSampleId ?? "__custom__";

  return (
    <section className="flex flex-col gap-4 rounded-lg border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
      <div>
        <label
          htmlFor="sample"
          className="mb-1 block text-sm font-medium text-zinc-700 dark:text-zinc-300"
        >
          Sample window
        </label>
        <select
          id="sample"
          value={selectValue}
          onChange={(e) => onSelectSample(e.target.value)}
          className="w-full rounded-md border border-zinc-300 bg-white px-3 py-2 text-sm dark:border-zinc-700 dark:bg-zinc-950"
        >
          {samples.map((s) => (
            <option key={s.id} value={s.id}>
              {s.label} (synthetic)
            </option>
          ))}
          {selectedSampleId === null ? (
            <option value="__custom__">Custom window (uploaded / pasted)</option>
          ) : null}
        </select>
      </div>

      <div className="flex flex-col gap-2">
        <label className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
          …or load your own window
        </label>
        <input
          type="file"
          accept="application/json,.json"
          onChange={onFile}
          className="block w-full text-sm text-zinc-600 file:mr-3 file:rounded-md file:border-0 file:bg-zinc-100 file:px-3 file:py-1.5 file:text-sm file:font-medium hover:file:bg-zinc-200 dark:text-zinc-400 dark:file:bg-zinc-800 dark:hover:file:bg-zinc-700"
        />
        <textarea
          value={pasted}
          onChange={(e) => setPasted(e.target.value)}
          placeholder='Paste JSON: {"ecg":[…512…],"ppg":[…512…],"acc":[[x,y,z]…512…]}'
          rows={3}
          className="w-full rounded-md border border-zinc-300 bg-white px-3 py-2 font-mono text-xs dark:border-zinc-700 dark:bg-zinc-950"
        />
        <button
          type="button"
          onClick={() => ingest(pasted, "pasted window")}
          disabled={pasted.trim().length === 0}
          className="self-start rounded-md border border-zinc-300 px-3 py-1.5 text-sm font-medium hover:bg-zinc-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-zinc-700 dark:hover:bg-zinc-800"
        >
          Load pasted JSON
        </button>
      </div>

      <div className="flex flex-wrap gap-2 border-t border-zinc-200 pt-3 dark:border-zinc-800">
        <button
          type="button"
          onClick={onRunPrediction}
          disabled={predicting}
          className="rounded-md bg-zinc-900 px-4 py-2 text-sm font-semibold text-white hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-50 dark:bg-zinc-100 dark:text-zinc-900 dark:hover:bg-zinc-300"
        >
          {predicting ? "Predicting…" : "Run prediction"}
        </button>
        <button
          type="button"
          onClick={onExplain}
          disabled={!canExplain || reporting}
          title={canExplain ? undefined : "Run a prediction first"}
          className="rounded-md border border-zinc-300 px-4 py-2 text-sm font-semibold hover:bg-zinc-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-zinc-700 dark:hover:bg-zinc-800"
        >
          {reporting ? "Explaining…" : "Explain with Claude"}
        </button>
      </div>
    </section>
  );
}
