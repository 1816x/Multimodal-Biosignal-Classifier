"use client";

import { useEffect, useState } from "react";
import InputControls from "@/components/InputControls";
import PredictionCard from "@/components/PredictionCard";
import ReportPanel from "@/components/ReportPanel";
import SignalPanel from "@/components/SignalPanel";
import StatusBanner, { type HealthState } from "@/components/StatusBanner";
import { ApiError, getHealth, postPredict, postReport } from "@/lib/api";
import { getDemoResult } from "@/lib/fixtures/demoResults";
import { DEFAULT_SAMPLE_ID, getSample, SAMPLES } from "@/lib/fixtures/samples";
import type {
  PredictionRequest,
  PredictionResponse,
  ReportResponse,
  Sourced,
} from "@/lib/types";

type Notice = { kind: "error" | "info"; text: string };

const DEFAULT_SAMPLE = getSample(DEFAULT_SAMPLE_ID)!;

export default function DashboardClient() {
  const [health, setHealth] = useState<HealthState>("checking");
  const [apiVersion, setApiVersion] = useState<string | undefined>();

  // Current input window + provenance for the synthetic badge.
  const [activeSampleId, setActiveSampleId] = useState<string | null>(DEFAULT_SAMPLE_ID);
  const [input, setInput] = useState<PredictionRequest>(DEFAULT_SAMPLE.input);
  const [inputLabel, setInputLabel] = useState<string>(DEFAULT_SAMPLE.label);
  const [inputSynthetic, setInputSynthetic] = useState<boolean>(true);
  const [inputNote, setInputNote] = useState<string | undefined>(DEFAULT_SAMPLE.note);

  const [prediction, setPrediction] = useState<Sourced<PredictionResponse> | null>(null);
  const [report, setReport] = useState<Sourced<ReportResponse> | null>(null);
  const [predicting, setPredicting] = useState(false);
  const [reporting, setReporting] = useState(false);
  const [notice, setNotice] = useState<Notice | null>(null);

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((h) => {
        if (cancelled) return;
        setHealth("live");
        setApiVersion(h.version);
      })
      .catch(() => {
        if (!cancelled) setHealth("offline");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function resetResults() {
    setPrediction(null);
    setReport(null);
    setNotice(null);
  }

  function selectSample(id: string) {
    const s = getSample(id);
    if (!s) return;
    setActiveSampleId(id);
    setInput(s.input);
    setInputLabel(s.label);
    setInputSynthetic(true);
    setInputNote(s.note);
    resetResults();
  }

  function loadCustom(custom: PredictionRequest, label: string) {
    setActiveSampleId(null);
    setInput(custom);
    setInputLabel(label);
    setInputSynthetic(false);
    setInputNote(undefined);
    resetResults();
  }

  // --- Prediction -----------------------------------------------------------
  async function runPrediction() {
    setPredicting(true);
    setNotice(null);
    setReport(null);
    try {
      const data = await postPredict(input);
      setPrediction({ source: "live", data });
    } catch (e) {
      handlePredictError(e);
    } finally {
      setPredicting(false);
    }
  }

  function handlePredictError(e: unknown) {
    const err = e instanceof ApiError ? e : new ApiError(-1, String(e));
    const demo = activeSampleId ? getDemoResult(activeSampleId) : undefined;

    if (err.status === 503 || err.status === 0) {
      if (demo) {
        const reason =
          err.status === 0
            ? "Dashboard/API unreachable — showing a canned demo prediction."
            : "Live model unavailable (503) — showing a canned demo prediction.";
        setPrediction({ source: "demo", data: demo.prediction, reason });
        setNotice({ kind: "info", text: reason });
      } else {
        setPrediction(null);
        setNotice({
          kind: "error",
          text: "No live model (503/offline) and no canned demo exists for a custom window. Pick a bundled sample to see a demo result.",
        });
      }
      return;
    }
    if (err.status === 422) {
      setNotice({ kind: "error", text: `Invalid input (422): ${err.detail}` });
      return;
    }
    setNotice({ kind: "error", text: `Prediction failed (${err.status}): ${err.detail}` });
  }

  // --- Report ---------------------------------------------------------------
  async function explain() {
    setReporting(true);
    setNotice(null);
    try {
      const data = await postReport(input);
      setReport({ source: "live", data });
      // Keep the prediction card consistent with the live report.
      setPrediction({
        source: "live",
        data: {
          predicted_class: data.predicted_class,
          confidence: data.confidence,
          relevant_segment: data.relevant_segment,
          disclaimer: data.disclaimer,
        },
      });
    } catch (e) {
      handleReportError(e);
    } finally {
      setReporting(false);
    }
  }

  function handleReportError(e: unknown) {
    const err = e instanceof ApiError ? e : new ApiError(-1, String(e));
    const demo = activeSampleId ? getDemoResult(activeSampleId) : undefined;

    if (err.status === 503 || err.status === 0 || err.status === 502) {
      if (demo) {
        const reason =
          err.status === 502
            ? "Claude API upstream failed (502) — showing a canned demo report."
            : err.status === 0
              ? "Dashboard/API unreachable — showing a canned demo report."
              : "Report layer unavailable (503) — showing a canned demo report.";
        setReport({ source: "demo", data: demo.report, reason });
        setNotice({ kind: "info", text: reason });
      } else {
        setReport(null);
        setNotice({
          kind: "error",
          text: `Report unavailable (${err.status}) and no canned demo exists for a custom window: ${err.detail}`,
        });
      }
      return;
    }
    if (err.status === 422) {
      setNotice({ kind: "error", text: `Invalid input (422): ${err.detail}` });
      return;
    }
    setNotice({ kind: "error", text: `Report failed (${err.status}): ${err.detail}` });
  }

  const segment = prediction ? prediction.data.relevant_segment : null;

  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-6">
      <header className="mb-4">
        <h1 className="text-xl font-bold tracking-tight sm:text-2xl">
          Multimodal Biosignal Classifier
        </h1>
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          Visualize an ECG + PPG + accelerometer window, the model&apos;s activity
          prediction, and a Claude-generated report.
        </p>
      </header>

      <div className="mb-4">
        <StatusBanner health={health} apiVersion={apiVersion} />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-5">
        {/* Left column: controls + results */}
        <div className="flex flex-col gap-4 lg:col-span-2">
          <InputControls
            samples={SAMPLES}
            selectedSampleId={activeSampleId}
            onSelectSample={selectSample}
            onLoadCustom={loadCustom}
            onError={(text) => setNotice({ kind: "error", text })}
            onRunPrediction={runPrediction}
            onExplain={explain}
            canExplain={prediction !== null}
            predicting={predicting}
            reporting={reporting}
          />

          <div className="flex items-center gap-2 text-xs text-zinc-500 dark:text-zinc-400">
            <span className="font-medium text-zinc-700 dark:text-zinc-300">
              Input: {inputLabel}
            </span>
            {inputSynthetic ? (
              <span className="rounded bg-zinc-100 px-1.5 py-0.5 dark:bg-zinc-800">
                synthetic
              </span>
            ) : (
              <span className="rounded bg-zinc-100 px-1.5 py-0.5 dark:bg-zinc-800">
                custom
              </span>
            )}
          </div>
          {inputNote ? (
            <p className="-mt-2 text-xs text-zinc-500 dark:text-zinc-400">{inputNote}</p>
          ) : null}

          {notice ? (
            <div
              className={`rounded-md border px-3 py-2 text-sm ${
                notice.kind === "error"
                  ? "border-red-300 bg-red-50 text-red-800 dark:border-red-800 dark:bg-red-950/50 dark:text-red-300"
                  : "border-sky-300 bg-sky-50 text-sky-800 dark:border-sky-800 dark:bg-sky-950/50 dark:text-sky-300"
              }`}
            >
              {notice.text}
            </div>
          ) : null}

          {prediction ? <PredictionCard result={prediction} /> : null}
          {report ? <ReportPanel result={report} /> : null}
        </div>

        {/* Right column: signal plots */}
        <div className="lg:col-span-3">
          <SignalPanel input={input} segment={segment} />
        </div>
      </div>
    </main>
  );
}
