import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import PredictionCard from "@/components/PredictionCard";
import type { PredictionResponse, Sourced } from "@/lib/types";

const base: PredictionResponse = {
  predicted_class: "cycling",
  confidence: 0.65,
  relevant_segment: [10, 42],
  disclaimer: "x",
};

describe("PredictionCard", () => {
  it("formats confidence as a percent and shows the class and segment", () => {
    const result: Sourced<PredictionResponse> = { source: "live", data: base };
    render(<PredictionCard result={result} />);
    expect(screen.getByText("Cycling")).toBeInTheDocument();
    expect(screen.getByText("65%")).toBeInTheDocument();
    expect(screen.getByText(/samples 10.*42/)).toBeInTheDocument();
  });

  it("shows no demo badge for a live result", () => {
    const result: Sourced<PredictionResponse> = { source: "live", data: base };
    render(<PredictionCard result={result} />);
    expect(screen.queryByText("Demo")).not.toBeInTheDocument();
  });

  it("shows a demo badge for a canned result", () => {
    const result: Sourced<PredictionResponse> = {
      source: "demo",
      data: base,
      reason: "canned",
    };
    render(<PredictionCard result={result} />);
    expect(screen.getByText("Demo")).toBeInTheDocument();
  });
});
