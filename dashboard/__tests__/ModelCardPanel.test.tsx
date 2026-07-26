import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import ModelCardPanel from "@/components/ModelCardPanel";

describe("ModelCardPanel", () => {
  it("surfaces the honest headline metrics", () => {
    render(<ModelCardPanel />);
    expect(screen.getByText(/About this model/i)).toBeInTheDocument();
    expect(screen.getByText("78%")).toBeInTheDocument(); // multimodal test accuracy
    expect(screen.getByText("61%")).toBeInTheDocument(); // ECG-only test accuracy
    expect(screen.getByText(/working/i)).toBeInTheDocument(); // a weakest class
    expect(screen.getByText(/PPG-DaLiA/i)).toBeInTheDocument();
  });
});
