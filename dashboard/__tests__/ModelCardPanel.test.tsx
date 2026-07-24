import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import ModelCardPanel from "@/components/ModelCardPanel";

describe("ModelCardPanel", () => {
  it("surfaces the honest headline metrics", () => {
    render(<ModelCardPanel />);
    expect(screen.getByText(/About this model/i)).toBeInTheDocument();
    expect(screen.getByText("65%")).toBeInTheDocument(); // multimodal test accuracy
    expect(screen.getByText("37%")).toBeInTheDocument(); // ECG-only test accuracy
    expect(screen.getByText(/walking/i)).toBeInTheDocument(); // a weakest class
    expect(screen.getByText(/PPG-DaLiA/i)).toBeInTheDocument();
  });
});
