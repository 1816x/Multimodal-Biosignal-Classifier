import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import DisclaimerBanner from "@/components/DisclaimerBanner";

describe("DisclaimerBanner", () => {
  it("renders the educational not-a-medical-device warning", () => {
    render(<DisclaimerBanner />);
    expect(screen.getByText(/NOT a medical device/i)).toBeInTheDocument();
  });

  it("renders the canonical disclaimer text", () => {
    render(<DisclaimerBanner />);
    expect(screen.getByText(/illustrative only/i)).toBeInTheDocument();
  });
});
