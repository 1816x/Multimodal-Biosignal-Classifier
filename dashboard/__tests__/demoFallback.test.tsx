import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

// Mock the API client: health is live, but /predict returns 503 (untrained model).
// Keep the real ApiError class so DashboardClient's `instanceof` checks hold.
vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    getHealth: vi.fn().mockResolvedValue({ status: "ok", version: "test" }),
    postPredict: vi.fn().mockRejectedValue(new actual.ApiError(503, "No model checkpoint")),
    postReport: vi.fn(),
  };
});

import DashboardClient from "@/components/DashboardClient";

afterEach(() => {
  vi.clearAllMocks();
});

describe("DashboardClient demo fallback", () => {
  it("shows a badged canned prediction with a reason when /predict returns 503", async () => {
    const user = userEvent.setup();
    render(<DashboardClient />);

    await user.click(screen.getByRole("button", { name: /run prediction/i }));

    // Canned demo prediction for the default sample (cycling) is shown, badged.
    expect(await screen.findByText("Demo")).toBeInTheDocument();
    expect(screen.getByText("Cycling")).toBeInTheDocument();
    // The reason for the fallback is surfaced honestly.
    expect(screen.getByText(/canned demo prediction/i)).toBeInTheDocument();
  });
});
