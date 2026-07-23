import { afterEach, describe, expect, it, vi } from "vitest";
import { proxy } from "@/lib/proxy";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("proxy", () => {
  it("relays the upstream status and JSON body verbatim", async () => {
    const upstream = new Response(JSON.stringify({ status: "ok", version: "1" }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(upstream));

    const res = await proxy("/health", { method: "GET" });
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ status: "ok", version: "1" });
  });

  it("relays a non-2xx status (e.g. 503) unchanged", async () => {
    const upstream = new Response(JSON.stringify({ detail: "not trained" }), {
      status: 503,
      headers: { "content-type": "application/json" },
    });
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(upstream));

    const res = await proxy("/predict", { method: "POST", body: "{}" });
    expect(res.status).toBe(503);
    expect((await res.json()).detail).toBe("not trained");
  });

  it("maps a backend connection failure to 503", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("ECONNREFUSED")));

    const res = await proxy("/health", { method: "GET" });
    expect(res.status).toBe(503);
    expect((await res.json()).detail).toMatch(/unreachable/i);
  });
});
