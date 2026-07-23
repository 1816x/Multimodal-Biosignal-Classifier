import { proxy } from "@/lib/proxy";

// Always run at request time so API_BASE_URL is read from the runtime env and
// the health check is never prerendered/cached.
export const dynamic = "force-dynamic";

export function GET() {
  return proxy("/health", { method: "GET" });
}
