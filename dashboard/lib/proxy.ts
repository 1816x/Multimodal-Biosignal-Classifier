/**
 * Server-side proxy helper for the Next.js Route Handlers under app/api/*.
 *
 * The browser only ever talks to this Next.js app (same origin); these handlers
 * forward to the FastAPI service. This keeps the upstream URL server-only (via
 * API_BASE_URL, deliberately NOT a NEXT_PUBLIC_ var) and avoids CORS entirely —
 * so the Python API needs no changes.
 *
 * A connection failure to the backend is normalized to HTTP 503 with the same
 * {"detail": "..."} envelope FastAPI uses, so the client's existing 503 path
 * also covers "backend offline".
 *
 * Server-only module. Educational prototype — NOT a medical device.
 */

const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

/** The upstream FastAPI base URL, read at request time (no trailing slash). */
export function apiBaseUrl(): string {
  const raw = process.env.API_BASE_URL ?? DEFAULT_API_BASE_URL;
  return raw.replace(/\/+$/, "");
}

/**
 * Forward a request to the FastAPI service and relay its status + JSON body
 * verbatim. On a connection failure, return 503 with a FastAPI-style detail.
 */
export async function proxy(path: string, init: RequestInit): Promise<Response> {
  const base = apiBaseUrl();
  let upstream: Response;
  try {
    upstream = await fetch(`${base}${path}`, init);
  } catch {
    return Response.json(
      { detail: `Backend unreachable — is the API running at ${base}? (set API_BASE_URL)` },
      { status: 503 },
    );
  }
  const body = await upstream.text();
  return new Response(body, {
    status: upstream.status,
    headers: {
      "content-type": upstream.headers.get("content-type") ?? "application/json",
    },
  });
}
