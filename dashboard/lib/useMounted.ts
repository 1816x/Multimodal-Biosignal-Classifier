"use client";

import { useSyncExternalStore } from "react";

const emptySubscribe = () => () => {};

/**
 * SSR-safe "are we on the client yet?" flag. Returns false during server render
 * and the hydration snapshot, true once running in the browser — without a
 * setState-in-effect (which React 19 / Next 16 lint flags). Used to defer
 * browser-only widgets (Recharts) past hydration to avoid mismatches.
 */
export function useMounted(): boolean {
  return useSyncExternalStore(
    emptySubscribe,
    () => true,
    () => false,
  );
}
