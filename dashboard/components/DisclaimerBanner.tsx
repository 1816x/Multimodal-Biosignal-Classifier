import { DISCLAIMER } from "@/lib/constants";

/**
 * Always-on educational disclaimer, mounted globally in the root layout. This is
 * a Phase 0 hard requirement: the not-a-medical-tool warning must be impossible
 * to miss and must not depend on any API response. Text mirrors the API's
 * canonical DISCLAIMER (api/src/biosignal_api/__init__.py).
 */
export default function DisclaimerBanner() {
  return (
    <div
      role="alert"
      className="border-b border-amber-300 bg-amber-100 px-4 py-2 text-center text-sm text-amber-900 dark:border-amber-800/60 dark:bg-amber-950/60 dark:text-amber-200"
    >
      <span aria-hidden="true">⚠️ </span>
      <strong>Educational prototype — NOT a medical device.</strong>{" "}
      <span className="opacity-90">{DISCLAIMER}</span>
    </div>
  );
}
