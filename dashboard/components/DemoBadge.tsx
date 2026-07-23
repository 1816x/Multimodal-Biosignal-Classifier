/**
 * Reusable marker for content that is NOT a live model output — synthetic input
 * signals or canned demo predictions/reports. The honesty rule: any demo or
 * synthetic data shown in the UI must carry this badge.
 */
export default function DemoBadge({
  children = "DEMO",
  title,
}: {
  children?: React.ReactNode;
  title?: string;
}) {
  return (
    <span
      title={title}
      className="inline-flex items-center gap-1 rounded-full border border-amber-400 bg-amber-100 px-2 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-amber-800 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-300"
    >
      {children}
    </span>
  );
}
