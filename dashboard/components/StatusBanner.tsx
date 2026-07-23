export type HealthState = "checking" | "live" | "offline";

/**
 * Global connectivity banner: shows whether the dashboard is talking to a live
 * API or running in demo mode (backend offline / unreachable). Purely
 * presentational — the health check itself lives in DashboardClient.
 */
export default function StatusBanner({
  health,
  apiVersion,
}: {
  health: HealthState;
  apiVersion?: string;
}) {
  const config: Record<HealthState, { dot: string; text: string; cls: string }> = {
    checking: {
      dot: "bg-zinc-400",
      text: "Checking API…",
      cls: "border-zinc-300 bg-zinc-100 text-zinc-700 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-300",
    },
    live: {
      dot: "bg-emerald-500",
      text: `Live API${apiVersion ? ` · v${apiVersion}` : ""}`,
      cls: "border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-300",
    },
    offline: {
      dot: "bg-amber-500",
      text: "Demo mode — API offline. Predictions and reports below are canned examples.",
      cls: "border-amber-300 bg-amber-50 text-amber-800 dark:border-amber-800 dark:bg-amber-950/50 dark:text-amber-300",
    },
  };
  const c = config[health];
  return (
    <div
      className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-sm ${c.cls}`}
    >
      <span className={`h-2.5 w-2.5 rounded-full ${c.dot}`} aria-hidden="true" />
      <span>{c.text}</span>
    </div>
  );
}
