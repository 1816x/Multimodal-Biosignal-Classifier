import type { Metadata } from "next";
import "./globals.css";
import DisclaimerBanner from "@/components/DisclaimerBanner";

export const metadata: Metadata = {
  title: "Multimodal Biosignal Classifier — Dashboard",
  description:
    "Educational prototype (NOT a medical device): visualize ECG + PPG + accelerometer " +
    "windows, the model's activity prediction, and a Claude-generated report.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className="h-full antialiased">
      <body className="flex min-h-full flex-col bg-zinc-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
        <DisclaimerBanner />
        <div className="flex-1">{children}</div>
      </body>
    </html>
  );
}
