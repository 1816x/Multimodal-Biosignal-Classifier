import { proxy } from "@/lib/proxy";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  const body = await request.text();
  return proxy("/predict", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body,
  });
}
