// Server-side check of a Cloudflare Turnstile token. Fails closed: anything but a clear
// "success: true" from Cloudflare is a refusal.

const SITEVERIFY = "https://challenges.cloudflare.com/turnstile/v0/siteverify";

export async function verifyTurnstile(
  token: string | undefined,
  secret: string,
  ip: string | null,
  fetchFn: typeof fetch = fetch,
): Promise<boolean> {
  if (!token) return false;
  const form = new URLSearchParams({ secret, response: token });
  if (ip) form.set("remoteip", ip);
  try {
    const res = await fetchFn(SITEVERIFY, {
      method: "POST",
      headers: { "content-type": "application/x-www-form-urlencoded" },
      body: form.toString(),
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) return false;
    const data = (await res.json()) as { success?: unknown };
    return data.success === true;
  } catch {
    return false;
  }
}
