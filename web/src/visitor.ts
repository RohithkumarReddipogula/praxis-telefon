// The visitor key: a keyed hash (HMAC-SHA256) of the client address, never the address.
// A plain hash of an IP address is easy to reverse (there are few addresses).

import { createHmac } from "node:crypto";
import { isIP } from "node:net";

/**
 * The client address from X-Forwarded-For: the entry the trusted proxy appended, counted
 * from the right (a client can prepend fake entries, not append). null if unusable.
 */
export function clientIp(headers: Headers, trustedHops: number): string | null {
  const raw = headers.get("x-forwarded-for");
  if (!raw) return null;
  const parts = raw.split(",").map((part) => part.trim()).filter(Boolean);
  if (parts.length < trustedHops) return null;
  const entry = parts[parts.length - trustedHops];
  return entry && isIP(entry) ? entry : null;
}

function canonical(ip: string): string {
  if (isIP(ip) === 6) return new URL(`http://[${ip}]/`).hostname; // one spelling per address
  return ip;
}

/** 16 hex characters; no address means one shared (conservative) key. */
export function visitorKey(ip: string | null, key: string): string {
  const subject = ip === null ? "unknown" : canonical(ip);
  return createHmac("sha256", key).update(subject).digest("hex").slice(0, 16);
}
