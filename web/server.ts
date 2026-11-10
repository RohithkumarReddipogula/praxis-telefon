// Railway entrypoint: static page + POST /api/token. Run: node server.ts
import { randomBytes } from "node:crypto";
import { readFileSync, readdirSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join } from "node:path";
import { loadConfig } from "./src/config.ts";
import { createHandler, type StaticFile } from "./src/handler.ts";
import { signToken } from "./src/sign.ts";
import { UsageStore } from "./src/store.ts";
import { verifyTurnstile } from "./src/turnstile.ts";

const TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

// Startup is logged step by step so a stuck or failing start shows where in Railway's log.
function fatal(where: string, error: unknown): never {
  console.error(JSON.stringify({ event: "fatal", where, message: error instanceof Error ? error.message : String(error) }));
  process.exit(1);
}
console.log(JSON.stringify({ event: "starting", node: process.version }));

let config: ReturnType<typeof loadConfig>;
try {
  config = loadConfig(process.env);
} catch (error) {
  fatal("config", error);
}
console.log(JSON.stringify({ event: "config_ok", port: config.port, db: config.dbPath }));
const files = new Map<string, StaticFile>();
for (const name of readdirSync("public")) {
  const type = TYPES[extname(name)];
  if (!type) continue; // only known file types are ever served
  files.set(name === "index.html" ? "/" : `/${name}`, { body: readFileSync(join("public", name)), type });
}

let store: UsageStore;
try {
  store = new UsageStore(config.dbPath, config);
} catch (error) {
  fatal("usage_db", error);
}
console.log(JSON.stringify({ event: "db_ok" }));

const handler = createHandler({
  config,
  store,
  verifyCaptcha: (token, ip) => verifyTurnstile(token, config.turnstileSecret, ip),
  sign: (spec) => signToken(config, spec),
  log: (record) => console.log(JSON.stringify(record)), // counters and event names only
  now: () => new Date(),
  randomHex: (bytes) => randomBytes(bytes).toString("hex"),
  files,
});

createServer(async (req, res) => {
  const request = new Request(new URL(req.url ?? "/", config.publicOrigin), {
    method: req.method,
    headers: req.headers as Record<string, string>,
    body: req.method === "POST" ? await new Promise<string>((ok) => {
      let data = "";
      req.on("data", (c) => { if (data.length < 4096) data += c; });
      req.on("end", () => ok(data));
    }) : undefined,
  });
  const response = await handler(request);
  res.writeHead(response.status, Object.fromEntries(response.headers));
  res.end(Buffer.from(await response.arrayBuffer()));
}).listen(config.port, () => console.log(JSON.stringify({ event: "listening", port: config.port })));
