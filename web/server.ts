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
};

const config = loadConfig(process.env);
const files = new Map<string, StaticFile>();
for (const name of readdirSync("public")) {
  const type = TYPES[extname(name)];
  if (!type) continue; // only known file types are ever served
  files.set(name === "index.html" ? "/" : `/${name}`, { body: readFileSync(join("public", name)), type });
}

const handler = createHandler({
  config,
  store: new UsageStore(config.dbPath, config),
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
