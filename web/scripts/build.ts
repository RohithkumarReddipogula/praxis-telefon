// Bundles livekit-client into public/livekit-client.js (the page imports it from there).
import { build } from "esbuild";

await build({
  entryPoints: ["node_modules/livekit-client/dist/livekit-client.esm.mjs"],
  outfile: "public/livekit-client.js",
  bundle: true,
  format: "esm",
  minify: true,
  target: "es2022",
  legalComments: "none",
});
console.log("built public/livekit-client.js");
