# Web demo: setup checklist (Cloudflare Turnstile + Railway)

Never paste secret values into the chat or into a file in the repo.

## 1. Cloudflare Turnstile (free)
- [ ] Cloudflare dashboard -> Turnstile -> Add widget. Mode: Managed. Hostnames: the Railway
      domain (step 2) and `localhost` for tests.
- [ ] Note the **site key** (public) and the **secret key** (private).
- [ ] For my first test only: Cloudflare's dummy keys exist for local runs; the real keys
      are needed on Railway.

## 2. Railway service
- [ ] New project, region **EU West (Amsterdam)**, deploy from this GitHub repo, **root
      directory `web`**.
- [ ] Build command `npm ci && npm run build`; start command `npm start` (Node 24+; set
      `NODE_VERSION=24` if the builder picks an older one).
- [ ] Add a **volume** mounted at `/data` (SQLite file `/data/usage.db`).
- [ ] Generate a public domain (Settings -> Networking). Health check path `/health`.
- [ ] Usage limits (Settings -> Usage): soft limit about 5 USD, hard limit about 10 USD
      (the hard limit takes the service offline).
- [ ] One replica only (the counters are in one SQLite file).

## 3. Variables (Railway service -> Variables)
| Name | Value |
|---|---|
| `LIVEKIT_URL` | `wss://praxis-telefon-demo-jmda6zh0.livekit.cloud` |
| `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` | a **new** API key pair from LiveKit Cloud (Settings -> Keys), named `web-token` so it can be revoked alone |
| `TURNSTILE_SITE_KEY` / `TURNSTILE_SECRET_KEY` | from step 1 |
| `IP_HASH_KEY` | random, 32+ characters (e.g. `openssl rand -hex 24`), not reused anywhere |
| `PUBLIC_ORIGIN` | `https://<your-railway-domain>` (no trailing slash, no path) |
| `CONTACT_EMAIL` | the email address shown on the page |
| `DB_PATH` | `/data/usage.db` |
| `TRUSTED_PROXY_HOPS` | `1` (check after the first test: the address in `X-Forwarded-For`) |

Optional (defaults in brackets): `WEB_MINUTES_PER_DAY` [20], `WEB_MINUTES_PER_MONTH` [60],
`WEB_CALL_SECONDS` [300], `VISITOR_TOKENS_PER_DAY` [3].

## 4. Then tell Claude
"Turnstile and Railway are set up." Claude then asks before the joint deploy (agent from
`cf56a86` + web) and the first browser test.
