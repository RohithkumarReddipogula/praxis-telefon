# Web demo image for Railway (service root directory: web). The Python agent has its own
# Dockerfile in the repository root and deploys to LiveKit Cloud; this one is separate.
FROM node:24-slim

WORKDIR /app
ENV NODE_ENV=production

COPY package.json package-lock.json ./
# esbuild (bundler) is a runtime dependency here on purpose: the page bundle is built in the image
RUN npm ci

COPY . .
RUN npm run build

# Runs as root on purpose: Railway mounts the volume at /data owned by root, and a non-root
# user could not create /data/usage.db (docs.railway.com/volumes/reference).
EXPOSE 8080
CMD ["node", "server.ts"]
