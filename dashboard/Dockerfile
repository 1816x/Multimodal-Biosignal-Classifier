# Dashboard (Next.js) production image — multi-stage, standalone output.
# Educational prototype — NOT a medical device.
# syntax=docker/dockerfile:1

FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

FROM node:22-alpine AS build
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
# API_BASE_URL is read server-side at request time (see lib/proxy.ts); override
# it in docker-compose to point at the api service.
ENV API_BASE_URL=http://api:8000

# Next.js standalone output: a minimal server + traced node_modules.
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static

EXPOSE 3000
CMD ["node", "server.js"]
