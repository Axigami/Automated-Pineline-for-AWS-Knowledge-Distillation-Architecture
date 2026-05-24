# ════════════════════════════════════════════════════════
#  Stage 1 — Builder
#  Cài dependencies + build Vite production bundle
# ════════════════════════════════════════════════════════
FROM node:20-alpine AS builder

WORKDIR /app

# Copy manifest trước để tận dụng Docker layer cache
COPY package.json package-lock.json ./

# Cài đúng version được lock (ci = không sinh ra lock file mới)
RUN npm ci

# Copy toàn bộ source code
COPY . .

# Build argument: cho phép truyền biến env lúc docker build
# Ví dụ: --build-arg VITE_SUPABASE_URL=https://xxx.supabase.co
ARG VITE_SUPABASE_URL
ARG VITE_SUPABASE_ANON_KEY
ARG VITE_API_GATEWAY_URL
ARG GEMINI_API_KEY
ARG APP_URL

ENV VITE_SUPABASE_URL=$VITE_SUPABASE_URL
ENV VITE_SUPABASE_ANON_KEY=$VITE_SUPABASE_ANON_KEY
ENV VITE_API_GATEWAY_URL=$VITE_API_GATEWAY_URL
ENV GEMINI_API_KEY=$GEMINI_API_KEY
ENV APP_URL=$APP_URL

RUN npm run build

# ════════════════════════════════════════════════════════
#  Stage 2 — Production
#  Nginx Alpine siêu nhẹ (~23 MB) phục vụ static files
# ════════════════════════════════════════════════════════
FROM nginx:1.27-alpine AS production

# Xoá config mặc định của nginx
RUN rm /etc/nginx/conf.d/default.conf

# Copy nginx config tùy chỉnh (SPA fallback)
COPY docker/nginx.conf /etc/nginx/conf.d/app.conf

# Copy build output từ stage 1
COPY --from=builder /app/dist /usr/share/nginx/html

# Expose port 80
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD wget -qO- http://localhost/healthz || exit 1

CMD ["nginx", "-g", "daemon off;"]
