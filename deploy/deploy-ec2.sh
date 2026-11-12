#!/bin/bash
# Runs ON chronicle-host (sent by deploy/publish-ec2.sh through SSM Run
# Command). Pulls IMAGE, reads settings from SSM Parameter Store under
# /chronicle/, (re)starts the container, fills the index if it is empty, and
# installs a daily refresh of the live sources.
#
#   IMAGE=<ecr-uri>:<tag> bash deploy-ec2.sh
set -euo pipefail
: "${IMAGE:?IMAGE is required}"
REGION=${REGION:-us-east-1}
ENVF=/opt/chronicle/app.env
mkdir -p /opt/chronicle

echo "== 1. settings from /chronicle/"
umask 077
aws ssm get-parameters-by-path --region "$REGION" --path /chronicle/ --with-decryption \
  --query 'Parameters[].[Name,Value]' --output text |
  while IFS=$'\t' read -r name value; do printf '%s=%s\n' "${name#/chronicle/}" "$value"; done > "$ENVF.new"
for k in GROQ_API_KEY NEON_DATABASE_URL EDGAR_USER_AGENT; do
  grep -q "^$k=." "$ENVF.new" || { echo "FATAL: parameter /chronicle/$k is missing"; rm -f "$ENVF.new"; exit 1; }
done
grep -q '^LLM_PROVIDER=' "$ENVF.new" || echo 'LLM_PROVIDER=groq' >> "$ENVF.new"
mv "$ENVF.new" "$ENVF"
echo "settings: $(cut -d= -f1 "$ENVF" | tr '\n' ' ')"

echo "== 2. pull $IMAGE"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${IMAGE%%/*}" >/dev/null
docker pull -q "$IMAGE"

echo "== 3. (re)start the container; indexes persist in the chronicle_state volume"
docker network inspect chronicle-net >/dev/null 2>&1 || docker network create chronicle-net >/dev/null
docker rm -f chronicle >/dev/null 2>&1 || true
# 8501 is published on loopback only (for the health check below); the
# internet reaches the app through Caddy on 443.
docker run -d --name chronicle --restart unless-stopped --network chronicle-net \
  -p 127.0.0.1:8501:8501 --env-file "$ENVF" -v chronicle_state:/state \
  --memory 1700m --log-opt max-size=20m --log-opt max-file=3 \
  "$IMAGE" >/dev/null
up=0
for i in $(seq 1 60); do
  if curl -fs -m 3 http://localhost:8501/_stcore/health >/dev/null; then up=1; break; fi
  sleep 3
done
if [ "$up" != 1 ]; then
  echo "FATAL: Streamlit did not become healthy in 180 s. Last log lines:"
  docker logs --tail 40 chronicle 2>&1
  exit 1
fi
echo "streamlit healthy"

echo "== 3b. HTTPS: Caddy with an automatic Let's Encrypt certificate"
# <a-b-c-d>.sslip.io resolves to the Elastic IP, so the hostname is stable for
# the life of the EIP. Caddy renews the cert itself and proxies WebSockets.
TOKEN=$(curl -s -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' http://169.254.169.254/latest/api/token)
PUBLIC_IP=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/public-ipv4)
SITE="${PUBLIC_IP//./-}.sslip.io"
cat > /opt/chronicle/Caddyfile <<CADDY
$SITE {
	encode zstd gzip
	header {
		Strict-Transport-Security "max-age=31536000"
		X-Content-Type-Options nosniff
		Referrer-Policy strict-origin-when-cross-origin
		-Server
	}
	reverse_proxy chronicle:8501
}
CADDY
if docker ps --format '{{.Names}}' | grep -qx caddy; then
  docker exec caddy caddy reload --config /etc/caddy/Caddyfile >/dev/null
else
  docker rm -f caddy >/dev/null 2>&1 || true
  docker run -d --name caddy --restart unless-stopped --network chronicle-net \
    -p 80:80 -p 443:443 -v /opt/chronicle/Caddyfile:/etc/caddy/Caddyfile:ro \
    -v caddy_data:/data -v caddy_config:/config \
    --memory 128m --log-opt max-size=10m --log-opt max-file=3 \
    caddy:2.8.4 >/dev/null
fi
cert=0
for i in $(seq 1 40); do
  if curl -fs -m 5 -o /dev/null "https://$SITE/_stcore/health"; then cert=1; break; fi
  sleep 3
done
if [ "$cert" != 1 ]; then
  echo "FATAL: https://$SITE did not answer with a valid certificate. Caddy log:"
  docker logs --tail 30 caddy 2>&1
  exit 1
fi
echo "https://$SITE serving with a valid certificate"

echo "== 4. fill the index now, so no visitor waits for it"
docker exec chronicle python -c '
from indexer import get_index_stats
from pipeline import ChroniclePipeline
if get_index_stats()["vector_chunks"] == 0:
    for r in ChroniclePipeline().run_live_data_agent("latest SEC filings and financial news", force=True):
        print(" ", r.source_id, r.status, r.message)
print("index:", get_index_stats())
'

echo "== 5. daily refresh of the live sources (06:17 UTC)"
cat > /etc/cron.d/chronicle-refresh <<'CRON'
17 6 * * * root docker exec chronicle python -c "from pipeline import ChroniclePipeline; [print(r.source_id, r.status, r.message) for r in ChroniclePipeline().run_live_data_agent('latest SEC filings and financial news')]" >> /var/log/chronicle-refresh.log 2>&1
CRON
chmod 644 /etc/cron.d/chronicle-refresh

docker image prune -f >/dev/null
docker stats --no-stream --format '{{.Name}} {{.MemUsage}}' chronicle
echo "== done"
