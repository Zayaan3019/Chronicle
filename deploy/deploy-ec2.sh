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
docker rm -f chronicle >/dev/null 2>&1 || true
docker run -d --name chronicle --restart unless-stopped \
  -p 8501:8501 --env-file "$ENVF" -v chronicle_state:/state \
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
