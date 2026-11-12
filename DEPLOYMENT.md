# Deploying Chronicle

Chronicle runs as a Docker container on its own always-on EC2 host
(`chronicle-host`, t3.small), behind a CloudFront distribution that gives it
an HTTPS URL and carries Streamlit's WebSocket. The LLM is Groq's hosted API.
Embeddings and reranking run locally in the container (`BAAI/bge-small-en-v1.5`
and `cross-encoder/ms-marco-MiniLM-L-6-v2`, baked into the image). Durable
relational state (conversation memory, the guardrail audit log, live-source
tracking) lives in Neon Postgres.

## Why not Streamlit Community Cloud

Free Streamlit apps sleep after a stretch without visitors. A woken app starts
on an empty disk, so the first visitor sat through a sleep screen and then a
full reindex of the live sources. On EC2 the container restarts itself
(`--restart unless-stopped`), and both indexes (`CHROMA_PERSIST_DIR`,
`BM25_INDEX_PATH`) live on the `chronicle_state` Docker volume. They survive
restarts and redeploys, and are filled at deploy time rather than by a visitor.

## Layout

| Piece | Where |
|---|---|
| Image | `Dockerfile` (CPU-only PyTorch, models baked in, non-root user) |
| Host, CloudFront, IAM | `infra/projects/chronicle` (Terraform, uses `modules/ec2-docker-host`) |
| Settings | SSM Parameter Store `/chronicle/*` (`GROQ_API_KEY`, `NEON_DATABASE_URL` as SecureString) |
| Deploy | `deploy/publish-ec2.sh` (build, push to ECR, run `deploy/deploy-ec2.sh` on the host via SSM) |

The host has no SSH and no open ports except 8501. That port accepts traffic
only from CloudFront's origin-facing address ranges, so the instance cannot be
reached directly.

## First deployment

```bash
# 1. settings: copies ./.env into /chronicle/* (values are never printed)
python deploy/put_params.py

# 2. infrastructure (10 resources, nothing replaced)
cd ../infra/projects/chronicle && terraform init && terraform apply && cd -

# 3. image + container (wait ~3 min after step 2 for the host to finish booting)
bash deploy/publish-ec2.sh
```

Step 3 prints each stage: settings found, image pulled, Streamlit healthy, the
index filled from the live SEC EDGAR and GDELT sources, and memory in use.
`terraform output url` gives the public link.

## Every later deploy

Commit, then run `bash deploy/publish-ec2.sh`. The image is tagged with the
commit, and the indexes carry over.

## Keeping the data current

The deploy installs `/etc/cron.d/chronicle-refresh`, which runs the live-data
agent daily at 06:17 UTC. The agent also refreshes stale sources when a
question needs them. Every chunk in the index comes from those live sources;
nothing is bundled or generated.

## Verifying a deployment

1. Open the CloudFront URL. The sidebar should show `LLM (groq): ready`.
2. Ask a real question, e.g. "What did Apple report about iPhone revenue in its
   most recent 10-K?" It should return a cited, grounded answer.
3. Restart the container with
   `aws ssm send-command ... --parameters commands="docker restart chronicle"`,
   then confirm that no "Cold-start index population" section appears in the
   sidebar (the index survived), and that earlier conversation memory persists
   through Neon.
