#!/bin/bash
# Build the image, push it to ECR, and deploy it on chronicle-host through
# SSM Run Command. Run from the repository root with AWS credentials:
#
#   bash deploy/publish-ec2.sh
#
# One-time prerequisites: the host exists (infra/projects/chronicle) and the
# settings are in SSM Parameter Store (deploy/put_params.py).
set -euo pipefail

REGION=us-east-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REPO="$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/chronicle"
TAG=$(git rev-parse --short=12 HEAD)
[ -z "$(git status --porcelain -- . ':!deploy')" ] || { echo "uncommitted changes; commit first so the tag means something"; exit 1; }
INSTANCE_ID=${INSTANCE_ID:-$(aws ec2 describe-instances --region $REGION \
  --filters Name=tag:Name,Values=chronicle-host Name=instance-state-name,Values=running \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)}
[ "$INSTANCE_ID" != None ] || { echo "no running chronicle-host; apply infra/projects/chronicle first"; exit 1; }

echo "== image $REPO:$TAG"
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin "${REPO%%/*}" >/dev/null
docker build -q -t "$REPO:$TAG" -t "$REPO:latest" .
docker push -q "$REPO:$TAG" && docker push -q "$REPO:latest"

echo "== deploy on $INSTANCE_ID"
SCRIPT=$(base64 -w0 deploy/deploy-ec2.sh)
PARAMS=$(mktemp)
python -c 'import json,sys; print(json.dumps({"commands": [sys.argv[1]], "executionTimeout": ["1800"]}))' \
  "echo $SCRIPT | base64 -d > /tmp/deploy-ec2.sh && IMAGE='$REPO:$TAG' bash /tmp/deploy-ec2.sh 2>&1" > "$PARAMS"
CID=$(aws ssm send-command --region $REGION --instance-ids "$INSTANCE_ID" --document-name AWS-RunShellScript \
  --comment "chronicle deploy $TAG" --parameters "file://$PARAMS" --query Command.CommandId --output text)
rm -f "$PARAMS"
echo "command $CID"
for i in $(seq 1 180); do
  STATUS=$(aws ssm get-command-invocation --region $REGION --command-id "$CID" --instance-id "$INSTANCE_ID" \
    --query Status --output text 2>/dev/null || echo Pending)
  case "$STATUS" in Success|Failed|Cancelled|TimedOut) break ;; esac
  sleep 10
done
aws ssm get-command-invocation --region $REGION --command-id "$CID" --instance-id "$INSTANCE_ID" \
  --query StandardOutputContent --output text
echo "status: $STATUS"
[ "$STATUS" = Success ]
