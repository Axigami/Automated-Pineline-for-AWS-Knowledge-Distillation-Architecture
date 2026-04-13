param (
    [string]$AwsRegion = "ap-southeast-2",
    [string]$ImageName = "distillation-finetune",
    [string]$ImageTag  = "latest"
)

# ── Get Account ID ────────────────────────────────────────────────────────────
$AwsAccountId = (aws sts get-caller-identity --query Account --output text)
if (-not $AwsAccountId) {
    Write-Host "Cannot get AWS Account ID. Please check AWS CLI profile." -ForegroundColor Red
    exit 1
}

$EcrUrl       = "${AwsAccountId}.dkr.ecr.${AwsRegion}.amazonaws.com"
$RepositoryUri = "${EcrUrl}/${ImageName}"

Write-Host "=============================================" -ForegroundColor Cyan
Write-Host " distillation-finetune  Build & Push to ECR" -ForegroundColor Cyan
Write-Host "=============================================" -ForegroundColor Cyan
Write-Host "Account ID  : $AwsAccountId"
Write-Host "Region      : $AwsRegion"
Write-Host "Image URL   : ${RepositoryUri}:${ImageTag}"
Write-Host "Script      : FineTuneTeacher.py"
Write-Host "=============================================" -ForegroundColor Cyan

# ── [1/5] ECR Login ───────────────────────────────────────────────────────────
Write-Host "`n[1/5] Logging into Amazon ECR..." -ForegroundColor Yellow
aws ecr get-login-password --region $AwsRegion | docker login --username AWS --password-stdin $EcrUrl
if ($LASTEXITCODE -ne 0) {
    Write-Host "ECR login failed!" -ForegroundColor Red
    exit 1
}

# ── [2/5] Create repo if needed ───────────────────────────────────────────────
Write-Host "`n[2/5] Checking ECR Repository '${ImageName}'..." -ForegroundColor Yellow
$RepoCheck = aws ecr describe-repositories --repository-names $ImageName --region $AwsRegion 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Repository does not exist. Creating..." -ForegroundColor Yellow
    aws ecr create-repository --repository-name $ImageName --region $AwsRegion
} else {
    Write-Host "Repository exists." -ForegroundColor Green
}

# ── [3/5] Build ───────────────────────────────────────────────────────────────
Write-Host "`n[3/5] Building Docker Image (${ImageName}:${ImageTag})..." -ForegroundColor Yellow

# Ensure FineTuneTeacher.py is in the finetune/ folder
if (-not (Test-Path ".\finetune\FineTuneTeacher.py")) {
    Write-Host "ERROR: .\finetune\FineTuneTeacher.py not found!" -ForegroundColor Red
    exit 1
}

docker build --provenance=false -t ${ImageName}:${ImageTag} .\finetune
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker build failed!" -ForegroundColor Red
    exit 1
}

# ── [4/5] Tag ─────────────────────────────────────────────────────────────────
Write-Host "`n[4/5] Tagging image..." -ForegroundColor Yellow
docker tag ${ImageName}:${ImageTag} ${RepositoryUri}:${ImageTag}

# ── [5/5] Push ────────────────────────────────────────────────────────────────
Write-Host "`n[5/5] Pushing image to ECR..." -ForegroundColor Yellow
docker push ${RepositoryUri}:${ImageTag}
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker push failed!" -ForegroundColor Red
    exit 1
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host "`n=============================================" -ForegroundColor Green
Write-Host " COMPLETE!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green
Write-Host "Image pushed to ECR:"
Write-Host "${RepositoryUri}:${ImageTag}" -ForegroundColor Green
Write-Host ""
Write-Host "Set env var này cho Lambda PrepareDistillationData:" -ForegroundColor Cyan
Write-Host "ECR_IMAGE_FINETUNE = ${RepositoryUri}:${ImageTag}" -ForegroundColor Yellow
