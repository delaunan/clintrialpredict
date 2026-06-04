#!/bin/bash

# Exit on error, undefined variable, or pipe failure
set -euo pipefail

# --- Configuration ---
PROJECT_ID="clintrial-predict-2025"
REGION="europe-west1"
IMAGE="europe-west1-docker.pkg.dev/clintrial-predict-2025/images/app-v01:latest"
API_SERVICE="clintrial-api"
UI_SERVICE="clintrial-ui"

# --- Helper Functions ---

get_api_url() {
    # Dynamically fetch the URL of the API service from Google Cloud
    local url
    url=$(gcloud run services describe "$API_SERVICE" --platform managed --region "$REGION" --format 'value(status.url)' 2>/dev/null || echo "")
    if [ -n "$url" ]; then
        echo "${url}/predict"
    else
        # Fallback to a safe placeholder if the service doesn't exist yet
        echo "PENDING_API_DEPLOYMENT"
    fi
}

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# --- Helper Functions ---

print_header() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}  Clinical Trial Predictor - Deployment Helper${NC}"
    echo -e "${BLUE}================================================================${NC}"
}

check_gcloud() {
    if command -v gcloud >/dev/null 2>&1; then
        echo -e "${GREEN}[OK] gcloud is installed.${NC}"
        return 0
    else
        echo -e "${RED}[MISSING] gcloud is not installed. Please install Google Cloud SDK.${NC}"
        return 1
    fi
}

check_docker() {
    if command -v docker >/dev/null 2>&1; then
        echo -e "${GREEN}[OK] Docker is installed.${NC}"
        if docker info >/dev/null 2>&1; then
            echo -e "${GREEN}[OK] Docker daemon is running.${NC}"
            return 0
        else
            echo -e "${RED}[ACTION NEEDED] Docker daemon is NOT running. Please start Docker Desktop.${NC}"
            return 1
        fi
    else
        echo -e "${RED}[MISSING] Docker is not installed.${NC}"
        return 1
    fi
}

check_project() {
    local current_project
    current_project=$(gcloud config get-value project 2>/dev/null || echo "NONE")
    if [ "$current_project" == "$PROJECT_ID" ]; then
        echo -e "${GREEN}[OK] Current gcloud project is $PROJECT_ID.${NC}"
        return 0
    else
        echo -e "${RED}[ACTION NEEDED] Current project is '$current_project', should be '$PROJECT_ID'.${NC}"
        return 1
    fi
}

check_auth() {
    local account
    account=$(gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>/dev/null || echo "")
    if [ -n "$account" ]; then
        echo -e "${GREEN}[OK] Active authenticated account: $account${NC}"
        return 0
    else
        echo -e "${RED}[ACTION NEEDED] No active gcloud account found.${NC}"
        return 1
    fi
}

check_docker_auth() {
    # This is harder to check definitively without a push, but we can check the config file
    if [ -f ~/.docker/config.json ] && grep -q "$REGION-docker.pkg.dev" ~/.docker/config.json 2>/dev/null; then
        echo -e "${GREEN}[OK] Docker seems configured for $REGION-docker.pkg.dev.${NC}"
        return 0
    else
        echo -e "${YELLOW}[ACTION NEEDED] Docker configuration for $REGION may be missing.${NC}"
        return 1
    fi
}

run_checks() {
    echo -e "${BLUE}Running diagnostic checks...${NC}"
    local failed=0
    
    check_gcloud || failed=1
    check_docker || failed=1
    check_auth || failed=1
    check_project || failed=1
    check_docker_auth || failed=1
    
    if [ $failed -eq 0 ]; then
        echo -e "\n${GREEN}AUTH_SETUP_NEEDED=false${NC}"
        return 0
    else
        echo -e "\n${RED}AUTH_SETUP_NEEDED=true${NC}"
        return 1
    fi
}

auth_setup() {
    print_header
    echo -e "${YELLOW}Starting authentication and setup...${NC}"
    
    echo -e "${BLUE}1. Running 'gcloud auth login'...${NC}"
    gcloud auth login
    
    echo -e "${BLUE}2. Setting project to $PROJECT_ID...${NC}"
    gcloud config set project "$PROJECT_ID"
    
    echo -e "${BLUE}3. Configuring Docker for $REGION...${NC}"
    gcloud auth configure-docker "$REGION-docker.pkg.dev"
    
    echo -e "\n${YELLOW}Setup complete. Re-running checks...${NC}"
    run_checks || echo -e "${YELLOW}Note: If Docker daemon was the only failure, please ensure Docker Desktop is open and running and rerun this command.${NC}"
}

require_ready() {
    if ! run_checks > /dev/null 2>&1; then
        echo -e "${RED}Prerequisites not met.${NC}"
        echo -e "${YELLOW}Please run: ./scripts/deploy.sh auth${NC}"
        exit 1
    fi
}

build_image() {
    echo -e "${BLUE}Building Docker image (linux/amd64)...${NC}"
    docker build --platform linux/amd64 -t "$IMAGE" .
}

push_image() {
    echo -e "${BLUE}Pushing Docker image to Artifact Registry...${NC}"
    docker push "$IMAGE"
}

deploy_api() {
    echo -e "${BLUE}Deploying API service: $API_SERVICE...${NC}"
    gcloud run deploy "$API_SERVICE" \
       --image "$IMAGE" \
       --memory 1Gi \
       --concurrency 10 \
       --min-instances 0 \
       --max-instances 4 \
       --cpu-boost \
       --region "$REGION" \
       --allow-unauthenticated \
       --project "$PROJECT_ID"
}

deploy_ui() {
    local service_name="${1:-$UI_SERVICE}"
    local variant="${2:-"trial_audit"}"
    local api_url
    api_url=$(get_api_url)

    echo -e "${BLUE}Deploying UI service: $service_name (Variant: $variant)...${NC}"
    echo -e "${BLUE}Targeting API URL: $api_url${NC}"
    
    gcloud run deploy "$service_name" \
       --image "$IMAGE" \
       --command "streamlit","run","frontend/app.py","--server.port","8080","--server.address","0.0.0.0" \
       --set-env-vars API_URL="$api_url",APP_VARIANT="$variant" \
       --memory 3Gi \
       --port 8080 \
       --concurrency 4 \
       --min-instances 0 \
       --max-instances 15 \
       --cpu-boost \
       --region "$REGION" \
       --allow-unauthenticated \
       --project "$PROJECT_ID"
}

# --- Main Logic ---

COMMAND=${1:-"check"}

case $COMMAND in
    check)
        print_header
        run_checks || exit 1
        ;;
    auth)
        auth_setup
        ;;
    build)
        print_header
        build_image
        ;;
    push)
        print_header
        push_image
        ;;
    api)
        print_header
        require_ready
        build_image
        push_image
        deploy_api
        ;;
    ui)
        print_header
        require_ready
        build_image
        push_image
        deploy_ui "$UI_SERVICE" "trial_audit"
        ;;
    trial-simulator)
        print_header
        require_ready
        build_image
        push_image
        deploy_ui "clintrial-simulator" "trial_simulator"
        ;;
    all)
        print_header
        require_ready
        build_image
        push_image
        deploy_api
        deploy_ui "$UI_SERVICE" "trial_audit"
        deploy_ui "clintrial-simulator" "trial_simulator"
        ;;
    *)
        echo "Usage: $0 {check|auth|build|push|api|ui|trial-simulator|all}"
        exit 1
        ;;
esac
