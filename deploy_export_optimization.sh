#!/usr/bin/env bash

set -euo pipefail

# Deployment script for export optimization updates
# This script applies the changes needed to fix 504 Gateway Timeout errors

COMPOSE="docker compose"
PROJECT_NAME="ihdc"

# =============== Fancy output ===============
if command -v tput >/dev/null 2>&1; then
  BOLD="$(tput bold)"; RESET="$(tput sgr0)"
  GREEN="$(tput setaf 2)"; RED="$(tput setaf 1)"; BLUE="$(tput setaf 4)"; YELLOW="$(tput setaf 3)"
else
  BOLD=""; RESET=""; GREEN=""; RED=""; BLUE=""; YELLOW=""
fi

banner() {
  local text="$1"
  printf "\n${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
  printf "${BOLD}${YELLOW}%s${RESET}\n" "$text"
  printf "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n\n"
}

echo_step() {
  printf "${GREEN}✓${RESET} %s\n" "$1"
}

echo_info() {
  printf "${BLUE}ℹ${RESET} %s\n" "$1"
}

echo_warn() {
  printf "${YELLOW}⚠${RESET} %s\n" "$1"
}

echo_error() {
  printf "${RED}✖${RESET} %s\n" "$1"
}

# =============== Pre-flight checks ===============
banner "Export Optimization Deployment"

echo_info "This will deploy optimizations to fix 504 Gateway Timeout errors during export."
echo_info "Changes include:"
echo_info "  - Raw SQL queries for faster database access"
echo_info "  - Optimized batch sizes (500 records per request)"
echo_info "  - Increased nginx timeouts (60s -> 300s)"
echo_info "  - Increased uvicorn timeouts"
echo_info "  - Parallel batch fetching (5 simultaneous requests)"
echo ""

# Check if docker-compose is available
if ! command -v docker &> /dev/null && ! command -v docker-compose &> /dev/null; then
    echo_error "Docker or docker-compose not found. Please install Docker first."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    echo_error "docker-compose.yml not found. Please run this script from the project root."
    exit 1
fi

echo_info "Press Enter to continue or Ctrl+C to cancel..."
read -r

# =============== Backup current state ===============
banner "Step 1: Backup"

BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo_info "Creating backup in $BACKUP_DIR..."

# Backup modified files
cp data/views.py "$BACKUP_DIR/views.py.backup" 2>/dev/null || echo_warn "Could not backup data/views.py"
cp data/templates/data/datasetlist.html "$BACKUP_DIR/datasetlist.html.backup" 2>/dev/null || echo_warn "Could not backup datasetlist.html"
cp nginx/iheardatacollection.conf "$BACKUP_DIR/nginx.conf.backup" 2>/dev/null || echo_warn "Could not backup nginx config"
cp docker-compose.yml "$BACKUP_DIR/docker-compose.yml.backup" 2>/dev/null || echo_warn "Could not backup docker-compose.yml"

echo_step "Backup completed: $BACKUP_DIR"

# =============== Build new images ===============
banner "Step 2: Build Docker Images"

echo_info "Building web service with updated code..."
if $COMPOSE build web; then
    echo_step "Build completed successfully"
else
    echo_error "Build failed!"
    exit 1
fi

# =============== Stop services ===============
banner "Step 3: Stop Services"

echo_info "Stopping services gracefully..."
if $COMPOSE stop web nginx; then
    echo_step "Services stopped"
else
    echo_warn "Some services may not have stopped cleanly"
fi

# =============== Start services ===============
banner "Step 4: Start Services"

echo_info "Starting web service..."
if $COMPOSE up -d web; then
    echo_step "Web service started"
else
    echo_error "Failed to start web service!"
    exit 1
fi

sleep 2

echo_info "Reloading nginx configuration..."
if $COMPOSE exec nginx nginx -t; then
    echo_step "Nginx config test passed"
    $COMPOSE exec nginx nginx -s reload
    echo_step "Nginx reloaded"
else
    echo_error "Nginx config test failed!"
    echo_warn "Restarting nginx with new config..."
    $COMPOSE restart nginx
fi

echo_info "Restarting nginx service to apply timeout changes..."
if $COMPOSE restart nginx; then
    echo_step "Nginx restarted successfully"
else
    echo_error "Failed to restart nginx!"
    exit 1
fi

# =============== Verify services ===============
banner "Step 5: Verify Deployment"

echo_info "Checking service status..."
$COMPOSE ps

echo ""
echo_info "Checking web service logs (last 10 lines)..."
$COMPOSE logs --tail=10 web

# =============== Success ===============
banner "Deployment Complete 🎉"

echo_step "Export optimization has been deployed successfully!"
echo ""
echo_info "Next steps:"
echo_info "  1. Test the export functionality at: https://ihearandsee-at-rail.com/datasetlist/"
echo_info "  2. Monitor logs: docker compose logs -f web"
echo_info "  3. Check for performance logs showing query times"
echo ""
echo_info "Expected performance for 23,453 records:"
echo_info "  - Each batch request: 2-5 seconds (500 records)"
echo_info "  - Parallel requests: 5 batches at once"
echo_info "  - Total time: ~20-50 seconds"
echo ""
echo_info "If issues occur, rollback using:"
echo_info "  cp $BACKUP_DIR/*.backup <original-paths>"
echo_info "  ./deploy.sh"
echo ""

printf "${GREEN}All done!${RESET} ✨\n\n"

