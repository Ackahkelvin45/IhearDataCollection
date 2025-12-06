#!/usr/bin/env bash

set -euo pipefail

# Simple deployment script for ihdc stack
# - Builds images
# - Runs DB migrations + collectstatic (via migration service)
# - Starts/updates web, nginx and celery-worker services

COMPOSE="docker compose"
PROJECT_NAME="ihdc"
NO_CACHE="false"

# =============== Fancy output / animations ===============
if command -v tput >/dev/null 2>&1; then
  BOLD="$(tput bold)"; DIM="$(tput dim)"; RESET="$(tput sgr0)"
  GREEN="$(tput setaf 2)"; RED="$(tput setaf 1)"; BLUE="$(tput setaf 4)"; CYAN="$(tput setaf 6)"; YELLOW="$(tput setaf 3)"
else
  BOLD=""; DIM=""; RESET=""; GREEN=""; RED=""; BLUE=""; CYAN=""; YELLOW=""
fi

SPINNER_PID=""
start_spinner() {
  local msg="$1"
  local frames=("⠋" "⠙" "⠸" "⠴" "⠦" "⠇")
  printf "${DIM}%s${RESET}\n" "$msg"
  (
    i=0
    while :; do
      printf "\r${CYAN}%s${RESET} ${DIM}%s${RESET}" "${frames[$i]}" "$msg"
      i=$(( (i+1) % ${#frames[@]} ))
      sleep 0.12
    done
  ) &
  SPINNER_PID=$!
  disown "$SPINNER_PID" 2>/dev/null || true
}

stop_spinner() {
  local status=$1
  local msg="$2"
  if [[ -n "${SPINNER_PID}" ]] && kill -0 "${SPINNER_PID}" 2>/dev/null; then
    kill "${SPINNER_PID}" 2>/dev/null || true
    wait "${SPINNER_PID}" 2>/dev/null || true
  fi
  if [[ "$status" -eq 0 ]]; then
    printf "\r${GREEN}✔${RESET} %s\n" "$msg"
  else
    printf "\r${RED}✖${RESET} %s\n" "$msg"
  fi
}

banner() {
  local text="$1"
  printf "\n${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
  printf "${BOLD}${YELLOW}%s${RESET}\n" "$text"
  printf "${BOLD}${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n\n"
}

usage() {
  echo "Usage: $0 [--no-cache] [--pull]" >&2
  exit 1
}

PULL="false"
while [[ ${1:-} =~ ^- ]]; do
  case "$1" in
    --no-cache) NO_CACHE="true" ; shift ;;
    --pull) PULL="true" ; shift ;;
    *) usage ;;
  esac
done

echo "[deploy] Using project: $PROJECT_NAME"

if [[ "$PULL" == "true" ]]; then
  banner "Pulling base images"
  start_spinner "Pulling latest images (this may take a while)..."
  if $COMPOSE pull; then
    stop_spinner 0 "Images pulled"
  else
    stop_spinner 1 "Image pull failed (continuing if local cache exists)"
  fi
fi

banner "Building images (no-cache=$NO_CACHE)"
start_spinner "Building Docker images..."
if [[ "$NO_CACHE" == "true" ]]; then
  if $COMPOSE build --no-cache; then stop_spinner 0 "Build completed"; else stop_spinner 1 "Build failed"; exit 1; fi
else
  if $COMPOSE build; then stop_spinner 0 "Build completed"; else stop_spinner 1 "Build failed"; exit 1; fi
fi

banner "Starting core dependencies"
start_spinner "Starting db and redis..."
if $COMPOSE up -d db redis; then stop_spinner 0 "Dependencies started"; else stop_spinner 1 "Failed to start dependencies"; exit 1; fi

banner "Applying migrations and collecting static"
# The migration service is defined in docker-compose.yml and runs:
# python manage.py migrate && collectstatic && seed_db
start_spinner "Running Django migrations + collectstatic..."
if $COMPOSE run --rm migration; then stop_spinner 0 "Migrations complete"; else stop_spinner 1 "Migrations failed"; exit 1; fi

banner "Starting application services"
start_spinner "Starting web, celery-worker and nginx..."
if $COMPOSE up -d web celery-worker nginx; then stop_spinner 0 "Services are up"; else stop_spinner 1 "Failed to start services"; exit 1; fi

banner "Service status"
$COMPOSE ps || true

banner "Deployment complete 🎉"
printf "${GREEN}Your app should be available at:${RESET} ${BOLD}http://localhost:8000${RESET}\n\n"

# Confetti animation
confetti=("✨" "🎉" "🚀" "🎯" "✅")
for i in {1..24}; do
  printf "${CYAN}%s${RESET} " "${confetti[$((RANDOM%${#confetti[@]}))]}"
  sleep 0.02
done
printf "\n\n"
