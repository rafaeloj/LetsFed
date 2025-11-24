#!/bin/bash
#
# Cleanup script for LetsFed Docker containers
#
# This script stops and removes all Docker containers, networks, and volumes
# created by docker-compose for the LetsFed project.
#
# Usage:
#   ./cleanup.sh [OPTIONS]
#
# Options:
#   --volumes, -v    Also remove named volumes declared in the docker-compose.yml
#   --help, -h       Show this help message
#

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default options
REMOVE_VOLUMES=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--volumes)
            REMOVE_VOLUMES=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Cleanup Docker containers for LetsFed project"
            echo ""
            echo "Options:"
            echo "  -v, --volumes    Also remove named volumes"
            echo "  -h, --help       Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}╔════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║    LetsFed Docker Cleanup Utility         ║${NC}"
echo -e "${YELLOW}╚════════════════════════════════════════════╝${NC}"
echo ""

# Check if docker-compose.yml exists
if [[ ! -f "docker-compose.yml" ]]; then
    echo -e "${RED}Error: docker-compose.yml not found in current directory${NC}"
    exit 1
fi

# Stop and remove containers, networks, and orphans
echo -e "${GREEN}→${NC} Stopping and removing containers..."
if [[ "$REMOVE_VOLUMES" == true ]]; then
    echo -e "${YELLOW}→${NC} Also removing volumes..."
    docker compose down --remove-orphans --volumes
else
    docker compose down --remove-orphans
fi

echo ""
echo -e "${GREEN}✓${NC} Cleanup completed successfully!"
echo ""

# Show remaining containers (if any)
REMAINING=$(docker ps -a --filter "name=letsfed" --filter "name=rfl" --format "{{.Names}}" 2>/dev/null || true)
if [[ -n "$REMAINING" ]]; then
    echo -e "${YELLOW}Warning: Some LetsFed containers still exist:${NC}"
    echo "$REMAINING"
    echo ""
    echo -e "${YELLOW}You may need to remove them manually with:${NC}"
    echo "  docker rm -f <container_name>"
else
    echo -e "${GREEN}✓${NC} No remaining LetsFed containers found"
fi

echo ""
