#!/bin/bash
# Profile benchmark2 (SVE computation only)
# Assumes benchmark2 is already built

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/_build"
BENCHMARK_BINARY="${BUILD_DIR}/benchmark2"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== Profiling benchmark2 ===${NC}"

# Check if benchmark2 exists
if [[ ! -f "${BENCHMARK_BINARY}" ]]; then
    echo -e "${RED}Error: benchmark2 not found at ${BENCHMARK_BINARY}${NC}"
    echo -e "${YELLOW}Please build it first:${NC}"
    echo -e "  cd ${SCRIPT_DIR}"
    echo -e "  bash run_with_rust_capi.sh"
    exit 1
fi

# Check if binary is executable
if [[ ! -x "${BENCHMARK_BINARY}" ]]; then
    echo -e "${RED}Error: ${BENCHMARK_BINARY} is not executable${NC}"
    exit 1
fi

echo -e "${GREEN}Found benchmark2 at: ${BENCHMARK_BINARY}${NC}"
echo ""

# Detect OS and choose profiling tool
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS: Use sample or Instruments
    if command -v sample &> /dev/null; then
        echo -e "${YELLOW}Using 'sample' profiler (macOS)${NC}"
        echo -e "${YELLOW}Profiling for 60 seconds...${NC}"
        echo ""
        # sample command: sample <pid|process-name> [duration] [samplingInterval] [-file <filename>]
        "${BENCHMARK_BINARY}" &
        BENCHMARK_PID=$!
        sleep 1  # Give process time to start
        sample ${BENCHMARK_PID} 60 -file "${BUILD_DIR}/profile.txt"
        wait ${BENCHMARK_PID} 2>/dev/null || true
        echo ""
        echo -e "${GREEN}Profile saved to: ${BUILD_DIR}/profile.txt${NC}"
        echo -e "${YELLOW}To view:${NC}"
        echo -e "  cat ${BUILD_DIR}/profile.txt"
    elif command -v instruments &> /dev/null; then
        echo -e "${YELLOW}Using 'instruments' profiler (macOS)${NC}"
        echo -e "${YELLOW}Starting Time Profiler...${NC}"
        echo ""
        instruments -t "Time Profiler" "${BENCHMARK_BINARY}" -D "${BUILD_DIR}/profile.trace"
        echo ""
        echo -e "${GREEN}Trace saved to: ${BUILD_DIR}/profile.trace${NC}"
        echo -e "${YELLOW}To view:${NC}"
        echo -e "  open ${BUILD_DIR}/profile.trace"
    else
        echo -e "${RED}Error: No profiling tool found (sample or instruments)${NC}"
        echo -e "${YELLOW}Falling back to basic execution...${NC}"
        "${BENCHMARK_BINARY}"
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux: Use perf
    if command -v perf &> /dev/null; then
        echo -e "${YELLOW}Using 'perf' profiler (Linux)${NC}"
        echo ""
        perf record -g "${BENCHMARK_BINARY}"
        perf report -g > "${BUILD_DIR}/perf_report.txt"
        echo ""
        echo -e "${GREEN}Profile saved to: ${BUILD_DIR}/perf_report.txt${NC}"
        echo -e "${YELLOW}To view:${NC}"
        echo -e "  cat ${BUILD_DIR}/perf_report.txt"
        echo -e "${YELLOW}Or use interactive viewer:${NC}"
        echo -e "  perf report"
    elif command -v valgrind &> /dev/null; then
        echo -e "${YELLOW}Using 'valgrind' profiler (Linux)${NC}"
        echo ""
        valgrind --tool=callgrind --callgrind-out-file="${BUILD_DIR}/callgrind.out" "${BENCHMARK_BINARY}"
        echo ""
        echo -e "${GREEN}Profile saved to: ${BUILD_DIR}/callgrind.out${NC}"
        echo -e "${YELLOW}To view:${NC}"
        echo -e "  kcachegrind ${BUILD_DIR}/callgrind.out"
    else
        echo -e "${RED}Error: No profiling tool found (perf or valgrind)${NC}"
        echo -e "${YELLOW}Falling back to basic execution...${NC}"
        "${BENCHMARK_BINARY}"
    fi
else
    echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
    echo -e "${YELLOW}Falling back to basic execution...${NC}"
    "${BENCHMARK_BINARY}"
fi

echo ""
echo -e "${GREEN}Profiling completed!${NC}"

