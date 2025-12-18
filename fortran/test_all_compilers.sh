#!/bin/bash

# Test with multiple Fortran compilers
# This script:
# 1. Tests with gfortran (baseline)
# 2. Tests with Intel ifx (if available)
# 3. Tests with Intel ifort (if available)
#
# Options:
#   --clean    Clean all _build* directories before testing

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Move to the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "${SCRIPT_DIR}"

# Parse arguments
CLEAN_BUILD=false
for arg in "$@"; do
    case $arg in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $arg${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Multi-compiler Fortran test suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 0: Clean build directories if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo -e "${YELLOW}Step 0: Cleaning build directories...${NC}"
    for build_dir in "${SCRIPT_DIR}"/_build*; do
        if [ -d "$build_dir" ]; then
            rm -rf "$build_dir"
            echo -e "${GREEN}Removed $(basename "$build_dir")${NC}"
        fi
    done
    echo ""
fi

# Track test results
GFORTRAN_PASSED=false
IFX_PASSED=false
IFORT_PASSED=false

# Step 1: Test with gfortran (baseline)
echo -e "${YELLOW}Step 1: Testing with gfortran (baseline)...${NC}"
if ./test_with_fortran.sh --compiler=gfortran 2>&1 | tee /tmp/gfortran_test.log; then
    echo -e "${GREEN}✓ gfortran tests passed${NC}"
    GFORTRAN_PASSED=true
else
    echo -e "${RED}✗ gfortran tests failed${NC}"
    cat /tmp/gfortran_test.log | tail -20
fi
echo ""

# Step 2: Test with Intel ifx (if available)
if command -v ifx &> /dev/null; then
    echo -e "${YELLOW}Step 2: Testing with Intel ifx...${NC}"
    if ./test_with_fortran.sh --compiler=ifx 2>&1 | tee /tmp/ifx_test.log; then
        echo -e "${GREEN}✓ Intel ifx tests passed${NC}"
        IFX_PASSED=true
    else
        echo -e "${RED}✗ Intel ifx tests failed${NC}"
        cat /tmp/ifx_test.log | tail -20
    fi
    echo ""
else
    echo -e "${YELLOW}Step 2: Intel ifx not found, skipping...${NC}"
    echo ""
fi

# Step 3: Test with Intel ifort (if available)
if command -v ifort &> /dev/null; then
    echo -e "${YELLOW}Step 3: Testing with Intel ifort...${NC}"
    if ./test_with_fortran.sh --compiler=ifort 2>&1 | tee /tmp/ifort_test.log; then
        echo -e "${GREEN}✓ Intel ifort tests passed${NC}"
        IFORT_PASSED=true
    else
        echo -e "${RED}✗ Intel ifort tests failed${NC}"
        cat /tmp/ifort_test.log | tail -20
    fi
    echo ""
else
    echo -e "${YELLOW}Step 3: Intel ifort not found, skipping...${NC}"
    echo ""
fi

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
if [ "$GFORTRAN_PASSED" = true ]; then
    echo -e "${GREEN}✓ gfortran: PASSED${NC}"
else
    echo -e "${RED}✗ gfortran: FAILED${NC}"
fi

if command -v ifx &> /dev/null; then
    if [ "$IFX_PASSED" = true ]; then
        echo -e "${GREEN}✓ Intel ifx: PASSED${NC}"
    else
        echo -e "${RED}✗ Intel ifx: FAILED${NC}"
    fi
else
    echo -e "${YELLOW}○ Intel ifx: NOT AVAILABLE${NC}"
fi

if command -v ifort &> /dev/null; then
    if [ "$IFORT_PASSED" = true ]; then
        echo -e "${GREEN}✓ Intel ifort: PASSED${NC}"
    else
        echo -e "${RED}✗ Intel ifort: FAILED${NC}"
    fi
else
    echo -e "${YELLOW}○ Intel ifort: NOT AVAILABLE${NC}"
fi

echo ""

# Exit with error if any test failed
if [ "$GFORTRAN_PASSED" = false ]; then
    exit 1
fi

if command -v ifx &> /dev/null && [ "$IFX_PASSED" = false ]; then
    exit 1
fi

if command -v ifort &> /dev/null && [ "$IFORT_PASSED" = false ]; then
    exit 1
fi

echo -e "${GREEN}=== All compiler tests completed successfully ===${NC}"

