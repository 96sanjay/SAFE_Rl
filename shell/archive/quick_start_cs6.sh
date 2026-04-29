#!/bin/bash
# Quick start script for cs6 schema migration

echo "=============================================="
echo "  CS6 SCHEMA MIGRATION - QUICK START"
echo "=============================================="

# Configuration - UPDATE THESE PATHS
REPO_ROOT="/home/extra-storage/THESIS/Safe-CityLearn-Fork/Safe-CityLearn-Fork"
WORKING_SCHEMA="/home/extra-storage/THESIS/CityLearn/data/datasets/citylearn_challenge_2022_phase_all_plus_evs/schema.json"
CS6_DIR="${REPO_ROOT}/data/cs6/cs6"

echo ""
echo "Step 1: Navigate to repository"
cd "$REPO_ROOT" || exit 1
pwd

echo ""
echo "Step 2: Find your cs6 schema"
if [ -f "${CS6_DIR}/schema.json" ]; then
    CS6_SCHEMA="${CS6_DIR}/schema.json"
    echo "✅ Found: ${CS6_SCHEMA}"
elif [ -f "${CS6_DIR}/schema_grouped.json" ]; then
    CS6_SCHEMA="${CS6_DIR}/schema_grouped.json"
    echo "✅ Found: ${CS6_SCHEMA}"
else
    echo "❌ No schema found in ${CS6_DIR}"
    echo "   Please create cs6/cs6/schema.json first"
    exit 1
fi

echo ""
echo "Step 3: Compare schemas (see what's different)"
python compare_schemas.py \
    --working "$WORKING_SCHEMA" \
    --target "$CS6_SCHEMA"

echo ""
echo "Step 4: Convert schema (if needed)"
python convert_cs6_schema.py \
    --input "$CS6_SCHEMA" \
    --output "${CS6_DIR}/schema_converted.json" \
    --reference "$WORKING_SCHEMA"

echo ""
echo "Step 5: Test schema compatibility"
python test_cs6_schema.py --schema "${CS6_DIR}/schema_converted.json"

echo ""
echo "=============================================="
echo "  MIGRATION COMPLETE!"
echo "=============================================="
echo ""
echo "Next: Set environment variable and test"
echo "  export CITYLEARN_SCHEMA='${CS6_DIR}/schema_converted.json'"
echo "  python -c 'from scripts.make_env import make_base_env; env = make_base_env(); print(env)'"
