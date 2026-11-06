#!/usr/bin/env bash
# Step B secondary: tools/builder_check.py FRA, in-memory overrides only (no layer file is edited). Arms: as configured (gihwr, deckmodel_top3),
# OBJECTIVE=model with the stored deckmodel_top3 and with the new deckmodel_v2 / deckmodel_top3_v2. Logs in data/Temp/dm1106/bc_*.log
R=/home/morris/Projects/mtga-draft-companion
export MTGA_COMPANION_DATA=$R/data
run() { name=$1; shift; uv run --directory $R python -u $R/tools/builder_check.py FRA --n 300 --jobs 10 "$@" > $R/data/Temp/dm1106/bc_$name.log 2>&1; }
run gihwr
run model_top3 --v2 deckbuilder.OBJECTIVE=model
run model_v2 --v2 deckbuilder.OBJECTIVE=model --v2 deckbuilder.MODEL=deckmodel_v2
run model_top3_v2 --v2 deckbuilder.OBJECTIVE=model --v2 deckbuilder.MODEL=deckmodel_top3_v2
