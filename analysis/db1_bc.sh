#!/usr/bin/env bash
R=/home/morris/Projects/mtga-draft-companion
export MTGA_COMPANION_DATA=$R/data
O=$R/data/Temp/sim/db1
uv run --directory $R python -u $R/analysis/db1_bc.py FRA $O/own_base.json --v2 deckbuilder.DEFAULT_BEST=False > $O/own_base.log 2>&1 &
uv run --directory $R python -u $R/analysis/db1_bc.py FRA $O/own_new.json --v2 deckbuilder.DEFAULT_BEST=True > $O/own_new.log 2>&1 &
wait
