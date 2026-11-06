#!/usr/bin/env bash
R=/home/morris/Projects/mtga-draft-companion
export MTGA_COMPANION_DATA=$R/data
O=$R/data/Temp/sim/db1
uv run --directory $R python -u $R/tools/mock_draft.py compare FRA --seeds 100 --start 1 --out $O/mock_base.json --v2 deckbuilder.DEFAULT_BEST=False > $O/mock_base.log 2>&1 &
uv run --directory $R python -u $R/tools/mock_draft.py compare FRA --seeds 100 --start 1 --out $O/mock_new.json > $O/mock_new.log 2>&1 &
uv run --directory $R python -u $R/tools/builder_check.py FRA --n 300 --jobs 5 --v2 deckbuilder.DEFAULT_BEST=False > $O/bc_base.log 2>&1 &
uv run --directory $R python -u $R/tools/builder_check.py FRA --n 300 --jobs 5 --v2 deckbuilder.DEFAULT_BEST=True > $O/bc_new.log 2>&1 &
wait
