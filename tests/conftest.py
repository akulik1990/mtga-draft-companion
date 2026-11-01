"""Shared fixture: a DraftSession over upstream's OTJ dataset fixture, in a temp data dir."""

import os
import tempfile

# Must precede any companion/src import: upstream fixes its paths from cwd at import time.
os.environ.setdefault("MTGA_COMPANION_DATA", tempfile.mkdtemp(prefix="companion-test-"))

import companion  # noqa: E402,F401

import shutil  # noqa: E402

import pytest  # noqa: E402

from companion import VENDOR
from companion.session import DraftSession
from src.configuration import Configuration
from src.limited_sets import SetDictionary, SetInfo
from src.log_scanner import ArenaScanner

OTJ = VENDOR / "tests" / "data" / "OTJ_PremierDraft_Data_2024_5_3.json"
# 90631 is absent from this 2024 snapshot; upstream omits unresolved IDs from the pack
# until its background resolver finds them, so 13 of the 14 cards are expected.
KNOWN_IN_PACK = 13

JOIN = '[UnityCrossThreadLogger]==> Event_Join {"id":"1","request":"{\\"EventName\\":\\"PremierDraft_OTJ_20240416\\"}"}\n'
P1P1 = (
    '[UnityCrossThreadLogger]Draft.Notify {"draftId":"1","SelfPick":1,"SelfPack":1,'
    '"PackCards":"90734,90584,90631,90362,90440,90349,90486,90527,90406,90439,90488,90480,90388,90459"}\n'
)


@pytest.fixture
def session(tmp_path, monkeypatch):
    sets, temp, logs = (tmp_path / d for d in ("Sets", "Temp", "Logs"))
    for d in (sets, temp, logs):
        d.mkdir()
    monkeypatch.setattr("src.constants.SETS_FOLDER", str(sets))
    monkeypatch.setattr("src.constants.TEMP_FOLDER", str(temp))
    monkeypatch.setattr("src.constants.DRAFT_LOG_FOLDER", str(logs))

    dataset = sets / "OTJ_PremierDraft_All_Data.json"
    shutil.copy(OTJ, dataset)
    listing = ([("OTJ", "PremierDraft", "All", "2024-04-16", "2024-05-03", 500, str(dataset), "2024-05-03 12:00:00")], [])
    monkeypatch.setattr("src.log_scanner.retrieve_local_set_list", lambda *a, **k: listing)
    monkeypatch.setattr("src.utils.retrieve_local_set_list", lambda *a, **k: listing)
    monkeypatch.setattr("companion.session.retrieve_local_set_list", lambda *a, **k: listing)
    monkeypatch.setattr("src.configuration.write_configuration", lambda *a, **k: True)
    monkeypatch.setattr("src.ui.orchestrator.write_configuration", lambda *a, **k: True)

    log = tmp_path / "Player.log"
    log.write_text("MTGA Log Start\n")
    config = Configuration()
    config.settings.arena_log_location = str(log)
    config.card_data.latest_dataset = dataset.name

    sets_dict = SetDictionary(data={"Outlaws": SetInfo(arena=["OTJ"], seventeenlands=["OTJ"], set_code="OTJ")})
    scanner = ArenaScanner(str(log), sets_dict, sets_location=str(sets), retrieve_unknown=True)
    scanner.file_size = 0
    s = DraftSession(scanner, config)
    s.log = log
    return s


def append(path, line):
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
