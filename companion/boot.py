"""Build a DraftSession the way upstream's main.load_data does, minus the splash screen."""

import logging

from src.configuration import read_configuration, write_configuration
from src.dataset_updater import DatasetUpdater
from src.file_extractor import retrieve_arena_directory, search_arena_log_locations
from src.limited_sets import LimitedSets
from src.log_scanner import ArenaScanner

from companion.session import DraftSession

log = logging.getLogger(__name__)


def build_session(log_override: str | None = None, sync: bool = True) -> DraftSession:
    config, _ = read_configuration()

    log_path = search_arena_log_locations(
        log_override,
        config.settings.arena_log_location,
        config_pinned=config.settings.arena_log_pinned,
    )
    if log_path and not config.settings.arena_log_pinned and config.settings.arena_log_location != log_path:
        config.settings.arena_log_location = log_path
        write_configuration(config)
    log.info("Arena log: %s", log_path or "not found")

    if not config.settings.database_location and log_path:
        db = retrieve_arena_directory(log_path)
        if db:
            config.settings.database_location = db
            write_configuration(config)

    if sync and config.settings.auto_sync_datasets:
        DatasetUpdater(config).sync_datasets(lambda msg: log.info("sync: %s", msg))

    scanner = ArenaScanner(
        filename=log_path,
        set_list=LimitedSets().retrieve_limited_sets(),
        retrieve_unknown=True,
        db_path=config.settings.database_location,
    )
    session = DraftSession(scanner, config)
    session.load_engine_settings()

    # Deep scan for an in-progress or recovered draft before serving.
    scanner.draft_start_search()
    if scanner.retrieve_current_limited_event()[0]:
        session.orchestrator.sync_dataset_to_event()
        scanner.draft_data_search()
    return session
