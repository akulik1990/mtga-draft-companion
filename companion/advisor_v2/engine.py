"""Advisor v2.

Phase 0 stub: identical to upstream so the API, UI and replay plumbing carry two
advisors from the start. The real scorer replaces this in Phase 2 (see CLAUDE.md).
"""

from src.advisor.engine import DraftAdvisor


class DraftAdvisorV2(DraftAdvisor):
    pass
