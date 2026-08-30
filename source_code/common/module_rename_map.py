"""Static import map for migrated modules.

This file documents the repository refactor from the original flat layout to the
new layered Source Code structure.
"""

MODULE_RENAME_MAP = {
    "auth.session_manager": "Source Code.ingestion.auth.session_manager",
    "auth.__init__": "Source Code.ingestion.auth.__init__",
    "providers.base": "Source Code.ingestion.providers.base",
    "providers.yfinance_provider": "Source Code.ingestion.providers.yfinance_provider",
    "providers.zerodha_provider": "Source Code.ingestion.providers.zerodha_provider",
    "providers.__init__": "Source Code.ingestion.providers.__init__",
    "data.fetcher": "Source Code.ingestion.data.fetcher",
    "data.storage": "Source Code.ingestion.data.storage",
    "data.__init__": "Source Code.ingestion.data.__init__",
    "analysis.indicators": "Source Code.processing.analysis.indicators",
    "analysis.patterns": "Source Code.processing.analysis.patterns",
    "analysis.screener": "Source Code.processing.analysis.screener",
    "analysis.__init__": "Source Code.processing.analysis.__init__",
    "backtesting.engine": "Source Code.analytics.backtesting.engine",
    "backtesting.metrics": "Source Code.analytics.backtesting.metrics",
    "backtesting.portfolio": "Source Code.analytics.backtesting.portfolio",
    "backtesting.__init__": "Source Code.analytics.backtesting.__init__",
    "algo.executor": "Source Code.analytics.strategy.algo.executor",
    "algo.strategy_base": "Source Code.analytics.strategy.algo.strategy_base",
    "algo.__init__": "Source Code.analytics.strategy.algo.__init__",
    "utils.helpers": "Source Code.common.utils.helpers",
    "utils.logger": "Source Code.common.utils.logger",
    "utils.validators": "Source Code.common.utils.validators",
    "utils.__init__": "Source Code.common.utils.__init__",
    "generate_session": "Source Code.orchestration.generate_session",
    "run_analysis": "Source Code.orchestration.run_analysis",
    "run_charts": "Source Code.orchestration.run_charts",
    "run_screener": "Source Code.orchestration.run_screener",
}

LEGACY_TO_NEW = MODULE_RENAME_MAP
NEW_TO_LEGACY = {v: k for k, v in MODULE_RENAME_MAP.items()}


def get_new_module_name(old_module_name: str) -> str:
    return MODULE_RENAME_MAP.get(old_module_name, old_module_name)


def get_legacy_module_name(new_module_name: str) -> str:
    return NEW_TO_LEGACY.get(new_module_name, new_module_name)
