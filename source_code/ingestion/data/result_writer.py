"""Persistence helpers for ranked trading signals."""

import json
import logging
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from source_code.common.path_resolver import ensure_dir, resolve_path

logger = logging.getLogger(__name__)

_SUPPORTED_FORMATS = {"csv", "json", "parquet", "html"}


class ResultWriter:
    """Write ranking results to the configured output directory."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        configured_dir = self.config.get("paths", {}).get("gold_data_dir", "data/gold")
        self.output_dir = resolve_path(configured_dir)

    def save_results(
        self,
        df: pd.DataFrame,
        output_path: Optional[Union[str, Path]] = None,
        format: str = "csv",
    ) -> str:
        """Save a DataFrame and return the absolute output path."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        output_format = format.lower().lstrip(".")
        if output_format not in _SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported format '{format}'. Choose from {sorted(_SUPPORTED_FORMATS)}"
            )

        path = self._resolve_output_path(output_path, output_format)
        ensure_dir(path.parent)

        if output_format == "csv":
            df.to_csv(path, index=False)
        elif output_format == "json":
            with path.open("w", encoding="utf-8") as file_handle:
                json.dump(df.to_dict(orient="records"), file_handle, indent=2, default=str)
        elif output_format == "html":
            path.write_text(self._build_dashboard(df), encoding="utf-8")
        else:
            try:
                df.to_parquet(path, index=False)
            except ImportError as exc:
                raise RuntimeError(
                    "Parquet output requires pyarrow or fastparquet to be installed"
                ) from exc

        logger.info("Saved %d rows to %s", len(df), path)
        return str(path)

    def _build_dashboard(self, df: pd.DataFrame) -> str:
        """Build a self-contained dashboard for opening directly in a browser."""
        bullish_count = int((df["State"] == "BULLISH").sum()) if "State" in df else 0
        bearish_count = int((df["State"] == "BEARISH").sum()) if "State" in df else 0
        table = df.to_html(index=False, classes="signals-table", border=0, na_rep="-")
        table = table.replace('<td>BULLISH</td>', '<td><span class="pill bullish">BULLISH</span></td>')
        table = table.replace('<td>BEARISH</td>', '<td><span class="pill bearish">BEARISH</span></td>')
        return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nifty Signal Desk</title>
<style>
:root {{ --ink:#17212b; --muted:#66737d; --paper:#f4f1ea; --panel:#fffdf8; --line:#ddd8cd; --green:#087f5b; --red:#b42318; --accent:#d97706; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:radial-gradient(circle at 90% 0%, #f9dfb3 0, transparent 28%), var(--paper); font:15px/1.5 Georgia, serif; }}
main {{ max-width:1180px; margin:0 auto; padding:42px 22px 64px; }}
.eyebrow {{ color:var(--accent); font:700 12px/1.2 ui-sans-serif, sans-serif; letter-spacing:2px; text-transform:uppercase; }}
h1 {{ margin:8px 0 4px; font-size:clamp(32px, 5vw, 58px); line-height:1; font-weight:500; }}
.subtitle {{ color:var(--muted); margin:0 0 28px; font-family:ui-sans-serif, sans-serif; }}
.stats {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; margin-bottom:25px; }}
.stat {{ background:var(--panel); border:1px solid var(--line); padding:18px; }} .stat strong {{ display:block; font:700 30px ui-sans-serif, sans-serif; }} .stat span {{ color:var(--muted); font:12px ui-sans-serif, sans-serif; text-transform:uppercase; letter-spacing:1px; }}
.toolbar {{ display:flex; justify-content:space-between; gap:14px; align-items:center; margin-bottom:12px; }}
input {{ width:270px; max-width:100%; padding:11px 13px; border:1px solid var(--line); background:var(--panel); font:14px ui-sans-serif, sans-serif; }}
.table-wrap {{ overflow-x:auto; background:var(--panel); border:1px solid var(--line); }} table {{ width:100%; border-collapse:collapse; font-family:ui-sans-serif, sans-serif; font-size:13px; }} th {{ color:var(--muted); text-align:left; font-size:11px; text-transform:uppercase; letter-spacing:.7px; cursor:pointer; }} th, td {{ padding:13px 14px; border-bottom:1px solid var(--line); white-space:nowrap; }} tbody tr:hover {{ background:#fff5df; }} .pill {{ display:inline-block; padding:3px 8px; font-size:11px; font-weight:700; }} .bullish {{ color:var(--green); background:#d9f4e8; }} .bearish {{ color:var(--red); background:#fde1de; }}
@media (max-width:600px) {{ main {{ padding:28px 14px 45px; }} .stats {{ grid-template-columns:1fr; }} .toolbar {{ align-items:stretch; flex-direction:column; }} input {{ width:100%; }} }}
</style></head><body><main>
<div class="eyebrow">Trading system / signal review</div><h1>Nifty Signal Desk</h1>
<p class="subtitle">Latest crossover rankings generated from the pipeline.</p>
<section class="stats"><div class="stat"><strong>{len(df)}</strong><span>Ranked symbols</span></div>
<div class="stat"><strong>{bullish_count}</strong><span>Bullish signals</span></div><div class="stat"><strong>{bearish_count}</strong><span>Bearish signals</span></div></section>
<div class="toolbar"><strong>Watchlist</strong><input id="search" type="search" placeholder="Filter by symbol or state" aria-label="Filter signals"></div>
<div class="table-wrap">{table}</div></main>
<script>const search=document.getElementById('search'); search.addEventListener('input',()=>{{const q=search.value.toLowerCase(); document.querySelectorAll('tbody tr').forEach(r=>{{r.hidden=!r.innerText.toLowerCase().includes(q)}})}}); document.querySelectorAll('th').forEach((h,i)=>h.addEventListener('click',()=>{{const b=document.querySelector('tbody'); [...b.rows].sort((a,c)=>a.cells[i].innerText.localeCompare(c.cells[i].innerText,undefined,{{numeric:true}})).forEach(r=>b.appendChild(r))}}));</script>
</body></html>'''

    def export_watchlist(
        self,
        ranked_df: pd.DataFrame,
        filename: str = "nifty100_signals",
        format: str = "csv",
    ) -> str:
        """Save a ranked watchlist using a filename without a required extension."""
        return self.save_results(ranked_df, self.output_dir / filename, format=format)

    def _resolve_output_path(
        self,
        output_path: Optional[Union[str, Path]],
        output_format: str,
    ) -> Path:
        if output_path is None:
            return self.output_dir / f"nifty100_signals.{output_format}"

        path = resolve_path(output_path)
        if path.suffix.lower() != f".{output_format}":
            path = path.with_suffix(f".{output_format}")
        return path


def save_results(
    df: pd.DataFrame,
    format: str = "csv",
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[dict] = None,
) -> str:
    """Functional wrapper around :class:`ResultWriter`."""
    return ResultWriter(config).save_results(df, output_path=output_path, format=format)


def export_watchlist(
    ranked_df: pd.DataFrame,
    filename: str = "nifty100_signals",
    format: str = "csv",
    config: Optional[dict] = None,
) -> str:
    """Functional wrapper for saving a ranked watchlist."""
    return ResultWriter(config).export_watchlist(ranked_df, filename, format=format)
