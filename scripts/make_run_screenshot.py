"""Screenshot the HTML run report of the latest successful pipeline run -> reports/figures/successful_run.png"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
REP = (Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reports").resolve()   # e.g. reports/movielens
(REP / "figures").mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1240, "height": 900}, device_scale_factor=2)
    pg.goto((REP / "run_report.html").as_uri())
    pg.screenshot(path=str(REP / "figures" / "successful_run.png"), full_page=True)
    b.close()
print(f"written {REP / 'figures' / 'successful_run.png'}")
