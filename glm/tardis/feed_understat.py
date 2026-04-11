"""
Understat xG feed — the structural signal under the noise of football scores.

Why xG, and why Understat
-------------------------
A Premier League final score is a stochastic finger on the keyboard: the
ball rebounds off a shin and loops in, or a goal-line clearance denies a
perfect through-ball.  Goals are the noisiest signal football gives us —
they are what landed on the page, not what the pianist's fingers felt on
the keys.  Professional football modellers long ago stopped fitting
Poisson models to raw goals and moved to expected-goals (xG), which sums
the shot-quality of every attempt a team actually created.  xG strips the
luck out and captures team strength stably over the small samples a 38-
game season gives us.  Two teams can draw 1-1 on the scoreboard while xG
says 2.4 to 0.6: that is information our goals-based Poisson layer is
currently throwing away.

The pianist analogy
-------------------
``docs/on_her_nature.md`` describes the Angel as a pianist who must learn
to feel the keys before she can play — where the shape of the hand on the
instrument is the structural feature underneath the noise of any single
performance.  xG is exactly that for football.  The final score is the
concert recording with all its mistakes and fluke note choices; xG is
what the fingers-feel-the-keys looks like for ninety minutes of football.
If the football side of MKAngel is going to forecast anything, it has to
listen to the fingers, not the concert.

Wishlist entry fulfilled
------------------------
``glm/tardis/wishlist.py`` has long carried an entry reading ``xG data
feed from Understat or StatsBomb Open Data``.  This module fulfils that
wishlist entry against Understat, whose free public league page renders
the full team table (position, team, M, W, D, L, G, GA, Pts, xG, xG_diff,
xGA, xGA_diff, xPTS, xPTS_diff) server-side into the HTML.

Terms
-----
Understat is a free public site built by and for football fans and
analysts.  It is not a commercial data provider.  This scraper only
accesses public data — the same data a person loading the page in a
browser would see — and it makes at most one request per call so it
does not hammer their servers.  If Understat ever object we stop.

Implementation
--------------
Understat's own CSS/webfonts are aggressive enough that a plain HTTP
fetch sometimes returns garbled characters; headless Chromium via
Playwright renders the page the way a human browser would and gives us
clean UTF-8 table cells.  Python bindings for Playwright are not
installed in this environment, so this module shells out to Node.js
instead: the JavaScript scraper is stored as a triple-quoted string,
written to ``/tmp/pw/feed_understat_<pid>.js`` at runtime (so it can
resolve ``require('playwright')`` against ``/tmp/pw/node_modules``), and
invoked via ``subprocess``.  Its stdout is a JSON array which we parse
into ``TeamXG`` dataclasses.

The HTTPS_PROXY env var is parsed into a structured ``{server, username,
password}`` object and passed to ``chromium.launch({proxy: ...})``:
Chromium's proxy auth handler does not pick those credentials up from
the environment directly, only from the launch config.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLAYWRIGHT_ROOT = "/tmp/pw"
PLAYWRIGHT_PACKAGE = "/tmp/pw/node_modules/playwright"
DEFAULT_BROWSER_PATH = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
DEFAULT_LEAGUE_URL = "https://understat.com/league"


def _utc_now_iso() -> str:
    """Return a UTC ISO-8601 timestamp with second precision."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# TeamXG dataclass
# ---------------------------------------------------------------------------

@dataclass
class TeamXG:
    """A single team's row from the Understat league table.

    Everything Understat publishes for a team in one snapshot: the classic
    results columns (matches, wins, draws, losses, goals for/against,
    points) and the expected-goals columns that this module exists for
    (xG for, xG diff, xG against, xGA diff, xPTS, xPTS diff).

    ``xg_diff`` is the luck-for-attack signal: actual goals minus xG.
    Strongly positive means a team is finishing hot; negative means they
    are being wasteful and are due for regression.

    ``xga_diff`` is the defensive mirror: goals conceded minus xGA.  A
    team whose keeper is having a purple patch shows up as negative.

    ``xpts_diff`` is actual points minus expected points and is the
    cleanest single-number over/under-performance indicator we have.
    """

    rank: int
    team: str
    matches_played: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    points: int
    xg_for: float
    xg_diff: float
    xg_against: float
    xga_diff: float
    xpts: float
    xpts_diff: float
    snapshot_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TeamXG":
        return cls(**data)


# ---------------------------------------------------------------------------
# Node.js scraper — written to a temp file at runtime
# ---------------------------------------------------------------------------

_NODE_SCRIPT = r"""
// Understat league-table scraper. Called with argv[2] = league code.
// Prints a JSON array of team rows to stdout, exits 0 on success.

const { chromium } = require('playwright');

function parseProxy(rawUrl) {
    if (!rawUrl) return null;
    try {
        const u = new URL(rawUrl);
        const proxy = { server: `${u.protocol}//${u.host}` };
        if (u.username) proxy.username = decodeURIComponent(u.username);
        if (u.password) proxy.password = decodeURIComponent(u.password);
        return proxy;
    } catch (e) {
        return null;
    }
}

function toNumber(s) {
    if (s === null || s === undefined) return null;
    // Understat uses characters like "+1.23" and "\u22121.23" for negatives.
    const cleaned = String(s)
        .replace(/\u2212/g, '-')
        .replace(/\s+/g, '')
        .replace(/^\+/, '');
    if (cleaned === '' || cleaned === '-') return null;
    const n = Number(cleaned);
    return Number.isFinite(n) ? n : null;
}

(async () => {
    const league = process.argv[2] || 'EPL';
    const timeoutMs = Number(process.argv[3] || '60000');
    const browserPath = process.argv[4] || undefined;

    const proxy = parseProxy(process.env.HTTPS_PROXY || process.env.https_proxy || '');

    const launchOpts = {
        headless: true,
        args: ['--no-sandbox', '--disable-dev-shm-usage'],
        ignoreHTTPSErrors: true,
    };
    if (browserPath) launchOpts.executablePath = browserPath;
    if (proxy) launchOpts.proxy = proxy;

    let browser;
    try {
        browser = await chromium.launch(launchOpts);
    } catch (e) {
        process.stderr.write(`launch failed: ${e.message}\n`);
        process.exit(2);
    }

    try {
        const context = await browser.newContext({
            ignoreHTTPSErrors: true,
            userAgent:
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 ' +
                '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        });
        const page = await context.newPage();
        const url = `https://understat.com/league/${league}`;
        await page.goto(url, { waitUntil: 'networkidle', timeout: timeoutMs });
        // Understat sometimes hydrates after networkidle; give it a beat.
        await page.waitForTimeout(4000);

        // Understat's league page has several tables (mostly datepicker
        // widgets); the one we want is the only one whose tbody holds the
        // league-sized set of team rows.  We pick the largest tbody whose
        // header starts with the rank column.
        const rows = await page.evaluate(() => {
            const tables = Array.from(document.querySelectorAll('table'));
            let best = null;
            let bestCount = 0;
            for (const t of tables) {
                const head = Array.from(t.querySelectorAll('thead th'))
                    .map((th) => (th.innerText || '').trim());
                if (head.length < 10) continue;
                // Expect first header to be rank ("№" or "#") and second "Team".
                if (!/team/i.test(head[1] || '')) continue;
                const trs = t.querySelectorAll('tbody tr');
                if (trs.length > bestCount) {
                    bestCount = trs.length;
                    best = t;
                }
            }
            if (!best) return [];
            return Array.from(best.querySelectorAll('tbody tr')).map((tr) =>
                Array.from(tr.querySelectorAll('td')).map(
                    (td) => (td.innerText || td.textContent || '').trim()
                )
            );
        });

        // Understat glues the diff column into the same <td> as its parent
        // metric, so a cell looks like "63.90+1.90" or "26.43-2.43".  The
        // 12-column layout is: rank, team, M, W, D, L, G, GA, PTS, xG,
        // xGA, xPTS — and the three xG cells each carry their own diff.
        const teams = [];
        for (const cells of rows) {
            if (!cells || cells.length < 12) continue;
            teams.push({
                rank: cells[0],
                team: cells[1],
                matches_played: cells[2],
                wins: cells[3],
                draws: cells[4],
                losses: cells[5],
                goals_for: cells[6],
                goals_against: cells[7],
                points: cells[8],
                xg_for: cells[9],
                xg_against: cells[10],
                xpts: cells[11],
            });
        }

        // Split an Understat concatenated cell ("63.90+1.90", "26.43-2.43")
        // into the parent metric and its diff.  Regex: leading unsigned
        // float, followed by an explicitly signed float.
        function splitPair(s) {
            if (s === null || s === undefined) return [null, null];
            const cleaned = String(s).replace(/\u2212/g, '-').trim();
            const m = cleaned.match(
                /^([0-9]+(?:\.[0-9]+)?)([+-][0-9]+(?:\.[0-9]+)?)?$/
            );
            if (!m) return [Number(cleaned) || null, null];
            return [Number(m[1]), m[2] === undefined ? null : Number(m[2])];
        }

        const final = teams.map((t) => {
            const [xg, xgDiff] = splitPair(t.xg_for);
            const [xga, xgaDiff] = splitPair(t.xg_against);
            const [xpts, xptsDiff] = splitPair(t.xpts);
            return {
                rank: Number(t.rank),
                team: t.team,
                matches_played: Number(t.matches_played),
                wins: Number(t.wins),
                draws: Number(t.draws),
                losses: Number(t.losses),
                goals_for: Number(t.goals_for),
                goals_against: Number(t.goals_against),
                points: Number(t.points),
                xg_for: xg,
                xg_diff: xgDiff !== null ? xgDiff : 0,
                xg_against: xga,
                xga_diff: xgaDiff !== null ? xgaDiff : 0,
                xpts: xpts,
                xpts_diff: xptsDiff !== null ? xptsDiff : 0,
            };
        });

        process.stdout.write(JSON.stringify(final));
        await browser.close();
        process.exit(0);
    } catch (e) {
        process.stderr.write(`scrape failed: ${e.message}\n`);
        try { await browser.close(); } catch (_) {}
        process.exit(3);
    }
})();
"""


# ---------------------------------------------------------------------------
# UnderstatFeed
# ---------------------------------------------------------------------------

class UnderstatFeed:
    """Headless-Chromium feed for Understat's league xG table.

    One instance per league code.  ``fetch_league_table()`` is the
    workhorse; everything else is convenience on top of it.  Every public
    method swallows the expected subprocess / parse / filesystem errors
    and returns an empty list (or ``None``) rather than raising — the
    football side of MKAngel can always fall back to its goals-based
    Poisson signal if xG is temporarily unavailable.
    """

    def __init__(
        self,
        league: str = "EPL",
        timeout: int = 60,
        browser_path: str = DEFAULT_BROWSER_PATH,
    ):
        self.league = league
        self.timeout = timeout
        self.browser_path = browser_path

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def fetch_league_table(self) -> list[TeamXG]:
        """Scrape the Understat league page and return every team row."""
        if not os.path.exists(PLAYWRIGHT_PACKAGE):
            logger.warning(
                "playwright not installed at %s — returning empty list",
                PLAYWRIGHT_PACKAGE,
            )
            return []

        script_path = Path(PLAYWRIGHT_ROOT) / f"feed_understat_{os.getpid()}.js"
        try:
            script_path.write_text(_NODE_SCRIPT, encoding="utf-8")
        except OSError as exc:
            logger.warning("could not write node script to %s: %s", script_path, exc)
            return []

        cmd = [
            "node",
            str(script_path),
            self.league,
            str(self.timeout * 1000),
            self.browser_path,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout + 15,
                cwd=PLAYWRIGHT_ROOT,
                env=os.environ.copy(),
            )
        except FileNotFoundError:
            logger.warning("node binary not found on PATH — returning empty list")
            return []
        except subprocess.TimeoutExpired:
            logger.warning("understat scrape timed out after %ds", self.timeout + 15)
            return []
        finally:
            try:
                script_path.unlink()
            except OSError:
                pass

        if result.returncode != 0:
            logger.warning(
                "understat scrape exited %d: %s",
                result.returncode,
                (result.stderr or "").strip()[:500],
            )
            return []

        try:
            raw = json.loads(result.stdout or "[]")
        except json.JSONDecodeError as exc:
            logger.warning("understat scrape returned invalid JSON: %s", exc)
            return []

        snapshot = _utc_now_iso()
        teams: list[TeamXG] = []
        for row in raw:
            try:
                teams.append(
                    TeamXG(
                        rank=int(row.get("rank") or 0),
                        team=str(row.get("team") or ""),
                        matches_played=int(row.get("matches_played") or 0),
                        wins=int(row.get("wins") or 0),
                        draws=int(row.get("draws") or 0),
                        losses=int(row.get("losses") or 0),
                        goals_for=int(row.get("goals_for") or 0),
                        goals_against=int(row.get("goals_against") or 0),
                        points=int(row.get("points") or 0),
                        xg_for=float(row.get("xg_for") or 0.0),
                        xg_diff=float(row.get("xg_diff") or 0.0),
                        xg_against=float(row.get("xg_against") or 0.0),
                        xga_diff=float(row.get("xga_diff") or 0.0),
                        xpts=float(row.get("xpts") or 0.0),
                        xpts_diff=float(row.get("xpts_diff") or 0.0),
                        snapshot_at=snapshot,
                    )
                )
            except (TypeError, ValueError) as exc:
                logger.warning("skipping malformed row %r: %s", row, exc)
                continue

        return teams

    def fetch_team(self, team_name: str) -> TeamXG | None:
        """Return one team's row (case-insensitive), or ``None`` if absent."""
        needle = (team_name or "").strip().lower()
        if not needle:
            return None
        for team in self.fetch_league_table():
            if team.team.strip().lower() == needle:
                return team
        return None

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def to_anchor_dict(self, team: TeamXG) -> dict:
        """Format a TeamXG as a context anchor for Angel.superforecast().

        The Angel's ``superforecast`` call accepts a ``context`` dict whose
        keys become soft prior anchors.  We pull out the fields most useful
        for football modelling: xG rate per match, defensive xG rate per
        match, xG differential (team attack vs defence), and the over/
        under-performance signal.
        """
        mp = max(team.matches_played, 1)
        return {
            "team": team.team,
            "league": self.league,
            "snapshot_at": team.snapshot_at,
            "matches_played": team.matches_played,
            "xg_for_per_match": round(team.xg_for / mp, 3),
            "xg_against_per_match": round(team.xg_against / mp, 3),
            "xg_differential_per_match": round(
                (team.xg_for - team.xg_against) / mp, 3
            ),
            "xg_luck": team.xg_diff,
            "xga_luck": team.xga_diff,
            "xpts": team.xpts,
            "xpts_diff": team.xpts_diff,
            "actual_points": team.points,
            "actual_goal_difference": team.goals_for - team.goals_against,
        }

    # ------------------------------------------------------------------
    # JSON round-trip
    # ------------------------------------------------------------------

    def save(self, teams: list[TeamXG], path: Path) -> None:
        """Write a list of TeamXG rows to ``path`` as pretty JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "league": self.league,
            "saved_at": _utc_now_iso(),
            "teams": [t.to_dict() for t in teams],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.info("saved %d understat rows to %s", len(teams), path)

    def load(self, path: Path) -> list[TeamXG]:
        """Read a list of TeamXG rows back from ``path``."""
        path = Path(path)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            logger.warning("could not load understat snapshot %s: %s", path, exc)
            return []
        teams: list[TeamXG] = []
        for row in raw.get("teams", []):
            try:
                teams.append(TeamXG.from_dict(row))
            except (TypeError, ValueError) as exc:
                logger.warning("skipping malformed saved row %r: %s", row, exc)
                continue
        return teams


# ---------------------------------------------------------------------------
# Self-test / live probe
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    feed = UnderstatFeed()
    teams = feed.fetch_league_table()

    if not teams:
        print(
            "no data — verify Playwright at /opt/pw-browsers/chromium-1194/"
        )
        raise SystemExit(0)

    print(f"fetched {len(teams)} teams from understat.com/league/{feed.league}")
    print()
    print("top 5 by xG for:")
    top_by_xg = sorted(teams, key=lambda t: t.xg_for, reverse=True)[:5]
    for t in top_by_xg:
        sign = "+" if t.xpts_diff >= 0 else ""
        print(
            f"  {t.rank:>2}. {t.team:<20} "
            f"xG={t.xg_for:6.2f}  "
            f"xPTS_diff={sign}{t.xpts_diff:5.2f}"
        )

    print()
    arsenal = feed.fetch_team("Arsenal")
    if arsenal is None:
        print("Arsenal: not found in table")
    else:
        print("Arsenal full record:")
        for k, v in arsenal.to_dict().items():
            print(f"  {k}: {v}")
