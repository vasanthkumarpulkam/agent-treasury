# Agent Treasury — Self-Sustaining Polymarket + Bitcoin Trading System

An autonomous multi-agent trading system with a hard survival mechanic: it trades Polymarket
prediction markets and Bitcoin spot, tracks its own treasury, and **halts itself** if it loses
money past configured thresholds. Agents (LLM-driven research/strategy) never hold funds or
execution rights — they only produce trade *proposals*. A separate, deliberately simple,
non-LLM **Governor** process is the only thing that can approve a proposal and place a real
order, and it enforces every risk rule mechanically.

**Read this before you do anything else:**

- This ships in **PAPER mode by default**. No real order is ever placed until you set
  `LIVE_MODE=true` in `.env` *and* supply real, funded, trade-permissioned API credentials.
- There is no MCP connector that executes trades on Polymarket or a crypto exchange (checked
  the registry — only read-only price-data connectors exist, e.g. CoinDesk, Crypto.com). Live
  execution here uses the same direct SDKs every real project in this space uses:
  [`py-clob-client`](https://github.com/Polymarket/py-clob-client) for Polymarket and
  [`ccxt`](https://github.com/ccxt/ccxt) for the Bitcoin exchange (Coinbase Advanced Trade by
  default, swappable). Your API keys live only in your local `.env`, never in this repo.
- This is not financial advice and there is no guarantee of profit. The kill-switch controls
  how much you can lose and how fast the system notices it's losing — it does not manufacture
  trading edge. Run in paper mode for weeks and inspect the logged decisions before risking
  real capital.
- When the kill-switch fires, the system **pauses and requires a human to review and
  explicitly resume it** (see `resurrection_mode` in config). Don't make resumption automatic.

## Architecture

```
pods/polymarket, pods/bitcoin   -->   orchestrator   -->   governor   -->   real order
   (propose trades only)              (routes/allocates)   (approves/kills)
```

- **`pods/polymarket/`** — research agent that scores Polymarket markets for edge and emits
  sized trade proposals. No execution rights.
- **`pods/bitcoin/`** — mechanical trend + volatility-regime signal agent for BTC spot. No
  execution rights.
- **`orchestrator/`** — polls both pods on a schedule, allocates capital between legs, forwards
  proposals to the Governor, logs everything.
- **`governor/`** — the trusted core. Holds the treasury ledger, enforces position/exposure
  limits, evaluates kill-switch conditions on every settlement, and is the only module allowed
  to call an execution client.
- **`storage/`** — SQLite-backed ledger and audit log (`storage/agent_treasury.db`).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `config/settings.yaml` for starting capital, kill-switch thresholds, allocation split,
and tradeable-market allowlist.

## Running

```bash
python3 -m pytest tests/test_governor.py -v   # unit tests for the survival mechanic
python3 tests/test_integration_smoke.py       # full pipeline, fake data, no network needed
python3 main.py                               # one real orchestration cycle (paper by default)
```

Schedule `main.py` with cron/launchd for continuous operation. It is intentionally not a
single long-running unattended daemon in v1 — each run reconciles balances, checks the
kill-switch, and only then evaluates new proposals.

## Recommended path to live capital

1. Run in paper mode for at least 4-8 weeks. Inspect `storage/agent_treasury.db` to see
   whether the strategy has real edge.
2. Backtest the BTC signal agent against historical data.
3. The Polymarket agent calls OpenRouter for a real probability estimate (set
   `OPENROUTER_API_KEY` in `.env`; get one at https://openrouter.ai/keys), gated by
   `min_edge_pct` and a per-cycle spend cap (`llm_max_spend_per_cycle_usd` in config).
   Without a key it safely falls back to "no edge" (same zero-trades default as before).
   An LLM reading a market description is still just one signal — treat it as a starting
   point to validate against real outcomes, not a finished edge.
4. Start live with a small `starting_capital_usd` and tight `max_drawdown_pct`.
5. Treat every kill-switch trigger as a stop-and-review event, not a bug to route around.
