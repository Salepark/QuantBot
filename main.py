"""
QuantBot CLI Entry Point
========================
An AI-powered quantitative trading agent that analyzes markets,
generates trading signals, runs backtests, and manages a paper portfolio.

Usage Examples:
    # Run market analysis with QuantBot AI agent
    python main.py analyze

    # Analyze with a specific user context
    python main.py analyze --context "NVDA에 대한 모멘텀 신호를 확인해주세요"

    # Run backtest on a specific symbol and strategy
    python main.py backtest --symbol AAPL --strategy momentum --period 1y

    # Launch the Streamlit dashboard
    python main.py dashboard

    # Run a full demo showcasing all components
    python main.py demo

    # Interactive multi-turn chat with the agent
    python main.py chat
"""

import argparse
import logging
import sys
import os

# ── Ensure the project root is on the path ────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Load .env before any imports that read settings ───────────────────────────
from dotenv import load_dotenv
load_dotenv()

# ── Set up logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("quantbot.main")


# ─── Command Handlers ─────────────────────────────────────────────────────────

def cmd_analyze(args: argparse.Namespace) -> None:
    """Run market analysis with the QuantBot AI agent."""
    print("\n" + "=" * 60)
    print("  QuantBot — AI Market Analysis")
    print("=" * 60)

    from agent.quantbot import QuantBot
    from agent.portfolio import Portfolio
    from agent.guardrails import Guardrails

    portfolio = Portfolio()
    guardrails = Guardrails()
    bot = QuantBot(portfolio=portfolio, guardrails=guardrails)

    context = getattr(args, "context", None) or (
        "현재 시장 상황을 분석하고 가장 유망한 투자 기회를 알려주세요. "
        "모멘텀과 평균 회귀 신호를 모두 확인해주세요."
    )

    print(f"\n📊 분석 요청: {context}\n")
    print("🤖 QuantBot 분석 중...\n")

    response = bot.analyze_market(user_context=context)
    print(response)
    print()


def cmd_backtest(args: argparse.Namespace) -> None:
    """Run a backtest on a strategy."""
    symbol = getattr(args, "symbol", "AAPL").upper()
    strategy_name = getattr(args, "strategy", "momentum").lower()
    period = getattr(args, "period", "1y")
    capital = float(getattr(args, "capital", 100_000))

    print("\n" + "=" * 60)
    print(f"  QuantBot — Backtest: {symbol} / {strategy_name}")
    print("=" * 60)

    from data.pipeline import DataPipeline
    from backtesting.engine import BacktestEngine
    from strategies.momentum import MomentumStrategy
    from strategies.mean_reversion import MeanReversionStrategy

    print(f"\n📈 심볼: {symbol}")
    print(f"🎯 전략: {strategy_name}")
    print(f"📅 기간: {period}")
    print(f"💰 초기 자본: ${capital:,.2f}\n")
    print("Loading price data...")

    pipeline = DataPipeline()
    price_df = pipeline.get_price_history(symbol, period=period)

    print(f"Loaded {len(price_df)} data points for {symbol}.\n")
    print("Running backtest...")

    if strategy_name == "momentum":
        strategy = MomentumStrategy()
    elif strategy_name == "mean_reversion":
        strategy = MeanReversionStrategy()
    else:
        print(f"❌ Unknown strategy: {strategy_name}. Use 'momentum' or 'mean_reversion'.")
        sys.exit(1)

    engine = BacktestEngine()
    result = engine.run(strategy, price_df, initial_capital=capital)
    engine.print_report(result)


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Launch the Streamlit dashboard."""
    print("\n" + "=" * 60)
    print("  QuantBot — Launching Dashboard")
    print("=" * 60)
    print("\n🚀 Starting Streamlit dashboard...")
    print("   Open http://localhost:8501 in your browser.\n")

    import subprocess
    dashboard_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dashboard", "app.py"
    )
    try:
        subprocess.run(["streamlit", "run", dashboard_path], check=True)
    except FileNotFoundError:
        print("❌ streamlit not found. Install with: pip install streamlit")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n👋 Dashboard stopped.")


def cmd_chat(args: argparse.Namespace) -> None:
    """Interactive multi-turn chat with the QuantBot agent."""
    print("\n" + "=" * 60)
    print("  QuantBot — Interactive Chat Mode")
    print("=" * 60)
    print("\n💬 QuantBot과 대화하세요. 종료하려면 'quit' 또는 'exit'를 입력하세요.\n")

    from agent.quantbot import QuantBot
    from agent.portfolio import Portfolio
    from agent.guardrails import Guardrails

    portfolio = Portfolio()
    guardrails = Guardrails()
    bot = QuantBot(portfolio=portfolio, guardrails=guardrails)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 채팅을 종료합니다.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "종료", "q"):
            print("\n👋 채팅을 종료합니다.")
            break

        if user_input.lower() in ("reset", "clear", "초기화"):
            bot.reset_conversation()
            print("🔄 대화 기록이 초기화되었습니다.\n")
            continue

        print("\n🤖 QuantBot: ", end="", flush=True)
        response = bot.chat(user_input)
        print(response)
        print()


def cmd_demo(args: argparse.Namespace) -> None:
    """Run a full demonstration of all QuantBot components."""
    print("\n" + "=" * 60)
    print("  QuantBot — Full System Demo")
    print("=" * 60)

    # ── 1. Data Collection ──────────────────────────────────────────────
    print("\n[1/6] 📥 Data Collection")
    print("-" * 40)
    from data.pipeline import DataPipeline
    pipeline = DataPipeline()

    try:
        snapshot = pipeline.collect_all()
        summary = snapshot.to_summary()
        print(f"✅ Market snapshot collected (age: {snapshot.age_seconds():.0f}s)")
        print(f"   Stocks: {len(summary['stocks'])} symbols")
        print(f"   Crypto: {len(summary['crypto'])} symbols")
        print(f"   Macro indicators: {len(summary['macro'])}")
        print(f"   Market sentiment: {summary['sentiment']:+.3f}")
    except Exception as e:
        print(f"⚠️  Data collection partial: {e}")

    # ── 2. Strategy Signals ──────────────────────────────────────────────
    print("\n[2/6] 📊 Strategy Analysis (AAPL)")
    print("-" * 40)
    from strategies.momentum import MomentumStrategy
    from strategies.mean_reversion import MeanReversionStrategy

    price_df = pipeline.get_price_history("AAPL")
    mom = MomentumStrategy()
    mr = MeanReversionStrategy()

    mom_signal = mom.generate_signal(price_df, "AAPL")
    mr_signal = mr.generate_signal(price_df, "AAPL")

    print(f"  Momentum:      {mom_signal.action.value:4s} | confidence={mom_signal.confidence:.2f}")
    print(f"                 {mom_signal.reasoning[:80]}...")
    print(f"  Mean Reversion:{mr_signal.action.value:4s} | confidence={mr_signal.confidence:.2f}")
    print(f"                 {mr_signal.reasoning[:80]}...")

    # ── 3. Backtest ──────────────────────────────────────────────────────
    print("\n[3/6] 🔬 Backtesting (AAPL, Momentum, 1y)")
    print("-" * 40)
    from backtesting.engine import BacktestEngine
    engine = BacktestEngine()
    result = engine.run(MomentumStrategy(), price_df)
    print(f"  Total Return  : {result.total_return * 100:+.2f}%")
    print(f"  Sharpe Ratio  : {result.sharpe:.4f}")
    print(f"  Max Drawdown  : {result.max_drawdown * 100:.2f}%")
    print(f"  Trades        : {result.trade_count}")
    print(f"  Win Rate      : {result.win_rate * 100:.1f}%")

    # ── 4. Portfolio Management ──────────────────────────────────────────
    print("\n[4/6] 💼 Portfolio Management")
    print("-" * 40)
    from agent.portfolio import Portfolio
    portfolio = Portfolio(initial_capital=100_000)

    portfolio.add_position("AAPL", 10, 182.5, note="Demo buy")
    portfolio.add_position("MSFT", 5, 383.0, note="Demo buy")
    portfolio.add_position("NVDA", 3, 855.0, note="Demo buy")

    summary = portfolio.get_summary()
    print(f"  Total Value   : ${summary['total_value']:,.2f}")
    print(f"  Cash          : ${summary['cash']:,.2f}")
    print(f"  Invested      : ${summary['invested_value']:,.2f}")
    print(f"  P&L           : ${summary['total_pnl']:+,.2f} ({summary['total_return_pct']:+.2f}%)")
    print(f"  Drawdown      : {summary['current_drawdown_pct']:.2f}%")
    print(f"  Allocation    : {summary['allocation']}")

    # ── 5. Guardrails Test ───────────────────────────────────────────────
    print("\n[5/6] 🛡️  Guardrails Validation")
    print("-" * 40)
    from agent.guardrails import Guardrails, TradeProposal

    guardrails = Guardrails()

    # Test 1: Valid trade
    valid_trade = TradeProposal("AAPL", "BUY", 10, 185.0, "Momentum signal")
    approved, reason = guardrails.validate_trade(valid_trade, portfolio)
    print(f"  Valid trade (AAPL BUY):   {'✅ Approved' if approved else '❌ Rejected'}")

    # Test 2: Invalid asset
    invalid_trade = TradeProposal("DOGE", "BUY", 1000, 0.1, "Speculative")
    approved2, reason2 = guardrails.validate_trade(invalid_trade, portfolio)
    print(f"  Invalid asset (DOGE):     {'✅ Approved' if approved2 else '❌ Rejected'}")
    if not approved2:
        print(f"    Reason: {reason2[:60]}...")

    # Test 3: Oversized position
    oversized = TradeProposal("AAPL", "BUY", 1000, 185.0, "Too large")
    approved3, reason3 = guardrails.validate_trade(oversized, portfolio)
    print(f"  Oversized position:       {'✅ Approved' if approved3 else '❌ Rejected'}")
    if not approved3:
        print(f"    Reason: {reason3[:60]}...")

    # ── 6. Notifications Test ────────────────────────────────────────────
    print("\n[6/6] 📨 Notifications (requires API keys)")
    print("-" * 40)
    from notifications.telegram_bot import TelegramNotifier
    from notifications.slack_bot import SlackNotifier

    tg = TelegramNotifier()
    sl = SlackNotifier()

    tg_status = "✅ Configured" if tg._bot else "⚠️  Not configured (set TELEGRAM_BOT_TOKEN)"
    sl_status = "✅ Configured" if sl._client else "⚠️  Not configured (set SLACK_BOT_TOKEN)"

    print(f"  Telegram: {tg_status}")
    print(f"  Slack:    {sl_status}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ QuantBot Demo Complete!")
    print("=" * 60)
    print("\n📋 Quick Start Guide:")
    print("  • Copy .env.example to .env and fill in your API keys")
    print("  • Run: python main.py analyze        → AI market analysis")
    print("  • Run: python main.py backtest --symbol AAPL  → Backtest")
    print("  • Run: python main.py dashboard      → Web dashboard")
    print("  • Run: python main.py chat           → Chat with the agent")
    print()


# ─── CLI Argument Parser ──────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantbot",
        description="QuantBot — AI-powered Quantitative Trading Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", metavar="command")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Run AI market analysis")
    p_analyze.add_argument(
        "--context", "-c",
        type=str,
        default=None,
        help="Optional context/question for the AI agent.",
    )

    # backtest
    p_backtest = subparsers.add_parser("backtest", help="Run strategy backtest")
    p_backtest.add_argument(
        "--symbol", "-s",
        type=str,
        default="AAPL",
        help="Ticker symbol to backtest (default: AAPL).",
    )
    p_backtest.add_argument(
        "--strategy",
        type=str,
        choices=["momentum", "mean_reversion"],
        default="momentum",
        help="Strategy to use (default: momentum).",
    )
    p_backtest.add_argument(
        "--period",
        type=str,
        choices=["1y", "2y"],
        default="1y",
        help="Lookback period (default: 1y).",
    )
    p_backtest.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Initial capital in USD (default: 100000).",
    )

    # dashboard
    subparsers.add_parser("dashboard", help="Launch Streamlit dashboard")

    # chat
    subparsers.add_parser("chat", help="Interactive multi-turn chat with the agent")

    # demo
    subparsers.add_parser("demo", help="Run a full demonstration of all components")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "analyze": cmd_analyze,
        "backtest": cmd_backtest,
        "dashboard": cmd_dashboard,
        "chat": cmd_chat,
        "demo": cmd_demo,
    }

    handler = dispatch.get(args.command)
    if handler:
        try:
            handler(args)
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user.")
            sys.exit(0)
        except Exception as exc:
            logger.exception("Unhandled error: %s", exc)
            print(f"\n❌ Error: {exc}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
