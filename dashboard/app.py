"""
QuantBot Streamlit Dashboard
Interactive web dashboard for portfolio monitoring, price charts,
strategy signals, and backtest results.

Run with: streamlit run dashboard/app.py
Or via:   python main.py dashboard
"""

import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Optional

# Add project root to path so imports work when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


def run() -> None:
    """Launch the Streamlit dashboard."""
    import subprocess
    dashboard_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    subprocess.run(["streamlit", "run", dashboard_path], check=True)


def _build_dashboard() -> None:
    """Build and render the full Streamlit dashboard."""
    import streamlit as st
    import plotly.graph_objects as go
    import pandas as pd
    import numpy as np

    # ── Page Configuration ─────────────────────────────────────────────────
    st.set_page_config(
        page_title="QuantBot Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📊 QuantBot Dashboard")
    st.caption("AI-powered Quantitative Trading Agent — Real-time Market Intelligence")

    # ── Sidebar Controls ───────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        # Asset selector
        from config.settings import ALLOWED_STOCKS, ALLOWED_CRYPTOS
        all_assets = ALLOWED_STOCKS + ALLOWED_CRYPTOS
        selected_asset = st.selectbox(
            "Select Asset",
            options=all_assets,
            index=0,
            help="Choose an asset to view price chart and signals.",
        )

        # Date range
        end_date = datetime.today()
        start_date = end_date - timedelta(days=365)
        date_range = st.date_input(
            "Date Range",
            value=(start_date, end_date),
            help="Select the date range for analysis.",
        )

        # Risk level
        risk_level = st.select_slider(
            "Risk Level",
            options=["conservative", "neutral", "aggressive"],
            value="neutral",
            help="Adjust strategy parameters based on your risk tolerance.",
        )

        # Refresh button
        st.divider()
        refresh = st.button("🔄 Refresh Data", use_container_width=True)

        st.divider()
        st.info(
            "ℹ️ **Paper Trading Mode**\n\n"
            "All trades are simulated. No real money is used."
        )

    # ── Load Data ──────────────────────────────────────────────────────────
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def load_market_data():
        from data.pipeline import DataPipeline
        pipeline = DataPipeline()
        try:
            return pipeline.collect_all()
        except Exception as e:
            logger.error("Failed to load market data: %s", e)
            return None

    @st.cache_data(ttl=300)
    def load_price_history(symbol: str, period: str = "1y"):
        from data.pipeline import DataPipeline
        pipeline = DataPipeline()
        try:
            return pipeline.get_price_history(symbol, period=period)
        except Exception as e:
            logger.error("Failed to load price history for %s: %s", symbol, e)
            return None

    @st.cache_data(ttl=300)
    def run_strategy_signals(symbol: str):
        from data.pipeline import DataPipeline
        from strategies.momentum import MomentumStrategy
        from strategies.mean_reversion import MeanReversionStrategy
        pipeline = DataPipeline()
        try:
            price_df = pipeline.get_price_history(symbol)
            mom = MomentumStrategy()
            mr = MeanReversionStrategy()
            return {
                "momentum": mom.generate_signal(price_df, symbol),
                "mean_reversion": mr.generate_signal(price_df, symbol),
            }
        except Exception as e:
            logger.error("Strategy error: %s", e)
            return None

    if refresh:
        st.cache_data.clear()
        st.rerun()

    # ── Section 1: Portfolio Overview ──────────────────────────────────────
    st.header("💼 Portfolio Overview")

    # Use a demo portfolio for the dashboard
    from agent.portfolio import Portfolio
    from config.settings import DEFAULT_INITIAL_CAPITAL
    portfolio = Portfolio(initial_capital=DEFAULT_INITIAL_CAPITAL)

    # Add some demo positions for visual richness
    portfolio.add_position("AAPL", 10, 182.5)
    portfolio.add_position("MSFT", 5, 383.0)
    portfolio.add_position("NVDA", 3, 855.0)

    summary = portfolio.get_summary()
    pnl = portfolio.calculate_pnl()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta_color = "normal" if pnl["total_pnl"] >= 0 else "inverse"
        st.metric(
            label="💰 Total Value",
            value=f"${summary['total_value']:,.2f}",
            delta=f"{pnl['total_return_pct']:+.2f}%",
            delta_color=delta_color,
        )

    with col2:
        daily_pnl = pnl["unrealised_pnl"]
        st.metric(
            label="📈 Unrealised P&L",
            value=f"${daily_pnl:+,.2f}",
            delta=f"${daily_pnl:+,.2f}",
            delta_color="normal" if daily_pnl >= 0 else "inverse",
        )

    with col3:
        drawdown = summary["current_drawdown_pct"]
        st.metric(
            label="⚠️ Current Drawdown",
            value=f"{drawdown:.2f}%",
            delta=f"{drawdown:.2f}%",
            delta_color="inverse" if drawdown < 0 else "off",
        )

    with col4:
        st.metric(
            label="💵 Cash Available",
            value=f"${summary['cash']:,.2f}",
            delta=f"{summary['cash'] / summary['total_value'] * 100:.1f}% of portfolio",
            delta_color="off",
        )

    # ── Section 2: Price Chart ─────────────────────────────────────────────
    st.header(f"📉 Price Chart — {selected_asset}")

    with st.spinner(f"Loading {selected_asset} price data..."):
        price_df = load_price_history(selected_asset)

    if price_df is not None and len(price_df) > 0:
        # Filter by date range
        if len(date_range) == 2:
            start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
            price_df = price_df[(price_df.index >= start) & (price_df.index <= end)]

        fig = go.Figure()

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=price_df.index,
                open=price_df["open"],
                high=price_df["high"],
                low=price_df["low"],
                close=price_df["close"],
                name=selected_asset,
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
            )
        )

        # Bollinger Bands
        if len(price_df) >= 20:
            close = price_df["close"]
            middle = close.rolling(20).mean()
            std = close.rolling(20).std()
            upper = middle + 2 * std
            lower = middle - 2 * std

            fig.add_trace(go.Scatter(
                x=price_df.index, y=upper,
                name="BB Upper", line=dict(color="rgba(173,204,255,0.7)", dash="dot"),
            ))
            fig.add_trace(go.Scatter(
                x=price_df.index, y=lower,
                name="BB Lower", line=dict(color="rgba(173,204,255,0.7)", dash="dot"),
                fill="tonexty", fillcolor="rgba(173,204,255,0.1)",
            ))
            fig.add_trace(go.Scatter(
                x=price_df.index, y=middle,
                name="BB Middle (20-day MA)", line=dict(color="rgba(255,165,0,0.8)"),
            ))

        # 50-day MA
        if len(price_df) >= 50:
            ma50 = price_df["close"].rolling(50).mean()
            fig.add_trace(go.Scatter(
                x=price_df.index, y=ma50,
                name="50-day MA", line=dict(color="rgba(255,100,100,0.7)"),
            ))

        fig.update_layout(
            title=f"{selected_asset} — Candlestick Chart",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            xaxis_rangeslider_visible=False,
            height=500,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No price data available for {selected_asset}.")

    # ── Section 3: Portfolio Allocation Pie Chart ──────────────────────────
    st.header("🥧 Portfolio Allocation")

    allocation = summary["allocation"]
    if allocation:
        col_chart, col_table = st.columns([3, 2])

        with col_chart:
            labels = list(allocation.keys())
            values = list(allocation.values())
            colors = [
                "#26a69a", "#42a5f5", "#ff7043", "#ab47bc",
                "#66bb6a", "#ffa726", "#29b6f6", "#ef5350", "#78909c",
            ]

            fig_pie = go.Figure(
                data=go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.4,
                    marker=dict(colors=colors[:len(labels)]),
                    textinfo="label+percent",
                )
            )
            fig_pie.update_layout(
                title="Asset Allocation",
                height=350,
                template="plotly_dark",
                showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_table:
            positions_df = portfolio.get_positions_df()
            if not positions_df.empty:
                st.dataframe(
                    positions_df[["asset", "quantity", "avg_cost", "current_price",
                                  "market_value", "unrealised_pnl", "pnl_pct"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No open positions.")

    # ── Section 4: Recent Signals Table ───────────────────────────────────
    st.header("📡 Strategy Signals")

    with st.spinner("Running strategy analysis..."):
        signals = run_strategy_signals(selected_asset)

    if signals:
        col_mom, col_mr = st.columns(2)

        with col_mom:
            mom_sig = signals["momentum"]
            action_color = (
                "🟢" if mom_sig.action.value == "BUY"
                else "🔴" if mom_sig.action.value == "SELL"
                else "🟡"
            )
            st.subheader(f"Momentum Strategy")
            st.markdown(
                f"**Signal:** {action_color} `{mom_sig.action.value}` "
                f"(Confidence: {mom_sig.confidence:.0%})"
            )
            st.caption(mom_sig.reasoning)
            if mom_sig.metadata:
                meta_df = pd.DataFrame([mom_sig.metadata])
                st.dataframe(meta_df, use_container_width=True, hide_index=True)

        with col_mr:
            mr_sig = signals["mean_reversion"]
            action_color = (
                "🟢" if mr_sig.action.value == "BUY"
                else "🔴" if mr_sig.action.value == "SELL"
                else "🟡"
            )
            st.subheader(f"Mean Reversion Strategy")
            st.markdown(
                f"**Signal:** {action_color} `{mr_sig.action.value}` "
                f"(Confidence: {mr_sig.confidence:.0%})"
            )
            st.caption(mr_sig.reasoning)
            if mr_sig.metadata:
                meta_df = pd.DataFrame([mr_sig.metadata])
                st.dataframe(meta_df, use_container_width=True, hide_index=True)

    # ── Section 5: Backtest Results ────────────────────────────────────────
    st.header("🔬 Backtest Results")

    bt_col1, bt_col2, bt_col3 = st.columns(3)

    with bt_col1:
        bt_symbol = st.selectbox("Symbol", ALLOWED_STOCKS[:5], key="bt_symbol")
    with bt_col2:
        bt_strategy = st.selectbox("Strategy", ["momentum", "mean_reversion"], key="bt_strategy")
    with bt_col3:
        bt_period = st.selectbox("Period", ["1y", "2y"], key="bt_period")

    if st.button("▶ Run Backtest", use_container_width=True):
        with st.spinner(f"Running {bt_strategy} backtest on {bt_symbol}..."):
            try:
                from backtesting.engine import BacktestEngine
                from strategies.momentum import MomentumStrategy
                from strategies.mean_reversion import MeanReversionStrategy
                from data.pipeline import DataPipeline

                bt_pipeline = DataPipeline()
                bt_price_df = bt_pipeline.get_price_history(bt_symbol, period=bt_period)

                bt_strat = (
                    MomentumStrategy() if bt_strategy == "momentum"
                    else MeanReversionStrategy()
                )
                engine = BacktestEngine()
                result = engine.run(bt_strat, bt_price_df)

                # Metrics row
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Total Return", f"{result.total_return * 100:+.2f}%")
                m2.metric("Sharpe", f"{result.sharpe:.3f}")
                m3.metric("Sortino", f"{result.sortino:.3f}")
                m4.metric("Max DD", f"{result.max_drawdown * 100:.2f}%")
                m5.metric("Trades", str(result.trade_count))
                m6.metric("Win Rate", f"{result.win_rate * 100:.1f}%")

                # Portfolio value over time chart
                if len(result.portfolio_values) > 1:
                    fig_bt = go.Figure()
                    fig_bt.add_trace(go.Scatter(
                        x=result.portfolio_values.index,
                        y=result.portfolio_values.values,
                        name="Portfolio Value",
                        fill="tozeroy",
                        fillcolor="rgba(38,166,154,0.2)",
                        line=dict(color="#26a69a"),
                    ))
                    fig_bt.add_hline(
                        y=result.initial_capital,
                        line_dash="dash",
                        annotation_text="Initial Capital",
                        line_color="gray",
                    )
                    fig_bt.update_layout(
                        title=f"{bt_symbol} — {bt_strategy.title()} Backtest",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        height=400,
                        template="plotly_dark",
                    )
                    st.plotly_chart(fig_bt, use_container_width=True)

            except Exception as e:
                st.error(f"Backtest failed: {e}")

    # ── Market Data Footer ─────────────────────────────────────────────────
    with st.expander("🌍 Live Market Data", expanded=False):
        with st.spinner("Fetching market data..."):
            snapshot = load_market_data()

        if snapshot:
            col_s, col_c, col_m = st.columns(3)

            with col_s:
                st.subheader("📊 Stocks")
                stock_data = {
                    k: v for k, v in snapshot.stocks.items()
                    if isinstance(v, dict) and "price" in v
                }
                if stock_data:
                    rows = [
                        {"Symbol": k, "Price": f"${v['price']:,.2f}",
                         "Change%": f"{v.get('change_pct', 0):+.2f}%"}
                        for k, v in list(stock_data.items())[:10]
                    ]
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            with col_c:
                st.subheader("🪙 Crypto")
                if snapshot.crypto:
                    rows = [
                        {"Symbol": k, "Price": f"${v:,.2f}"}
                        for k, v in snapshot.crypto.items()
                    ]
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            with col_m:
                st.subheader("📉 Macro Indicators")
                if snapshot.macro:
                    rows = [
                        {"Indicator": k,
                         "Value": f"{v.get('value', 0):.2f}",
                         "Description": v.get("description", "")[:30]}
                        for k, v in snapshot.macro.items()
                    ]
                    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            # Sentiment indicator
            sentiment = snapshot.sentiment.get("overall_sentiment", 0)
            st.subheader("🗞️ Market Sentiment")
            col_sent1, col_sent2 = st.columns([2, 1])
            with col_sent1:
                st.progress((sentiment + 1) / 2)  # normalize -1..1 to 0..1
            with col_sent2:
                s_label = "Bullish 📈" if sentiment > 0.1 else ("Bearish 📉" if sentiment < -0.1 else "Neutral 🔄")
                st.write(f"{s_label} ({sentiment:+.2f})")


# ── Entry Point ──────────────────────────────────────────────────────────────

def main():
    """Called by Streamlit when running the dashboard directly."""
    _build_dashboard()


if __name__ == "__main__":
    main()
else:
    # When loaded as a module by streamlit, build the dashboard immediately
    try:
        import streamlit as st
        _build_dashboard()
    except ModuleNotFoundError:
        pass  # streamlit not yet available during import
