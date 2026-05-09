"""
QuantBot AI Trading Agent
The main AI agent powered by Anthropic's Claude claude-sonnet-4-6 model.
Uses ReAct pattern (Thought → Action → Observation) with Anthropic tool_use format.

System prompt is written in Korean as required.
"""

import json
import logging
from typing import Optional

import anthropic

from config.settings import ANTHROPIC_API_KEY, PAPER_TRADING
from agent.guardrails import Guardrails, TradeProposal
from agent.portfolio import Portfolio

logger = logging.getLogger(__name__)

# ─── System Prompt (Korean) ───────────────────────────────────────────────────
SYSTEM_PROMPT = """당신은 QuantBot입니다 — 세계 최고 수준의 AI 자산 운용 매니저입니다.

## 역할 및 페르소나
당신은 20년 이상의 글로벌 투자 경험을 가진 퀀트 전문가로서, 다음과 같은 역할을 수행합니다:
- **시장 분석가**: 주식, 암호화폐, 거시경제 지표를 종합적으로 분석합니다
- **리스크 매니저**: 포트폴리오 리스크를 철저히 관리하고 손실을 최소화합니다
- **전략가**: 모멘텀 전략과 평균 회귀 전략을 상황에 맞게 적용합니다
- **AI 트레이더**: 데이터 기반의 객관적 투자 결정을 내립니다

## 사고 방식 (ReAct 패턴)
모든 분석과 결정은 다음 단계를 따릅니다:

**Thought (생각)**: 현재 상황과 문제를 명확히 파악합니다
**Action (행동)**: 필요한 데이터를 수집하고 분석 도구를 활용합니다
**Observation (관찰)**: 결과를 분석하고 패턴을 식별합니다
**Decision (결정)**: 데이터에 기반한 합리적 투자 결정을 내립니다

## 핵심 원칙
1. **데이터 우선주의**: 모든 결정은 실제 데이터와 지표에 근거합니다
2. **리스크 관리**: 포지션 크기는 항상 허용 한도 내에서 관리합니다
3. **분산 투자**: 단일 자산에 과도하게 집중하지 않습니다
4. **감정 배제**: 시장 공포나 탐욕에 흔들리지 않고 객관적으로 판단합니다
5. **규정 준수**: 모든 거래는 가드레일 시스템을 통과해야 합니다

## 분석 프레임워크
### 기술적 분석
- RSI (상대강도지수): 14일 기준, 35 이하 매수 신호, 70 이상 매도 신호
- MACD: 12/26/9 설정으로 추세 전환 포착
- 볼린저 밴드: 20일/2표준편차로 과매수/과매도 판단
- 이동평균: 20일/50일 이동평균선 크로스오버

### 거시경제 분석
- 연방기금금리 (FEDFUNDS): 통화정책 방향 파악
- 소비자물가지수 (CPI): 인플레이션 수준 모니터링
- 수익률 곡선 (T10Y2Y): 경기 침체 신호 감지
- VIX 지수: 시장 변동성 및 공포 지수
- 실업률 (UNRATE): 경제 건강 상태 파악

### 감성 분석
- 뉴스 헤드라인의 감성 점수 (-1: 극도 부정, +1: 극도 긍정)
- 주요 금융 키워드 분석 (상승: surge, rally, gain / 하락: crash, decline, bear)

## 투자 가능 자산
### 주식 (Top 10 대형주)
AAPL(애플), MSFT(마이크로소프트), GOOGL(구글), AMZN(아마존), NVDA(엔비디아),
META(메타), TSLA(테슬라), BRK-B(버크셔해서웨이), JPM(JP모건), V(비자)

### 암호화폐 (Top 5)
BTCUSDT(비트코인), ETHUSDT(이더리움), BNBUSDT(BNB), SOLUSDT(솔라나), XRPUSDT(리플)

## 리스크 한도
- 단일 포지션 최대 비중: 포트폴리오의 10%
- 최대 허용 낙폭 (MDD): 15%
- 기본 운용 모드: 페이퍼 트레이딩 (실제 거래 없음)

## 응답 스타일
- 분석 결과는 명확하고 구체적으로 설명합니다
- 투자 제안 시 근거와 리스크를 함께 제시합니다
- 불확실한 경우 솔직하게 인정하고 추가 데이터 수집을 제안합니다
- 한국어로 주로 소통하되, 기술적 용어는 영어를 병기합니다

지금부터 당신은 QuantBot으로서 사용자의 투자 결정을 지원합니다.
항상 체계적이고 데이터 기반의 분석을 제공하세요."""

# ─── Tool Definitions (Anthropic tool_use format) ─────────────────────────────
QUANTBOT_TOOLS = [
    {
        "name": "get_market_data",
        "description": (
            "현재 시장 데이터를 가져옵니다. 주식, 암호화폐 가격, 거시경제 지표, "
            "뉴스 감성 분석을 포함한 전체 시장 스냅샷을 반환합니다. "
            "Fetches the current market snapshot including stocks, crypto, macro, and sentiment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "force_refresh": {
                    "type": "boolean",
                    "description": "캐시를 무시하고 데이터를 강제로 재수집할지 여부 (default: false)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "run_strategy_analysis",
        "description": (
            "특정 자산에 대해 모멘텀 전략과 평균 회귀 전략을 실행합니다. "
            "RSI, MACD, 볼린저 밴드를 계산하고 매수/매도/보유 신호를 생성합니다. "
            "Runs momentum and mean reversion strategies on an asset and returns trading signals."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "분석할 자산 심볼 (예: AAPL, BTCUSDT)",
                },
                "period": {
                    "type": "string",
                    "description": "분석 기간 (예: 1y, 6mo, 3mo). 기본값: 1y",
                    "enum": ["3mo", "6mo", "1y", "2y"],
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_portfolio_status",
        "description": (
            "현재 포트폴리오 상태를 조회합니다. "
            "보유 포지션, 현금, 수익/손실, 자산 배분, 현재 낙폭을 반환합니다. "
            "Returns current portfolio positions, cash balance, P&L, and drawdown."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "propose_trade",
        "description": (
            "거래를 제안하고 가드레일 검증을 통해 승인 여부를 확인합니다. "
            "자산 화이트리스트, 포지션 크기 한도, 낙폭 한도를 자동으로 검사합니다. "
            "Proposes a trade and validates it against all guardrails before execution."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "asset": {
                    "type": "string",
                    "description": "거래할 자산 심볼 (예: AAPL, BTCUSDT)",
                },
                "action": {
                    "type": "string",
                    "enum": ["BUY", "SELL"],
                    "description": "거래 방향: BUY(매수) 또는 SELL(매도)",
                },
                "quantity": {
                    "type": "number",
                    "description": "거래 수량 (주식 수 또는 암호화폐 단위)",
                },
                "price": {
                    "type": "number",
                    "description": "예상 실행 가격 (USD)",
                },
                "reasoning": {
                    "type": "string",
                    "description": "거래 근거 설명",
                },
            },
            "required": ["asset", "action", "quantity", "price"],
        },
    },
    {
        "name": "run_backtest",
        "description": (
            "특정 자산과 전략에 대해 백테스트를 실행합니다. "
            "샤프 비율, 소르티노 비율, 최대 낙폭, 수익률 등의 성과 지표를 반환합니다. "
            "Runs a historical backtest for a strategy and returns performance metrics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "백테스트할 자산 심볼 (예: AAPL, MSFT)",
                },
                "strategy": {
                    "type": "string",
                    "enum": ["momentum", "mean_reversion"],
                    "description": "사용할 전략: momentum(모멘텀) 또는 mean_reversion(평균 회귀)",
                },
                "period": {
                    "type": "string",
                    "enum": ["1y", "2y", "3y"],
                    "description": "백테스트 기간 (기본값: 1y)",
                },
                "initial_capital": {
                    "type": "number",
                    "description": "초기 투자금액 (USD, 기본값: 100000)",
                },
            },
            "required": ["symbol", "strategy"],
        },
    },
]


class QuantBot:
    """
    QuantBot — AI-powered trading agent using Anthropic's Claude claude-sonnet-4-6.

    Uses the ReAct pattern (Thought → Action → Observation) with Anthropic's
    tool_use format for structured tool calls.
    """

    def __init__(
        self,
        portfolio: Optional[Portfolio] = None,
        guardrails: Optional[Guardrails] = None,
    ) -> None:
        if not ANTHROPIC_API_KEY:
            logger.warning(
                "ANTHROPIC_API_KEY not set. QuantBot will use mock responses."
            )

        self.client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY or "sk-placeholder")
        self.portfolio = portfolio or Portfolio()
        self.guardrails = guardrails or Guardrails()
        self.conversation_history: list[dict] = []

        # Lazy-import data pipeline to avoid circular imports
        self._pipeline = None

    # ─── Public Methods ───────────────────────────────────────────────────────

    def analyze_market(self, user_context: str = "") -> str:
        """
        Run a full market analysis using the ReAct loop.

        Args:
            user_context: Optional context/question from the user.

        Returns:
            The agent's final analysis as a string.
        """
        if not user_context:
            user_context = (
                "현재 시장 상황을 분석하고 투자 기회를 찾아주세요. "
                "모든 허용 자산에 대한 신호를 확인하고 가장 유망한 기회를 추천해주세요."
            )

        logger.info("Starting market analysis: %s", user_context[:80])
        return self._run_agent_loop(user_context)

    def make_decision(self, context: str = "") -> str:
        """
        Run one decision-making iteration of the agent loop.

        Args:
            context: Optional market context.

        Returns:
            Agent's decision and reasoning.
        """
        if not context:
            context = (
                "포트폴리오 현황을 확인하고 현재 시장 조건에서 최적의 다음 행동을 결정해주세요."
            )
        return self._run_agent_loop(context)

    def chat(self, user_message: str) -> str:
        """
        Multi-turn conversation with the agent.

        Args:
            user_message: User's message or question.

        Returns:
            Agent's response.
        """
        return self._run_agent_loop(user_message, use_history=True)

    def reset_conversation(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared.")

    # ─── Private Agent Loop ───────────────────────────────────────────────────

    def _run_agent_loop(self, user_message: str, use_history: bool = False) -> str:
        """
        Execute the ReAct agent loop with tool use.

        Args:
            user_message: The user's input message.
            use_history:  If True, maintain multi-turn conversation history.

        Returns:
            Final text response from the agent.
        """
        if not ANTHROPIC_API_KEY:
            return self._mock_response(user_message)

        # Build messages list
        if use_history:
            messages = list(self.conversation_history)
        else:
            messages = []

        messages.append({"role": "user", "content": user_message})

        max_iterations = 10
        iteration = 0
        final_response = ""

        while iteration < max_iterations:
            iteration += 1
            logger.debug("Agent loop iteration %d", iteration)

            try:
                with self.client.messages.stream(
                    model="claude-sonnet-4-6",
                    max_tokens=4096,
                    system=SYSTEM_PROMPT,
                    tools=QUANTBOT_TOOLS,
                    messages=messages,
                ) as stream:
                    response = stream.get_final_message()

            except anthropic.AuthenticationError:
                logger.error("Invalid Anthropic API key.")
                return "오류: Anthropic API 키가 유효하지 않습니다. .env 파일을 확인해주세요."
            except anthropic.RateLimitError:
                logger.warning("Rate limit hit.")
                return "오류: API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요."
            except Exception as exc:
                logger.error("Agent loop error: %s", exc)
                return f"오류가 발생했습니다: {exc}"

            # Append assistant response to messages
            messages.append({"role": "assistant", "content": response.content})

            # Extract text from the response
            for block in response.content:
                if block.type == "text":
                    final_response = block.text

            # Check stop reason
            if response.stop_reason == "end_turn":
                break

            if response.stop_reason == "tool_use":
                # Execute all tool calls and collect results
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_result = self._execute_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(tool_result, ensure_ascii=False, default=str),
                        })

                if tool_results:
                    messages.append({"role": "user", "content": tool_results})
                else:
                    break
            else:
                break

        # Update conversation history for multi-turn conversations
        if use_history:
            self.conversation_history = messages

        return final_response or "분석을 완료할 수 없었습니다. 다시 시도해주세요."

    def _execute_tool(self, tool_name: str, tool_input: dict) -> dict:
        """
        Execute a tool call and return the result.

        Args:
            tool_name:  Name of the tool to call.
            tool_input: Input parameters for the tool.

        Returns:
            Dict containing the tool result.
        """
        logger.info("Executing tool: %s with input: %s", tool_name, tool_input)

        try:
            if tool_name == "get_market_data":
                return self._tool_get_market_data(tool_input)

            elif tool_name == "run_strategy_analysis":
                return self._tool_run_strategy_analysis(tool_input)

            elif tool_name == "get_portfolio_status":
                return self._tool_get_portfolio_status(tool_input)

            elif tool_name == "propose_trade":
                return self._tool_propose_trade(tool_input)

            elif tool_name == "run_backtest":
                return self._tool_run_backtest(tool_input)

            else:
                return {"error": f"알 수 없는 도구: {tool_name}"}

        except Exception as exc:
            logger.error("Tool execution error for %s: %s", tool_name, exc)
            return {"error": f"도구 실행 오류: {exc}"}

    # ─── Tool Implementations ─────────────────────────────────────────────────

    def _tool_get_market_data(self, inputs: dict) -> dict:
        """Fetch current market snapshot."""
        pipeline = self._get_pipeline()
        force = inputs.get("force_refresh", False)

        try:
            snapshot = pipeline.collect_all(force=force)
            return {
                "status": "success",
                "data": snapshot.to_summary(),
                "snapshot_age_seconds": round(snapshot.age_seconds(), 1),
                "paper_trading": PAPER_TRADING,
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    def _tool_run_strategy_analysis(self, inputs: dict) -> dict:
        """Run momentum and mean reversion strategies on an asset."""
        from strategies.momentum import MomentumStrategy
        from strategies.mean_reversion import MeanReversionStrategy

        symbol = inputs.get("symbol", "AAPL").upper()
        period = inputs.get("period", "1y")

        pipeline = self._get_pipeline()

        try:
            price_df = pipeline.get_price_history(symbol, period=period)

            momentum = MomentumStrategy()
            mr = MeanReversionStrategy()

            mom_signal = momentum.generate_signal(price_df, symbol)
            mr_signal = mr.generate_signal(price_df, symbol)

            # Determine consensus signal
            signals = [mom_signal.action.value, mr_signal.action.value]
            buy_count = signals.count("BUY")
            sell_count = signals.count("SELL")

            if buy_count == 2:
                consensus = "STRONG BUY"
            elif sell_count == 2:
                consensus = "STRONG SELL"
            elif buy_count == 1:
                consensus = "WEAK BUY"
            elif sell_count == 1:
                consensus = "WEAK SELL"
            else:
                consensus = "HOLD"

            return {
                "status": "success",
                "symbol": symbol,
                "period": period,
                "data_points": len(price_df),
                "momentum_signal": {
                    "action": mom_signal.action.value,
                    "confidence": mom_signal.confidence,
                    "reasoning": mom_signal.reasoning,
                    "indicators": mom_signal.metadata,
                },
                "mean_reversion_signal": {
                    "action": mr_signal.action.value,
                    "confidence": mr_signal.confidence,
                    "reasoning": mr_signal.reasoning,
                    "indicators": mr_signal.metadata,
                },
                "consensus": consensus,
            }
        except Exception as exc:
            return {"status": "error", "symbol": symbol, "message": str(exc)}

    def _tool_get_portfolio_status(self, _inputs: dict) -> dict:
        """Return current portfolio state."""
        summary = self.portfolio.get_summary()
        positions_df = self.portfolio.get_positions_df()
        trades_df = self.portfolio.get_trade_log_df()

        return {
            "status": "success",
            "portfolio": summary,
            "positions": positions_df.to_dict(orient="records"),
            "recent_trades": trades_df.tail(5).to_dict(orient="records"),
        }

    def _tool_propose_trade(self, inputs: dict) -> dict:
        """Propose a trade and validate against guardrails."""
        asset = inputs.get("asset", "").upper()
        action = inputs.get("action", "BUY").upper()
        quantity = float(inputs.get("quantity", 0))
        price = float(inputs.get("price", 0))
        reasoning = inputs.get("reasoning", "No reason provided.")

        proposal = TradeProposal(
            asset=asset,
            action=action,
            quantity=quantity,
            price=price,
            reasoning=reasoning,
        )

        approved, rejection_reason = self.guardrails.validate_trade(proposal, self.portfolio)

        if approved:
            # Execute the paper trade
            if action == "BUY":
                success = self.portfolio.add_position(asset, quantity, price, note=reasoning)
                if success:
                    return {
                        "status": "executed",
                        "action": action,
                        "asset": asset,
                        "quantity": quantity,
                        "price": price,
                        "trade_value": round(quantity * price, 2),
                        "paper_trading": PAPER_TRADING,
                        "message": f"[PAPER] {action} {quantity:.4f} x {asset} @ ${price:.2f} 실행 완료",
                    }
                else:
                    return {
                        "status": "failed",
                        "message": f"매수 실패: 잔고 부족 (필요: ${quantity * price:.2f})",
                    }
            elif action == "SELL":
                success = self.portfolio.remove_position(asset, quantity, price, note=reasoning)
                if success:
                    return {
                        "status": "executed",
                        "action": action,
                        "asset": asset,
                        "quantity": quantity,
                        "price": price,
                        "paper_trading": PAPER_TRADING,
                        "message": f"[PAPER] {action} {quantity:.4f} x {asset} @ ${price:.2f} 실행 완료",
                    }
                else:
                    return {
                        "status": "failed",
                        "message": f"매도 실패: {asset} 포지션이 없거나 수량이 부족합니다.",
                    }
        else:
            return {
                "status": "rejected",
                "action": action,
                "asset": asset,
                "rejection_reason": rejection_reason,
                "guardrails_limits": self.guardrails.limits,
            }

    def _tool_run_backtest(self, inputs: dict) -> dict:
        """Run a backtest and return performance metrics."""
        from backtesting.engine import BacktestEngine
        from strategies.momentum import MomentumStrategy
        from strategies.mean_reversion import MeanReversionStrategy

        symbol = inputs.get("symbol", "AAPL").upper()
        strategy_name = inputs.get("strategy", "momentum").lower()
        period = inputs.get("period", "1y")
        initial_capital = float(inputs.get("initial_capital", 100_000.0))

        pipeline = self._get_pipeline()

        try:
            price_df = pipeline.get_price_history(symbol, period=period)

            if strategy_name == "momentum":
                strategy = MomentumStrategy()
            elif strategy_name == "mean_reversion":
                strategy = MeanReversionStrategy()
            else:
                return {"status": "error", "message": f"알 수 없는 전략: {strategy_name}"}

            engine = BacktestEngine()
            result = engine.run(strategy, price_df, initial_capital=initial_capital)

            return {
                "status": "success",
                "symbol": symbol,
                "strategy": strategy_name,
                "period": period,
                "initial_capital": result.initial_capital,
                "final_value": result.final_value,
                "total_return_pct": round(result.total_return * 100, 2),
                "annualised_return_pct": round(result.annualised_return * 100, 2),
                "sharpe_ratio": result.sharpe,
                "sortino_ratio": result.sortino,
                "max_drawdown_pct": round(result.max_drawdown * 100, 2),
                "volatility_pct": round(result.volatility * 100, 2),
                "trade_count": result.trade_count,
                "win_rate_pct": round(result.win_rate * 100, 2),
            }
        except Exception as exc:
            return {"status": "error", "symbol": symbol, "message": str(exc)}

    # ─── Helper Methods ───────────────────────────────────────────────────────

    def _get_pipeline(self):
        """Lazy-load the DataPipeline to avoid circular imports."""
        if self._pipeline is None:
            from data.pipeline import DataPipeline
            self._pipeline = DataPipeline()
        return self._pipeline

    def _mock_response(self, user_message: str) -> str:
        """Return a mock response when no API key is set."""
        return (
            "⚠️  Anthropic API 키가 설정되지 않았습니다.\n\n"
            "실제 AI 분석을 사용하려면 .env 파일에 ANTHROPIC_API_KEY를 설정해주세요.\n\n"
            f"요청하신 내용: {user_message[:100]}...\n\n"
            "**모의 분석 결과:**\n"
            "- AAPL: RSI=42.3, MACD 상승 교차 → 매수 신호 (신뢰도: 0.68)\n"
            "- MSFT: RSI=71.2, 볼린저 상단 돌파 → 매도 신호 (신뢰도: 0.74)\n"
            "- BTCUSDT: RSI=58.1, 중립 구간 → 보유 유지\n\n"
            "실제 분석을 위해 API 키를 설정해주세요."
        )
