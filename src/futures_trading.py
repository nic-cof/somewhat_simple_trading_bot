import pandas as pd
import numpy as np
from typing import Dict, Optional, TypedDict, List
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
import logging
import traceback
from collections import defaultdict
# from pathlib import Path


class TradeInfo(TypedDict, total=False):
    """Type definition for trade information"""
    timestamp: pd.Timestamp
    direction: str
    price: float
    contracts: int
    confidence: float
    slippage: float
    commission: float
    pnl: float
    conditions: Dict


class MarketConditions(TypedDict, total=False):
    """Type definition for market conditions"""
    timestamp: pd.Timestamp
    is_valid_time: bool
    liquidity_score: float
    volatility_regime: str
    price_movement: Optional[float]
    session: str
    extreme_conditions: Dict[str, bool]


@dataclass
class TradingParameters:
    """Parameters for futures trading"""
    initial_capital: float
    margin_per_contract: float
    max_risk_per_trade_pct: float
    commission_per_contract: float = 0.00  # Round trip commission
    exchange_fees_per_contract: float = 1.00  # Round trip exchange fees
    nfa_fees_per_contract: float = 0.04  # Round trip NFA fees
    profit_factor_required: float = 2.0
    slippage_per_contract: float = 0.25
    volatility_scaling: bool = True
    max_position_size: int = 10
    max_trades_per_day: int = 10
    max_drawdown_pct: float = 0.10  # 10% max drawdown
    trading_hours: Dict = None
    confidence_levels: Dict[str, float] = None
    position_multipliers: Dict[str, float] = None
    liquidity_threshold: int = 50
    min_confidence: float = 0.45


@dataclass
class PositionSizeDiagnostics:
    """Enhanced diagnostics for position sizing decisions"""

    decisions: Dict = field(default_factory=lambda: {
        'total_checks': 0,
        'drawdown_filtered': 0,
        'daily_loss_filtered': 0,
        'confidence_filtered': 0,
        'volume_filtered': 0,
        'volatility_adjusted': 0,
        'liquidity_adjusted': 0,
        'successful_sizing': 0
    })

    confidence_distribution: List[float] = field(default_factory=list)
    volume_distribution: List[float] = field(default_factory=list)
    position_sizes: List[int] = field(default_factory=list)

    def log_check(self, confidence: float, volume: float, volatility: float,
                  current_drawdown: float, daily_pnl: float, result_size: int,
                  drawdown_limit: float, min_confidence: float):
        """Log a position size check with all relevant parameters"""

        self.decisions['total_checks'] += 1
        self.confidence_distribution.append(confidence)
        self.volume_distribution.append(volume)

        if result_size > 0:
            self.decisions['successful_sizing'] += 1
            self.position_sizes.append(result_size)

        # Track specific filter reasons
        if current_drawdown > drawdown_limit:
            self.decisions['drawdown_filtered'] += 1

        if daily_pnl < 0 and abs(daily_pnl) > drawdown_limit:
            self.decisions['daily_loss_filtered'] += 1

        if confidence < min_confidence:
            self.decisions['confidence_filtered'] += 1

        if volume < 20:
            self.decisions['volume_filtered'] += 1

    def get_position_diagnostics(self):
        """Get detailed summary statistics of position sizing decisions"""
        total = self.decisions['total_checks']
        if total == 0:
            return self.decisions

        return {
            'total_signals': total,
            'success_rate': self.decisions['successful_sizing'] / total * 100,
            'filter_rates': {
                'drawdown': self.decisions['drawdown_filtered'] / total * 100,
                'daily_loss': self.decisions['daily_loss_filtered'] / total * 100,
                'confidence': self.decisions['confidence_filtered'] / total * 100,
                'volume': self.decisions['volume_filtered'] / total * 100
            },
            'avg_confidence': np.mean(self.confidence_distribution) if self.confidence_distribution else 0,
            'avg_volume': np.mean(self.volume_distribution) if self.volume_distribution else 0,
            'avg_position_size': np.mean(self.position_sizes) if self.position_sizes else 0
        }


class FuturesTrader:
    """
    Enhanced futures trading implementation with comprehensive risk management.

    This class handles:
    - Position sizing and risk management
    - Trade execution and tracking
    - Performance analysis
    - Market condition monitoring
    """

    def __init__(self, params: TradingParameters):
        """
        Initialize the FuturesTrader.

        Args:
            params: TradingParameters object with trading configuration
        """
        self.params = params
        self.logger = logging.getLogger(__name__)

        # Set default trading hours if not provided
        self.trading_hours = params.trading_hours or {
            'regular': (time(8, 30), time(15, 15)),
            'overnight': [(time(15, 30), time(23, 59)),
                          (time(0, 0), time(8, 15))]
        }

        # Set default confidence levels if not provided
        self.confidence_levels = params.confidence_levels or {
            'high': 0.92,
            'medium': 0.87,
            'low': 0.82
        }

        # Set default position multipliers if not provided
        self.position_multipliers = params.position_multipliers or {
            'high': 1.0,
            'medium': 0.5,
            'low': 0.25
        }

        # Risk management parameters
        self.max_daily_loss_pct = 0.02  # 2% max daily loss
        self.max_drawdown_pct = 0.10  # 10% max drawdown
        self.position_heat = 0.25  # Reduce position size when P&L is negative
        self.liquidity_threshold = 1000  # Minimum volume for full position size

        # Initialize tracking variables
        self.metrics = {}
        self.daily_trades = {}
        self.daily_pnl = {}
        self.current_drawdown = 0
        self.high_water_mark = params.initial_capital
        self.current_capital = params.initial_capital
        self.trades_history = []
        self.last_price = None
        self.diagnostics = PositionSizeDiagnostics()
        self.active_positions = []

        self.trade_logger = self.setup_trade_logging()

    def validate_trade_conditions(self, confidence: float, volume: float, timestamp: pd.Timestamp) -> bool:
        try:
            # Add detailed logging
            self.logger.debug(f"\nValidating trade conditions:")
            self.logger.debug(f"Confidence: {confidence:.3f} (min: {self.params.min_confidence})")
            self.logger.debug(f"Volume: {volume} (min: {self.params.liquidity_threshold})")

            # Check trading hours first
            if not self.is_valid_trading_time(timestamp):
                self.logger.debug(f"Invalid trading time: {timestamp}")
                return False

            # Check confidence threshold
            if confidence < self.params.min_confidence:
                self.logger.debug(f"Low confidence: {confidence:.3f} < {self.params.min_confidence}")
                return False

            # Check volume threshold
            if volume < self.params.liquidity_threshold:
                self.logger.debug(f"Low volume: {volume} < {self.params.liquidity_threshold}")
                return False

            # Check daily loss limit
            trade_date = timestamp.date()
            daily_pnl = self.daily_pnl.get(trade_date, 0)
            max_daily_loss = -self.current_capital * self.max_daily_loss_pct

            if daily_pnl < max_daily_loss:
                self.logger.debug(f"Daily loss limit reached: {daily_pnl:.2f} < {max_daily_loss:.2f}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating trade conditions: {str(e)}")
            return False

    def calculate_position_size(self, confidence: float, volatility: float, volume: float,
                                timestamp: pd.Timestamp, atr: float,
                                avg_volume: Optional[float] = None) -> int:
        """Enhanced position sizing with dynamic scaling"""
        try:
            # Base validation
            if not self.validate_trade_conditions(confidence, volume, timestamp):
                return 0

            # More aggressive base size calculation
            if confidence >= self.confidence_levels['high']:
                base_size = min(3, self.params.max_position_size)  # Cap at max_position_size
            elif confidence >= self.confidence_levels['medium']:
                base_size = 2
            else:
                base_size = 1

            # Enhanced volatility scaling
            vol_scale = 1.0
            if volatility > 0:
                # Scale down size in high volatility, scale up in low volatility
                vol_target = 0.01  # 1% target volatility
                vol_scale = min(2.0, vol_target / volatility)
                vol_scale = max(0.5, vol_scale)  # Limit scaling range

            # Enhanced volume scaling
            vol_scale = 1.0
            if avg_volume and avg_volume > 0:
                vol_ratio = volume / avg_volume
                vol_scale = min(1.5, vol_ratio)  # Allow up to 50% increase
                vol_scale = max(0.5, vol_scale)  # Don't reduce below 50%

            # Calculate position size with all factors
            position_size = int(base_size * vol_scale * vol_scale)

            # Apply risk-based limits
            risk_per_contract = atr * 2  # Use 2 ATR for risk calculation
            max_risk = self.current_capital * self.params.max_risk_per_trade_pct
            risk_based_size = max(1, int(max_risk / (risk_per_contract * 5)))  # 5 is MES multiplier

            # Take minimum of calculated size and risk-based size
            final_size = min(
                position_size,
                risk_based_size,
                self.params.max_position_size,
                max(1, int(volume * 0.01))  # Limit to 1% of volume
            )

            return max(1, final_size)  # Ensure at least 1 contract

        except Exception as e:
            self.logger.error(f"Error in position sizing: {str(e)}")
            return 0

    def execute_trade(self, direction: int, price: float, volume: float,
                      volatility: float, timestamp: pd.Timestamp,
                      confidence: float, atr: float) -> Dict:
        """Enhanced trade execution with better validation and risk checks"""
        try:
            # Price sanity checks
            if price <= 0 or price > 10000:  # MES futures should be roughly between $500-$5500
                self.logger.error(f"Invalid price detected: ${price:.2f}")
                return {'executed': False, 'reason': 'invalid_price'}

            # Direction validation
            if direction not in [-1, 1]:
                self.logger.error(f"Invalid direction: {direction}")
                return {'executed': False, 'reason': 'invalid_direction'}

            # Add before position size calculation
            self.logger.info(f"Price before position sizing: ${price:.2f}")  # Add this

            # Log initial parameters with more detail
            self.trade_logger.info(f"\nTrade Attempt at {timestamp}:")
            self.trade_logger.info(f"Price: ${price:.2f}, Direction: {'Short' if direction == -1 else 'Long'}")
            self.trade_logger.info(f"Confidence: {confidence:.3f}, ATR: {atr:.2f}, Volume: {volume}")

            # Market condition check
            market_conditions = self.analyze_market_conditions(
                price, volatility, volume, timestamp)

            if not market_conditions['is_valid_time']:
                self.logger.info("Rejected: Invalid trading time")
                return {'executed': False, 'reason': 'invalid_trading_time'}

            # Position size calculation
            n_contracts = self.calculate_position_size(
                confidence, volatility, volume, timestamp, atr)

            self.logger.info(f"Calculated position size: {n_contracts}")

            if n_contracts == 0:
                self.logger.info("Rejected: Zero position size")
                return {'executed': False, 'reason': 'zero_position_size'}

            # Check existing positions for this timestamp
            existing_positions = [p for p in self.active_positions
                                  if p['timestamp'] == timestamp]
            if existing_positions:
                self.logger.info(f"Rejected: Already have position for this timestamp")
                return {'executed': False, 'reason': 'duplicate_timestamp'}

            # Enhanced risk checks
            trade_value = 50 * n_contracts # * self.params.margin_per_contract  # Use margin instead of notional
            position_limit = self.current_capital * 0.15  # Increased to 15%

            self.logger.info(f"Trade value: ${trade_value:.2f}, Position limit: ${position_limit:.2f}")

            if trade_value > position_limit:
                self.logger.info("Rejected: Position too large")
                return {'executed': False, 'reason': 'position_too_large'}

            # Calculate and validate costs
            commission = (self.params.commission_per_contract +
                          self.params.exchange_fees_per_contract +
                          self.params.nfa_fees_per_contract) * n_contracts

            slippage = self.calculate_slippage(
                price, volume, volatility,
                market_conditions['session'] == 'overnight'
            )

            slippage_cost = slippage * n_contracts * 5

            # Cost threshold check
            max_cost = self.current_capital * 0.002
            total_cost = commission + slippage_cost

            self.logger.info(f"Total cost: ${total_cost:.2f}, Max cost: ${max_cost:.2f}")

            if total_cost > max_cost:
                self.logger.info("Rejected: Excessive costs")
                return {'executed': False, 'reason': 'excessive_costs'}

            # Calculate stop levels
            stop_loss, profit_target = self.calculate_stop_levels(
                entry_price=price,
                atr=atr,
                direction=direction,
                confidence=confidence
            )

            self.logger.info(f"Stop loss: {stop_loss:.2f}, Profit target: {profit_target:.2f}")

            # Validate stop levels
            if stop_loss <= 0 or profit_target <= 0:
                self.logger.error(f"Invalid stop levels - Stop: ${stop_loss:.2f}, Target: ${profit_target:.2f}")
                return {'executed': False, 'reason': 'invalid_stop_levels'}

            # Final price movement check
            if direction == 1 and stop_loss >= price:  # Long position
                self.logger.error("Invalid stop loss above entry price for long position")
                return {'executed': False, 'reason': 'invalid_stop_placement'}
            elif direction == -1 and stop_loss <= price:  # Short position
                self.logger.error("Invalid stop loss below entry price for short position")
                return {'executed': False, 'reason': 'invalid_stop_placement'}

            # Add before trade creation
            self.logger.info(f"Trade creation - Raw price: ${price:.2f}")
            if not (3500 <= price <= 5500):  # 2023 MES range
                self.logger.error(f"Price outside normal range: ${price:.2f}")
                return {'executed': False, 'reason': 'price_range_error'}

            # Create trade
            trade = {
                'timestamp': timestamp,
                'direction': direction,
                'price': price,
                'entry_price': price,
                'exit_price': None,
                'contracts': n_contracts,
                'confidence': confidence,
                'slippage': slippage_cost,
                'commission': commission,
                'conditions': market_conditions,
                'status': 'open',
                'pnl': 0,
                'stop_loss': stop_loss,
                'profit_target': profit_target,
                'atr_at_entry': atr,
                'volatility_at_entry': volatility,
                'trade_value': trade_value,
                'total_cost': total_cost
            }

            # Check daily trade limits
            trade_date = timestamp.date()
            daily_trades = len([t for t in self.active_positions
                                if t['timestamp'].date() == trade_date])

            self.logger.info(f"Daily trades: {daily_trades}")

            if daily_trades >= self.params.max_trades_per_day:
                self.logger.info("Rejected: Daily trade limit reached")
                return {'executed': False, 'reason': 'daily_limit_reached'}

            # Update account state
            self.current_capital -= total_cost
            self.active_positions.append(trade)

            self.logger.info(f"Trade executed successfully - Direction: {'Short' if direction == -1 else 'Long'}, "
                             f"Price: ${price:.2f}, Contracts: {n_contracts}")
            return {'executed': True, 'trade': trade}

        except Exception as e:
            self.logger.error(f"Error executing trade: {str(e)}")
            return {'executed': False, 'reason': str(e)}

    def calculate_stop_levels(self, entry_price: float, atr: float, direction: int,
                              confidence: float) -> tuple[float, float]:
        """Calculate wider stop loss and profit target levels"""
        try:
            # Much wider stops for more room to breathe
            if confidence >= self.confidence_levels['high']:
                stop_multiple = 3.0    # Was 2.0
                target_multiple = 4.5  # Was 3.0
            elif confidence >= self.confidence_levels['medium']:
                stop_multiple = 2.5    # Was 1.5
                target_multiple = 3.75 # Was 2.5
            else:
                stop_multiple = 2.0    # Was 1.0
                target_multiple = 3.0  # Was 2.0

            # Calculate distances
            stop_distance = atr * stop_multiple
            target_distance = atr * target_multiple

            # Ensure minimum distance (increased)
            min_distance = 1.0  # Was 0.50
            stop_distance = max(stop_distance, min_distance)
            target_distance = max(target_distance, min_distance * 2)

            if direction == 1:  # Long
                stop_loss = entry_price - stop_distance
                profit_target = entry_price + target_distance
            else:  # Short
                stop_loss = entry_price + stop_distance
                profit_target = entry_price - target_distance

            return stop_loss, profit_target

        except Exception as e:
            self.logger.error(f"Error calculating stop levels: {str(e)}")
            return entry_price * 0.98, entry_price * 1.02  # Wider default stops

    def update_positions(self, current_price: float, timestamp: pd.Timestamp, current_atr: float) -> None:
        """Update and manage open positions."""
        for position in self.active_positions[:]:  # Create copy of list for iteration
            if position['status'] == 'open':
                # Calculate elapsed time
                elapsed_minutes = (timestamp - position['timestamp']).total_seconds() / 60
                max_hold_time = 240  # 4 hours

                # Calculate stop levels
                stop_loss, profit_target = self.calculate_stop_levels(
                    position['entry_price'],
                    current_atr,
                    position['direction'],
                    position['confidence']
                )

                # Check exit conditions
                should_exit = False
                exit_reason = ''

                # Time-based exit
                if elapsed_minutes >= max_hold_time:
                    should_exit = True
                    exit_reason = 'time_exit'
                    self.logger.info(f"Time exit triggered after {elapsed_minutes:.1f} minutes")

                # Price-based exits
                elif position['direction'] == 1:  # Long position
                    if current_price <= stop_loss:
                        should_exit = True
                        exit_reason = 'stop_loss'
                    elif current_price >= profit_target:
                        should_exit = True
                        exit_reason = 'profit_target'
                else:  # Short position
                    if current_price >= stop_loss:
                        should_exit = True
                        exit_reason = 'stop_loss'
                    elif current_price <= profit_target:
                        should_exit = True
                        exit_reason = 'profit_target'

                # Execute exit if conditions are met
                if should_exit:
                    position['exit_reason'] = exit_reason
                    self.close_position(position, current_price, timestamp)
                    self.active_positions.remove(position)  # Remove closed position

    def close_position(self, position: Dict, exit_price: float, timestamp: pd.Timestamp) -> None:
        """Close a position and update account balance."""
        try:
            # Validate exit price
            if exit_price <= 0 or exit_price > 10000:
                self.logger.error(f"Invalid exit price detected: ${exit_price:.2f}")
                # Use last known good price or entry price
                exit_price = position['entry_price']

            # Log position closure details
            self.logger.info(f"\nClosing position at {timestamp}:")
            self.logger.info(f"Entry price: ${position['entry_price']:.2f}")
            self.logger.info(f"Exit price: ${exit_price:.2f}")
            self.logger.info(f"Direction: {'Short' if position['direction'] == -1 else 'Long'}")

            # Calculate price difference in points
            price_diff = exit_price - position['entry_price']
            if position['direction'] == -1:  # Short position
                price_diff = -price_diff

            # Validate price movement
            max_allowed_move = position['entry_price'] * 0.10  # 10% max move
            if abs(price_diff) > max_allowed_move:
                self.logger.warning(f"Unusually large price movement detected: ${price_diff:.2f}")
                price_diff = max_allowed_move if price_diff > 0 else -max_allowed_move

            # MES multiplier is $5 per point
            contract_multiplier = 5

            # Calculate raw P&L from price movement
            position_pnl = price_diff * position['contracts'] * contract_multiplier

            # Verify reasonable P&L range (-$500 to +$500 per contract)
            max_pnl_per_contract = 500
            if abs(position_pnl / position['contracts']) > max_pnl_per_contract:
                self.logger.warning(f"Unrealistic P&L detected: ${position_pnl:.2f} "
                                    f"per contract. Capping at ${max_pnl_per_contract}")
                position_pnl = (max_pnl_per_contract if position_pnl > 0 else -max_pnl_per_contract) * position[
                    'contracts']

            # Calculate total costs if not already included
            total_costs = position.get('commission', 0) + position.get('slippage', 0)

            # Calculate final P&L
            final_pnl = position_pnl - total_costs

            # Calculate trade duration
            duration_td = timestamp - position['timestamp']
            duration_minutes = duration_td.total_seconds() / 60

            # Validate duration
            if duration_minutes < 0:
                self.logger.error(f"Invalid trade duration: {duration_minutes} minutes")
                duration_minutes = 0

            # Update position information
            position.update({
                'exit_price': exit_price,
                'exit_time': timestamp,
                'duration': duration_minutes,
                'raw_pnl': position_pnl,
                'total_costs': total_costs,
                'pnl': final_pnl,
                'status': 'closed',
                'price_movement': price_diff
            })

            # Update account balance
            self.current_capital += final_pnl

            # Update daily P&L tracking
            trade_date = timestamp.date()
            if trade_date not in self.daily_pnl:
                self.daily_pnl[trade_date] = 0
            self.daily_pnl[trade_date] += final_pnl

            # Update high water mark and drawdown
            self.high_water_mark = max(self.high_water_mark, self.current_capital)
            self.current_drawdown = max(0, (self.high_water_mark - self.current_capital) / self.high_water_mark)

            # Store completed trade
            self.trades_history.append(position)

            # Log trade details with enhanced information
            self.logger.info(
                f"Closed position: Entry=${position['entry_price']:.2f}, "
                f"Exit=${exit_price:.2f}, Direction={'Short' if position['direction'] == -1 else 'Long'}, "
                f"Contracts={position['contracts']}, Raw_PnL=${position_pnl:.2f}, "
                f"Costs=${total_costs:.2f}, Final_PnL=${final_pnl:.2f}, "
                f"Duration={duration_minutes:.1f}min"
            )

        except Exception as e:
            self.logger.error(f"Error closing position: {str(e)}")
            self.logger.error(f"Position details: {position}")
            raise

    def validate_position_value(self, price: float, n_contracts: int) -> bool:
        position_value = 50 * n_contracts * self.params.margin_per_contract
        max_position_value = self.current_capital * 0.1  # Max 10% of capital
        return position_value <= max_position_value

    def analyze_market_conditions(self,
                                  price: float,
                                  volatility: float,
                                  volume: float,
                                  timestamp: pd.Timestamp) -> MarketConditions:
        """Enhanced market condition analysis"""
        try:
            conditions = {
                'timestamp': timestamp,
                'is_valid_time': self.is_valid_trading_time(timestamp),
                'session': self.get_trading_session(timestamp)
            }

            # Enhanced liquidity analysis
            avg_volume = getattr(self, 'avg_volume', volume)
            volume_ratio = volume / avg_volume if avg_volume > 0 else 0
            conditions['liquidity_score'] = min(1.0, volume_ratio)

            # Enhanced volatility analysis
            vol_score = 0.0
            if volatility > 0:
                target_vol = 0.01  # 1% target volatility
                vol_score = min(1.0, target_vol / volatility)
            conditions['volatility_score'] = vol_score

            # Time-based factors
            conditions['time_factor'] = self.get_time_multiplier(timestamp)

            # Market state classification
            conditions['market_state'] = self.classify_market_state(
                volume_ratio=volume_ratio,
                volatility=volatility,
                session=conditions['session']
            )

            return conditions

        except Exception as e:
            self.logger.error(f"Error in market condition analysis: {str(e)}")
            return {}

    def classify_market_state(self, volume_ratio: float, volatility: float,
                              session: str) -> str:
        """Classify current market state"""
        if session == 'regular':
            if volume_ratio > 1.2 and volatility > 0.001:
                return 'high_activity'
            elif volume_ratio < 0.8 or volatility < 0.0005:
                return 'low_activity'
            return 'normal'
        else:  # overnight
            if volume_ratio > 0.8 and volatility > 0.0008:
                return 'active_overnight'
            return 'thin_overnight'

    def is_valid_trading_time(self, timestamp: pd.Timestamp) -> bool:
        try:
            # Ensure we're working with Chicago time
            ct_time = timestamp
            if timestamp.tz is not None:
                ct_time = timestamp.tz_convert('America/Chicago')
            elif timestamp.tz is None:
                ct_time = timestamp.tz_localize('America/Chicago')

            current_time = ct_time.time()

            # Add debug logging to see what times we're actually checking
            self.logger.debug(f"Checking timestamp {timestamp}")
            self.logger.debug(f"Chicago time: {ct_time}")
            self.logger.debug(f"Current time: {current_time}")

            # Regular session check
            regular_start, regular_end = self.trading_hours['regular']
            if regular_start <= current_time <= regular_end:
                self.logger.debug("Valid regular session time")
                return True

            # Overnight session checks
            for overnight_start, overnight_end in self.trading_hours['overnight']:
                if overnight_start > overnight_end:
                    # Session crosses midnight
                    if current_time >= overnight_start or current_time <= overnight_end:
                        self.logger.debug("Valid overnight session time (crossing midnight)")
                        return True
                else:
                    if overnight_start <= current_time <= overnight_end:
                        self.logger.debug("Valid overnight session time")
                        return True

            self.logger.debug("Invalid trading time")
            return False

        except Exception as e:
            self.logger.error(f"Error checking trading time: {str(e)}")
            return False

    def get_confidence_category(self, confidence: float) -> str:
        """Determine confidence category based on thresholds."""
        if confidence >= self.confidence_levels['high']:
            return 'high'
        elif confidence >= self.confidence_levels['medium']:
            return 'medium'
        elif confidence >= self.confidence_levels['low']:
            return 'low'
        return 'none'

    def get_volatility_multiplier(self, current_volatility: float,
                                  historical_volatility: Optional[pd.Series] = None) -> float:
        """Calculate position size multiplier based on volatility."""
        try:
            if historical_volatility is not None and len(historical_volatility) >= 20:
                vol_percentile = (pd.Series(historical_volatility)
                .rolling(20)
                .rank(pct=True)
                .iloc[-1])

                if vol_percentile < 0.25:
                    return 1.2  # Low volatility
                elif vol_percentile > 0.75:
                    return 0.7  # High volatility
                return 1.0  # Normal volatility

            else:
                if current_volatility < 0.0005:
                    return 1.2  # Low volatility
                elif current_volatility > 0.002:
                    return 0.7  # High volatility
                return 1.0  # Normal volatility

        except Exception as e:
            self.logger.error(f"Error calculating volatility multiplier: {str(e)}")
            return 1.0

    def get_time_multiplier(self, timestamp: pd.Timestamp) -> float:
        """Get position size multiplier based on time of day."""
        try:
            ct_time = timestamp.tz_convert('America/Chicago')
            current_time = ct_time.time()

            # Market open period (first 30 mins)
            open_time = self.trading_hours['regular'][0]
            open_end = (datetime.combine(datetime.today(), open_time) +
                        timedelta(minutes=30)).time()
            if open_time <= current_time <= open_end:
                return 0.5

            # Market close period (last 30 mins)
            close_time = self.trading_hours['regular'][1]
            close_start = (datetime.combine(datetime.today(), close_time) -
                           timedelta(minutes=30)).time()
            if close_start <= current_time <= close_time:
                return 0.5

            # Regular session
            if (self.trading_hours['regular'][0] <= current_time <=
                    self.trading_hours['regular'][1]):
                return 1.0

            # Overnight session
            return 0.5

        except Exception as e:
            self.logger.error(f"Error getting time multiplier: {str(e)}")
            return 0.5

    def calculate_slippage(self, price: float, volume: float,
                           volatility: float, is_overnight: bool) -> float:
        """Calculate dynamic slippage based on market conditions."""
        try:
            # Base slippage is 1 tick (0.25 points for MES)
            base_slippage = 0.25  # 1 tick for MES

            # Adjust for volume
            volume_factor = min(1.0, volume / self.liquidity_threshold)
            volume_factor = max(0.2, volume_factor)  # Floor at 0.2

            # Adjust for volatility - more conservative
            vol_factor = 1.0
            if volatility > 0:
                vol_factor = min(2.0, 1 + (volatility * 1000))  # Scale volatility impact

            # Calculate adjusted slippage
            slippage = base_slippage * (1 / volume_factor) * vol_factor

            # Overnight adjustment
            if is_overnight:
                slippage *= 1.25  # 25% increase for overnight

            # Cap slippage at 4 ticks (1.00 point)
            return min(slippage, 1.00)

        except Exception as e:
            self.logger.error(f"Error calculating slippage: {str(e)}")
            return base_slippage

    def get_trading_session(self, timestamp: pd.Timestamp) -> str:
        """Determine current trading session."""
        try:
            ct_time = timestamp.tz_convert('America/Chicago')
            current_time = ct_time.time()

            # Check regular session
            if (self.trading_hours['regular'][0] <= current_time <=
                    self.trading_hours['regular'][1]):
                return 'regular'

            # Check overnight session
            for start, end in self.trading_hours['overnight']:
                if start <= current_time <= end:
                    return 'overnight'
                if start > end and (current_time >= start or current_time <= end):
                    return 'overnight'

            return 'outside'

        except Exception as e:
            self.logger.error(f"Error determining trading session: {str(e)}")
            return 'unknown'

    def get_volatility_regime(self, volatility: float) -> str:
        """Determine volatility regime."""
        if volatility < 0.0005:  # 5 bps
            return 'low'
        elif volatility > 0.002:  # 20 bps
            return 'high'
        return 'normal'

    def update_trade_tracking(self, trade: TradeInfo) -> None:
        """Update tracking variables after a trade."""
        try:
            trade_date = trade['timestamp'].date()

            # Ensure we're not reusing PnL from previous runs
            if trade_date not in self.daily_pnl:
                self.daily_pnl[trade_date] = 0

            if 'pnl' in trade:
                self.daily_pnl[trade_date] += trade['pnl']
                self.current_capital += trade['pnl']  # Update capital

                # Update high water mark and drawdown
                self.high_water_mark = max(self.high_water_mark, self.current_capital)
                self.current_drawdown = (self.high_water_mark - self.current_capital) / self.high_water_mark

        except Exception as e:
            self.logger.error(f"Error updating trade tracking: {str(e)}")

    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate annualized Sharpe ratio."""
        try:
            if len(returns) < 2:
                return 0.0

            # Assuming returns are daily
            rf_rate = 0.04  # 4% annual risk-free rate
            daily_rf = (1 + rf_rate) ** (1 / 252) - 1

            excess_returns = returns - daily_rf
            annualized_sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())

            return float(annualized_sharpe)

        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {str(e)}")
            return 0.0

    def calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio using downside deviation."""
        try:
            if len(returns) < 2:
                return 0.0

            # Assuming returns are daily
            rf_rate = 0.04  # 4% annual risk-free rate
            daily_rf = (1 + rf_rate) ** (1 / 252) - 1

            excess_returns = returns - daily_rf
            downside_returns = excess_returns[excess_returns < 0]

            if len(downside_returns) == 0:
                return float('inf')

            downside_std = np.sqrt(np.mean(downside_returns ** 2))
            sortino_ratio = np.sqrt(252) * (excess_returns.mean() / downside_std)

            return float(sortino_ratio)

        except Exception as e:
            self.logger.error(f"Error calculating Sortino ratio: {str(e)}")
            return 0.0

    def calculate_profit_factor(self) -> float:
        """Calculate profit factor (gross profit / gross loss)."""
        try:
            profits = [t['pnl'] for t in self.trades_history if t.get('pnl', 0) > 0]
            losses = [abs(t['pnl']) for t in self.trades_history if t.get('pnl', 0) < 0]

            gross_profit = sum(profits) if profits else 0
            gross_loss = sum(losses) if losses else 0

            if gross_loss == 0:
                return float('inf') if gross_profit > 0 else 0.0

            return gross_profit / gross_loss

        except Exception as e:
            self.logger.error(f"Error calculating profit factor: {str(e)}")
            return 0.0

    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        try:
            metrics = {
                'total_trades': len(self.trades_history),
                'total_pnl': sum(t.get('pnl', 0) for t in self.trades_history),
                'win_rate': 0.0,
                'average_trade': 0.0,
                'profit_factor': self.calculate_profit_factor(),
                'max_drawdown': self.current_drawdown,
                'risk_metrics': {}
            }

            if metrics['total_trades'] > 0:
                winning_trades = sum(1 for t in self.trades_history if t.get('pnl', 0) > 0)
                metrics['win_rate'] = winning_trades / metrics['total_trades']
                metrics['average_trade'] = metrics['total_pnl'] / metrics['total_trades']

                # Calculate daily returns for risk metrics
                daily_returns = pd.Series(self.daily_pnl.values()) / self.current_capital
                if len(daily_returns) > 1:
                    metrics['risk_metrics'] = {
                        'sharpe_ratio': self.calculate_sharpe_ratio(daily_returns),
                        'sortino_ratio': self.calculate_sortino_ratio(daily_returns),
                        'max_daily_loss': min(self.daily_pnl.values()),
                        'max_daily_profit': max(self.daily_pnl.values())
                    }

            # Add volume analysis
            volume_data = [t.get('contracts', 0) for t in self.trades_history]
            if volume_data:
                metrics['volume_metrics'] = {
                    'average_position_size': np.mean(volume_data),
                    'max_position_size': max(volume_data),
                    'total_contracts_traded': sum(volume_data)
                }

            # Add cost analysis
            slippage_costs = sum(t.get('slippage', 0) for t in self.trades_history)
            commission_costs = sum(t.get('commission', 0) for t in self.trades_history)
            metrics['cost_metrics'] = {
                'total_slippage': slippage_costs,
                'total_commission': commission_costs,
                'total_costs': slippage_costs + commission_costs,
                'cost_per_trade': (slippage_costs + commission_costs) / metrics['total_trades']
                if metrics['total_trades'] > 0 else 0
            }

            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}

    def get_trading_summary(self) -> Dict:
        """Get comprehensive trading summary."""
        try:
            performance_metrics = self.calculate_performance_metrics()

            summary = {
                'capital': {
                    'initial': self.params.initial_capital,
                    'current': self.current_capital,
                    'high_water_mark': self.high_water_mark,
                    'return': (self.current_capital - self.params.initial_capital) /
                              self.params.initial_capital
                },
                'performance': performance_metrics,
                'risk_management': {
                    'current_drawdown': self.current_drawdown,
                    'max_position_size': self.params.max_position_size,
                    'daily_loss_limit': self.max_daily_loss_pct,
                    'max_drawdown_limit': self.max_drawdown_pct
                },
                'trading_activity': {
                    'total_trading_days': len(self.daily_trades),
                    'average_trades_per_day': len(self.trades_history) / len(self.daily_trades)
                    if self.daily_trades else 0,
                    'profitable_days': sum(1 for pnl in self.daily_pnl.values() if pnl > 0),
                    'losing_days': sum(1 for pnl in self.daily_pnl.values() if pnl < 0)
                }
            }

            return summary

        except Exception as e:
            self.logger.error(f"Error getting trading summary: {str(e)}")
            return {}

    def setup_trade_logging(self):
        """Setup dedicated trade logging"""
        trade_logger = logging.getLogger('trade_execution')
        trade_logger.setLevel(logging.INFO)

        # Create trade log file handler
        handler = logging.FileHandler('logs/trade_execution.log')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        trade_logger.addHandler(handler)

        return trade_logger


class ExecutionFailureTracker:
    def __init__(self):
        self.failure_reasons = defaultdict(int)
        self.failure_details = defaultdict(list)
        self.total_attempts = 0

    def log_failure(self, reason: str, details: Dict):
        self.failure_reasons[reason] += 1
        self.failure_details[reason].append(details)
        self.total_attempts += 1

    def get_summary(self):
        return {
            'total_attempts': self.total_attempts,
            'failure_counts': dict(self.failure_reasons),
            'failure_rates': {
                reason: count / self.total_attempts
                for reason, count in self.failure_reasons.items()
            },
            'sample_failures': {
                reason: details[:5]  # First 5 examples of each failure type
                for reason, details in self.failure_details.items()
            }
        }
