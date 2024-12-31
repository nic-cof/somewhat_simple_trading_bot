# pipeline_state.py
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import logging
import json
from pathlib import Path


@dataclass
class MetricsContainer:
    timestamp: datetime
    values: Dict
    metadata: Dict = field(default_factory=dict)


class PipelineState:
    """Centralized state and logging management"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Setup centralized logging
        self.logger = self._setup_logger()

        # State tracking
        self.metrics: Dict[str, List[MetricsContainer]] = {}
        self.stage: str = "initialization"
        self.errors: List[Dict] = []
        self.warnings: List[Dict] = []

        # Performance tracking
        self.start_time = datetime.now()
        self.stage_timings: Dict[str, Dict] = {}

    def _setup_logger(self) -> logging.Logger:
        """Setup single centralized logger"""
        logger = logging.getLogger('pipeline')
        logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(self.log_dir / 'pipeline.log')
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(fh)

        # Console handler with simpler format
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(message)s'))
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

        return logger

    def log_metrics(self, stage: str, metrics: Dict, metadata: Optional[Dict] = None):
        """Log metrics with timestamp and optional metadata"""
        if stage not in self.metrics:
            self.metrics[stage] = []

        container = MetricsContainer(
            timestamp=datetime.now(),
            values=metrics,
            metadata=metadata or {}
        )

        self.metrics[stage].append(container)
        self.logger.info(f"{stage} metrics: {metrics}")

    def log_error(self, error: Exception, context: str):
        """Log error with context"""
        self.errors.append({
            'timestamp': datetime.now().isoformat(),
            'stage': self.stage,
            'context': context,
            'error': str(error),
            'type': type(error).__name__
        })
        self.logger.error(f"Error in {context}: {str(error)}")

    def log_warning(self, message: str, context: str):
        """Log warning with context"""
        self.warnings.append({
            'timestamp': datetime.now().isoformat(),
            'stage': self.stage,
            'context': context,
            'message': message
        })
        self.logger.warning(f"Warning in {context}: {message}")

    def log_progress(self, current: int, total: int, stage: str):
        """Log progress with optional progress bar"""
        if current % (total // 100) == 0:
            progress = (current / total) * 100
            bar_length = 20
            filled_length = int(bar_length * current // total)
            bar = '=' * filled_length + '-' * (bar_length - filled_length)
            self.logger.info(f'\r{stage}: [{bar}] {progress:.1f}%')

    def start_stage(self, stage: str):
        """Mark start of processing stage"""
        self.stage = stage
        self.stage_timings[stage] = {
            'start': datetime.now()
        }
        self.logger.info(f"\nStarting {stage}")

    def complete_stage(self, stage: str, metrics: Optional[Dict] = None):
        """Mark completion of processing stage"""
        end_time = datetime.now()
        duration = (end_time - self.stage_timings[stage]['start']).total_seconds()

        self.stage_timings[stage]['end'] = end_time
        self.stage_timings[stage]['duration'] = duration

        if metrics:
            self.log_metrics(stage, metrics)

        self.logger.info(f"Completed {stage} in {duration:.2f}s")

    def save_state(self):
        """Save complete state to file"""
        state = {
            'metrics': {
                stage: [
                    {
                        'timestamp': m.timestamp.isoformat(),
                        'values': m.values,
                        'metadata': m.metadata
                    } for m in metrics
                ] for stage, metrics in self.metrics.items()
            },
            'errors': self.errors,
            'warnings': self.warnings,
            'timings': {
                stage: {
                    k: v.isoformat() if isinstance(v, datetime) else v
                    for k, v in timing.items()
                } for stage, timing in self.stage_timings.items()
            }
        }

        output_file = self.log_dir / f'pipeline_state_{datetime.now():%Y%m%d_%H%M%S}.json'
        with open(output_file, 'w') as f:
            json.dump(state, f, indent=2)

        self.logger.info(f"Saved pipeline state to {output_file}")