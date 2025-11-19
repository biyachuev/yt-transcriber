"""
Cost tracking for OpenAI API usage.
Tracks token usage and estimates costs for different operations.
"""

from typing import Dict, Optional
from dataclasses import dataclass, field
from .logger import logger, format_orange


# OpenAI model pricing (as of January 2025, per 1M tokens)
OPENAI_PRICING = {
    "gpt-4": {
        "input": 30.0,  # $30 per 1M input tokens
        "output": 60.0,  # $60 per 1M output tokens
    },
    "gpt-4o": {
        "input": 2.5,  # $2.50 per 1M input tokens
        "output": 10.0,  # $10 per 1M output tokens
    },
    "gpt-4o-mini": {
        "input": 0.15,  # $0.15 per 1M input tokens
        "output": 0.60,  # $0.60 per 1M output tokens
    },
    "gpt-3.5-turbo": {
        "input": 0.50,  # $0.50 per 1M input tokens
        "output": 1.50,  # $1.50 per 1M output tokens
    },
}


@dataclass
class UsageStats:
    """Statistics for a single API call or operation."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: Optional[str] = None  # Track which model was used

    def cost_usd(self, model: Optional[str] = None) -> float:
        """
        Estimate cost in USD based on model-specific pricing.

        Args:
            model: Model name to use for pricing. If None, uses self.model or defaults to gpt-4.

        Returns:
            Estimated cost in USD
        """
        model_to_use = model or self.model or "gpt-4"

        # Get pricing for this model, default to gpt-4 if not found
        pricing = OPENAI_PRICING.get(model_to_use, OPENAI_PRICING["gpt-4"])

        # Prices are per 1M tokens, convert to actual cost
        input_cost = (self.prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (self.completion_tokens / 1_000_000) * pricing["output"]
        return input_cost + output_cost


@dataclass
class CostTracker:
    """
    Track OpenAI API costs across different operations.

    Operations tracked:
    - transcription: Whisper API calls (tracked by duration, not tokens)
    - translation: GPT-4 translation calls
    - refinement: GPT-4 text refinement calls
    - summarization: GPT-4 summarization calls
    """

    transcription: UsageStats = field(default_factory=UsageStats)
    translation: UsageStats = field(default_factory=UsageStats)
    refinement: UsageStats = field(default_factory=UsageStats)
    summarization: UsageStats = field(default_factory=UsageStats)

    # Whisper-specific tracking (duration-based pricing)
    transcription_duration_seconds: float = 0.0

    def add_transcription(self, audio_duration_seconds: float):
        """
        Track Whisper API transcription cost.

        Whisper API pricing: $0.006 per minute
        Note: We don't get token counts from Whisper, just duration.

        Args:
            audio_duration_seconds: Duration of audio in seconds
        """
        self.transcription_duration_seconds += audio_duration_seconds

    @property
    def transcription_cost(self) -> float:
        """
        Calculate Whisper transcription cost.

        Whisper API pricing: $0.006 per minute

        Returns:
            Cost in USD
        """
        duration_minutes = self.transcription_duration_seconds / 60.0
        return duration_minutes * 0.006

    def add_translation(
        self, prompt_tokens: int, completion_tokens: int, model: Optional[str] = None
    ):
        """
        Track translation operation tokens.

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            model: Model name used for this translation (for accurate cost tracking)
        """
        self.translation.prompt_tokens += prompt_tokens
        self.translation.completion_tokens += completion_tokens
        self.translation.total_tokens += prompt_tokens + completion_tokens
        # Store the model if this is the first translation or update if provided
        if model and not self.translation.model:
            self.translation.model = model

    def add_refinement(
        self, prompt_tokens: int, completion_tokens: int, model: Optional[str] = None
    ):
        """
        Track refinement operation tokens.

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            model: Model name used for this refinement (for accurate cost tracking)
        """
        self.refinement.prompt_tokens += prompt_tokens
        self.refinement.completion_tokens += completion_tokens
        self.refinement.total_tokens += prompt_tokens + completion_tokens
        if model:
            self.refinement.model = model

    def add_summarization(
        self, prompt_tokens: int, completion_tokens: int, model: Optional[str] = None
    ):
        """
        Track summarization operation tokens.

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
            model: Model name used for this summarization (for accurate cost tracking)
        """
        self.summarization.prompt_tokens += prompt_tokens
        self.summarization.completion_tokens += completion_tokens
        self.summarization.total_tokens += prompt_tokens + completion_tokens
        if model:
            self.summarization.model = model

    @property
    def total_cost(self) -> float:
        """Calculate total cost across all operations."""
        return (
            self.transcription_cost
            + self.translation.cost_usd()
            + self.refinement.cost_usd()
            + self.summarization.cost_usd()
        )

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens used across all operations."""
        return (
            self.translation.total_tokens
            + self.refinement.total_tokens
            + self.summarization.total_tokens
        )

    def print_summary(self):
        """Print a formatted cost summary."""
        if self.total_tokens == 0 and self.transcription_duration_seconds == 0:
            logger.info("\nNo OpenAI API calls were made")
            return

        logger.info("\n" + "=" * 60)
        logger.info("OpenAI API Cost Summary (estimated)")
        logger.info("=" * 60)

        if self.transcription_duration_seconds > 0:
            duration_minutes = self.transcription_duration_seconds / 60.0
            logger.info("\nTranscription (Whisper API):")
            logger.info(
                f"  Duration:      {duration_minutes:.2f} minutes ({self.transcription_duration_seconds:.1f} seconds)"
            )
            logger.info(
                f"  Cost:          {format_orange(f'${self.transcription_cost:.4f}')} (at $0.006/min)"
            )

        if self.translation.total_tokens > 0:
            model_name = self.translation.model or "gpt-4"
            logger.info(f"\nTranslation ({model_name}):")
            logger.info(f"  Input tokens:  {self.translation.prompt_tokens:,}")
            logger.info(f"  Output tokens: {self.translation.completion_tokens:,}")
            logger.info(f"  Total tokens:  {self.translation.total_tokens:,}")
            logger.info(
                f"  Cost:          {format_orange(f'${self.translation.cost_usd():.4f}')}"
            )

        if self.refinement.total_tokens > 0:
            model_name = self.refinement.model or "gpt-4"
            logger.info(f"\nRefinement ({model_name}):")
            logger.info(f"  Input tokens:  {self.refinement.prompt_tokens:,}")
            logger.info(f"  Output tokens: {self.refinement.completion_tokens:,}")
            logger.info(f"  Total tokens:  {self.refinement.total_tokens:,}")
            logger.info(
                f"  Cost:          {format_orange(f'${self.refinement.cost_usd():.4f}')}"
            )

        if self.summarization.total_tokens > 0:
            model_name = self.summarization.model or "gpt-4"
            logger.info(f"\nSummarization ({model_name}):")
            logger.info(f"  Input tokens:  {self.summarization.prompt_tokens:,}")
            logger.info(f"  Output tokens: {self.summarization.completion_tokens:,}")
            logger.info(f"  Total tokens:  {self.summarization.total_tokens:,}")
            logger.info(
                f"  Cost:          {format_orange(f'${self.summarization.cost_usd():.4f}')}"
            )

        logger.info("\n" + "-" * 60)
        logger.info(f"TOTAL TOKENS: {self.total_tokens:,}")
        logger.info(
            f"TOTAL COST (estimated): {format_orange(f'${self.total_cost:.4f}')} USD"
        )
        logger.info("=" * 60)


# Global cost tracker instance
_cost_tracker: Optional[CostTracker] = None


def get_cost_tracker() -> CostTracker:
    """Get the global cost tracker instance."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker


def reset_cost_tracker():
    """Reset the global cost tracker."""
    global _cost_tracker
    _cost_tracker = CostTracker()
