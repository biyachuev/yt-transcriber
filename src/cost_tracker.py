"""
Cost tracking for OpenAI API usage.
Tracks token usage and estimates costs for different operations.
"""
from typing import Dict, Optional
from dataclasses import dataclass, field
from .logger import logger


@dataclass
class UsageStats:
    """Statistics for a single API call or operation."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        """
        Estimate cost in USD based on GPT-4 pricing.

        GPT-4 pricing (as of 2025):
        - Input: $0.03 per 1K tokens
        - Output: $0.06 per 1K tokens
        """
        input_cost = (self.prompt_tokens / 1000) * 0.03
        output_cost = (self.completion_tokens / 1000) * 0.06
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

    def add_translation(self, prompt_tokens: int, completion_tokens: int):
        """
        Track translation operation tokens.

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
        """
        self.translation.prompt_tokens += prompt_tokens
        self.translation.completion_tokens += completion_tokens
        self.translation.total_tokens += (prompt_tokens + completion_tokens)

    def add_refinement(self, prompt_tokens: int, completion_tokens: int):
        """
        Track refinement operation tokens.

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
        """
        self.refinement.prompt_tokens += prompt_tokens
        self.refinement.completion_tokens += completion_tokens
        self.refinement.total_tokens += (prompt_tokens + completion_tokens)

    def add_summarization(self, prompt_tokens: int, completion_tokens: int):
        """
        Track summarization operation tokens.

        Args:
            prompt_tokens: Number of input tokens
            completion_tokens: Number of output tokens
        """
        self.summarization.prompt_tokens += prompt_tokens
        self.summarization.completion_tokens += completion_tokens
        self.summarization.total_tokens += (prompt_tokens + completion_tokens)

    @property
    def total_cost(self) -> float:
        """Calculate total cost across all operations."""
        return (
            self.transcription_cost +
            self.translation.cost_usd +
            self.refinement.cost_usd +
            self.summarization.cost_usd
        )

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens used across all operations."""
        return (
            self.translation.total_tokens +
            self.refinement.total_tokens +
            self.summarization.total_tokens
        )

    def print_summary(self):
        """Print a formatted cost summary."""
        if self.total_tokens == 0 and self.transcription_duration_seconds == 0:
            logger.info("\nNo OpenAI API calls were made")
            return

        logger.info("\n" + "=" * 60)
        logger.info("OpenAI API Cost Summary")
        logger.info("=" * 60)

        if self.transcription_duration_seconds > 0:
            duration_minutes = self.transcription_duration_seconds / 60.0
            logger.info("\nTranscription (Whisper API):")
            logger.info(f"  Duration:      {duration_minutes:.2f} minutes ({self.transcription_duration_seconds:.1f} seconds)")
            logger.info(f"  Cost:          ${self.transcription_cost:.4f} (at $0.006/min)")

        if self.translation.total_tokens > 0:
            logger.info("\nTranslation:")
            logger.info(f"  Input tokens:  {self.translation.prompt_tokens:,}")
            logger.info(f"  Output tokens: {self.translation.completion_tokens:,}")
            logger.info(f"  Total tokens:  {self.translation.total_tokens:,}")
            logger.info(f"  Cost:          ${self.translation.cost_usd:.4f}")

        if self.refinement.total_tokens > 0:
            logger.info("\nRefinement:")
            logger.info(f"  Input tokens:  {self.refinement.prompt_tokens:,}")
            logger.info(f"  Output tokens: {self.refinement.completion_tokens:,}")
            logger.info(f"  Total tokens:  {self.refinement.total_tokens:,}")
            logger.info(f"  Cost:          ${self.refinement.cost_usd:.4f}")

        if self.summarization.total_tokens > 0:
            logger.info("\nSummarization:")
            logger.info(f"  Input tokens:  {self.summarization.prompt_tokens:,}")
            logger.info(f"  Output tokens: {self.summarization.completion_tokens:,}")
            logger.info(f"  Total tokens:  {self.summarization.total_tokens:,}")
            logger.info(f"  Cost:          ${self.summarization.cost_usd:.4f}")

        logger.info("\n" + "-" * 60)
        logger.info(f"TOTAL TOKENS: {self.total_tokens:,}")
        logger.info(f"ESTIMATED COST: ${self.total_cost:.4f} USD")
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
