"""
A modular and concise simulation engine for running game-based tournaments.
This module provides a Simulation class to orchestrate games, update ratings,
and track player history, designed for flexible integration with various strategies.
"""

import logging
import random
from typing import Any, Callable, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# Configure a logger for the module
logger = logging.getLogger(__name__)


class Simulation:
    """Orchestrates a tournament simulation over a number of games."""

    def __init__(
        self,
        players_df: pd.DataFrame,
        rating_system: Any,
        pairing_strategy: Callable[[pd.DataFrame], List[Tuple[Any, Any]]],
        skill_update_strategy: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
        log_every_n_games: int = 25,
    ):
        """
        Initializes the Simulation.

        Args:
            players_df (pd.DataFrame): DataFrame with player data, must include
                'player_id' and 'true_skill' columns.
            rating_system (Any): An object with a `record_match` method.
            pairing_strategy (Callable): Function to generate game pairings.
            skill_update_strategy (Optional[Callable]): Optional function to update
                player skills over time. Defaults to None.
            log_every_n_games (int): Frequency for logging history. Defaults to 25.
        """
        if "player_id" not in players_df.columns or "true_skill" not in players_df.columns:
            raise ValueError("players_df must contain 'player_id' and 'true_skill' columns.")

        self.players_df = players_df.copy().set_index("player_id")
        self.rating_system = rating_system
        self.pairing_strategy = pairing_strategy
        self.skill_update_strategy = skill_update_strategy
        self.log_every_n_games = log_every_n_games
        self.history: List[pd.DataFrame] = []
        self.history_df: Optional[pd.DataFrame] = None
        
        # ADD: Automatically add players to the rating system
        if hasattr(self.rating_system, 'add_player'):
            initial_ratings = self.rating_system.get_ratings_df()
            for player_id in self.players_df.index:
                if player_id not in initial_ratings.index:
                    self.rating_system.add_player(player_id)

        logger.info("Simulation initialized with %d players.", len(self.players_df))

    def _simulate_outcome(self, p1_id: Any, p2_id: Any) -> float:
        """Simulates a game outcome based on players' true skill."""
        skill1, skill2 = self.players_df.loc[[p1_id, p2_id], "true_skill"]
        win_prob_p1 = 1 / (1 + 10 ** ((skill2 - skill1) / 400))
        return 1.0 if random.random() < win_prob_p1 else 0.0

    def _log_history(self, game_num: int):
        """Saves a snapshot of the current player data and ratings."""
        current_ratings = self.rating_system.get_ratings_df()
        log_df = self.players_df.join(current_ratings)
        # Reset index to make 'player_id' a column again for the history log
        log_df = log_df.reset_index() 
        self.history.append(log_df.assign(game_num=game_num))

    def run(self, num_games: int) -> pd.DataFrame:
        """
        Runs the full simulation for a specified number of games.

        Args:
            num_games (int): The total number of games to simulate.

        Returns:
            pd.DataFrame: The final state of the players DataFrame. The full
                          history is available in the `self.history_df` attribute.
        """
        logger.info("Starting simulation for %d games.", num_games)
        self._log_history(game_num=0)  # Log initial state

        for game_num in tqdm(range(1, num_games + 1), desc="Simulating Games"):
            # 1. Pair players and play games
            # Note: The pairing strategy call is addressed in Issue #2
            pairings, _ = self.pairing_strategy(self.players_df) # Unpack the tuple
            for p1_id, p2_id in pairings:
                outcome = self._simulate_outcome(p1_id, p2_id)
                self.rating_system.record_match(p1_id, p2_id, outcome)

            # 2. Update player skills if a strategy is provided
            if self.skill_update_strategy:
                # This is more robust against strategies that might change the index order
                self.players_df = self.skill_update_strategy(self.players_df)
            
            # 3. Log history at specified intervals
            if game_num % self.log_every_n_games == 0:
                self._log_history(game_num)

        # FIX: Log the final state if it wasn't already logged
        if num_games % self.log_every_n_games != 0:
            self._log_history(num_games)

        # Consolidate history and return final state
        if self.history:
            self.history_df = pd.concat(self.history, ignore_index=True)
            logger.info("Simulation finished. History DataFrame created.")
        
        # Return the final state with updated ratings
        final_df = self.players_df.join(self.rating_system.get_ratings_df())
        return final_df.reset_index(drop=True)