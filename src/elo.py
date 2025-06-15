"""
A concise, modular implementation of the Elo rating system using pandas.
"""

from typing import Union, Optional, Any
import pandas as pd

class Elo:
    """Manages player ratings using the Elo rating system."""

    def __init__(self, k_factor: int = 32, initial_rating: int = 1200):
        """
        Initializes the Elo rating system manager.

        Args:
            k_factor (int): The K-factor, determining rating volatility. Defaults to 32.
            initial_rating (int): The rating for new players. Defaults to 1200.
        """
        assert k_factor > 0, "K-factor must be a positive integer."
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings_df = pd.DataFrame(columns=['rating']).rename_axis('player_id')

    def add_player(self, player_id: Any, rating: Optional[int] = None):
        """
        Adds a new player to the rating system.

        Args:
            player_id: The unique identifier for the player.
            rating: The player's starting rating. Defaults to `initial_rating`.

        Raises:
            ValueError: If the player_id already exists.
        """
        if player_id in self.ratings_df.index:
            raise ValueError(f"Player with ID '{player_id}' already exists.")
        
        start_rating = rating if rating is not None else self.initial_rating
        self.ratings_df.loc[player_id] = float(start_rating)

    def _calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculates the probability of player A winning against player B."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def record_match(self, player1_id: Any, player2_id: Any, score_for_player1: float):
        """
        Records a match result and updates ratings for the two players.

        Args:
            player1_id: The ID of the first player.
            player2_id: The ID of the second player.
            score_for_player1 (float): The outcome for player 1. Typically 1.0
                                       for a win, 0.5 for a draw, 0.0 for a loss.

        Raises:
            ValueError: If either player_id does not exist.
        """
        try:
            p1_rating, p2_rating = self.ratings_df.loc[[player1_id, player2_id], 'rating']
        except KeyError as e:
            raise ValueError(f"Player with ID '{e.args[0]}' not found.") from e

        expected_p1_score = self._calculate_expected_score(p1_rating, p2_rating)
        rating_change = self.k_factor * (score_for_player1 - expected_p1_score)

        self.ratings_df.loc[player1_id, 'rating'] += rating_change
        self.ratings_df.loc[player2_id, 'rating'] -= rating_change # Zero-sum change

    def get_rating(self, player_id: Union[str, int]) -> float:
        """Retrieves the current rating for a specific player."""
        try:
            return self.ratings_df.at[player_id, 'rating']
        except KeyError as e:
            raise ValueError(f"Player with ID '{e.args[0]}' not found.") from e

    def get_ratings_df(self) -> pd.DataFrame:
        """Returns a copy of the full player ratings DataFrame."""
        return self.ratings_df.copy()