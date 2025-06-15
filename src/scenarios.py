"""
scenarios.py

A modular library for defining and executing different player scenarios.

This module provides functions for:
- Pairing players randomly.
- Pairing players based on groups ("islands") with a chance for crossover.
- Updating player skills with a random drift.
"""

import random
from typing import List, Dict, Tuple, Optional, Any
import collections

import numpy as np
import pandas as pd

# --- Type Aliases ---
PlayerID = Any
GroupID = Any
Pairing = Tuple[PlayerID, PlayerID]
Pairings = List[Pairing]
PlayerGroups = Dict[PlayerID, GroupID]


def random_pairing(
    players_df: pd.DataFrame
) -> Tuple[Pairings, Optional[PlayerID]]:
    """
    Generates random pairings from a DataFrame of players.

    Args:
        players_df: A DataFrame where the index contains the player identifiers.
        
    Returns:
        A tuple containing a list of pairs and an optional unpaired player.
    """
    player_ids = players_df.index.tolist()
    shuffled = random.sample(player_ids, k=len(player_ids))
    pairings = [(shuffled[i], shuffled[i + 1]) for i in range(0, len(shuffled) - 1, 2)]
    unpaired = shuffled[-1] if len(player_ids) % 2 != 0 else None
    return pairings, unpaired


def island_pairing(
    players_df: pd.DataFrame,
    player_groups: PlayerGroups,
    crossover_chance: float = 0.1,
) -> Tuple[Pairings, Optional[PlayerID]]:
    """
    Generates pairings with a preference for within-group ("island") matches.

    Args:
        players_df: A DataFrame where the index contains the player identifiers.
        player_groups: A dictionary mapping each player ID to their group ID.
        crossover_chance: The probability of attempting a crossover match.

    Returns:
        A tuple containing a list of pairs and an optional unpaired player.
    """
    if not (0.0 <= crossover_chance <= 1.0):
        raise ValueError("crossover_chance must be between 0.0 and 1.0")

    player_ids = players_df.index.tolist()
    unpaired = set(player_ids)
    pairings: Pairings = []
    
    while len(unpaired) >= 2:
        player1 = random.choice(list(unpaired))
        p1_group = player_groups[player1]

        same_group_partners = [p for p in unpaired if p != player1 and player_groups[p] == p1_group]
        diff_group_partners = [p for p in unpaired if player_groups[p] != p1_group]

        # Prioritize crossover or same-group based on chance
        use_crossover = random.random() < crossover_chance
        primary_list = diff_group_partners if use_crossover else same_group_partners
        secondary_list = same_group_partners if use_crossover else diff_group_partners

        # Select a partner from the first non-empty list
        partner_list = primary_list or secondary_list
        if not partner_list:
            break  # No more pairs are possible

        player2 = random.choice(partner_list)
        pairings.append((player1, player2))
        unpaired.remove(player1)
        unpaired.remove(player2)

    return pairings, (list(unpaired)[0] if unpaired else None)


def drifting_update(
    player_df: pd.DataFrame, 
    drift_amt: float,
    skill_col: str = "skill"
) -> pd.DataFrame:
    """
    Updates player skills by a random "drift" amount in a new DataFrame.

    Args:
        player_df: DataFrame containing player data with a skill column.
        drift_amt: The maximum absolute value of the skill drift.
        skill_col: The name of the skill column.

    Returns:
        A new pandas DataFrame with updated skill values.
    """
    if skill_col not in player_df.columns:
        raise KeyError(f"Column '{skill_col}' not found in the DataFrame.")
    
    drifts = np.random.uniform(-drift_amt, drift_amt, size=len(player_df))
    return player_df.assign(**{skill_col: player_df[skill_col] + drifts})