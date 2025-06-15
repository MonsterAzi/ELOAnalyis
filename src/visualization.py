# visualization.py

"""
A module for creating visualizations from simulation results.

This module provides functions to plot key metrics from the simulation history,
helping to analyze the performance and convergence of rating systems. The primary
function plots the correlation between player ratings and their true skills over time.

Dependencies:
    - pandas
    - matplotlib
    - seaborn
    - metrics.py (from the same project)
"""

from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from metrics import calculate_correlation_metrics

# Map internal metric names to more descriptive labels for plotting
METRIC_LABELS = {
    "pearson": "Pearson's r",
    "spearman": "Spearman's ρ",
    "kendall": "Kendall's τ",
    "chatterjee": "Chatterjee's ξ",
}


def plot_correlation_over_time(
    history_df: pd.DataFrame,
    rating_col: str,
    skill_col: str = "true_skill",
    time_col: str = "game_num",
    title: str = "Correlation Between Skill and Rating Over Time",
    palette: Optional[Union[str, List[str]]] = "viridis",
    output_path: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    use_scale: bool = True,
) -> plt.Axes:
    """
    Plots the evolution of correlation metrics between skill and rating over time.
    """
    # 1. Validate inputs (no change)
    required = {time_col, skill_col, rating_col}
    if history_df.empty or not required.issubset(history_df.columns):
        raise ValueError(f"history_df is empty or missing columns: {required - set(history_df.columns)}")

    # 2. Calculate correlations (no change)
    corr_df = (
        history_df.groupby(time_col)
        .apply(
            lambda df: pd.Series(
                calculate_correlation_metrics(df[skill_col], df[rating_col])._asdict()
            ),
            include_groups=False,
        )
        .fillna(0)
    )

    # 3. Setup plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    else:
        fig = ax.get_figure()

    sns.set_theme(style="whitegrid")

    # 4. Melt data and apply better labels
    melted_df = corr_df.reset_index().melt(
        id_vars=[time_col], var_name="Metric", value_name="Correlation"
    )
    melted_df["Metric"] = melted_df["Metric"].map(METRIC_LABELS)

    # 5. Plot the data
    sns.lineplot(data=melted_df, x=time_col, y="Correlation", hue="Metric", marker="o", palette=palette, ax=ax)

    # 6. Finalize plot axes and labels
    ax.set(
        title=title,
        xlabel="Number of Games",
        ylabel="Correlation Coefficient",
    )
    ax.legend(title="Correlation Metric")
    ax.grid(True, which="both", linestyle="--")
    
    gamma = 3
    
    def forward(x):
        """Forward transform function for the power scale."""
        return x ** gamma

    def inverse(x):
        """Inverse transform function for the power scale."""
        return x ** (1. / gamma)

    # --- ADDED: APPLY THE LOGIT SCALE ---
    if use_scale:
        ax.set_yscale('function', functions=(forward, inverse))
        # The default formatter can be ugly, so let's use a cleaner one.
        from matplotlib.ticker import ScalarFormatter
        ax.yaxis.set_major_formatter(ScalarFormatter())
    else:
        # Set a standard linear scale if not using logit
        ax.set_ylim(max(-0.1, melted_df["Correlation"].min() - 0.1), 1.1)

    # 7. Save figure if path is provided (no change)
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return ax

if __name__ == "__main__":
    """
    A simple demonstration of the plotting function when the script is run directly.

    This block generates a mock history DataFrame that simulates the convergence of
    player ratings towards their true skills over time, and then plots the
    resulting correlation metrics.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    def create_mock_history_df(
        num_players: int = 50,
        num_time_steps: int = 20,
        games_per_step: int = 25,
        initial_rating: int = 1200,
    ) -> pd.DataFrame:
        """
        Generates a realistic-looking mock history DataFrame for testing.

        The generated data shows player ratings converging towards their true skill
        over time, with some added noise for realism.

        Returns:
            pd.DataFrame: A DataFrame with columns ['game_num', 'player_id',
                          'true_skill', 'rating'].
        """
        print(f"Generating mock data for {num_players} players over {num_time_steps} time steps...")
        player_ids = [f"Player_{i}" for i in range(num_players)]
        true_skills = np.random.normal(1500, 200, num_players)
        player_skills = dict(zip(player_ids, true_skills))
        history_records = []
        time_steps = [i * games_per_step for i in range(num_time_steps + 1)]

        for game_num in time_steps:
            progress = game_num / time_steps[-1] if len(time_steps) > 1 and time_steps[-1] > 0 else 1
            for player_id, skill in player_skills.items():
                base_rating = initial_rating * (1 - progress) + skill * progress
                noise = np.random.normal(0, 50 * (1 - progress + 0.1))
                rating = base_rating + noise
                history_records.append({
                    "game_num": game_num, "player_id": player_id,
                    "true_skill": skill, "rating": rating,
                })
        return pd.DataFrame(history_records)

    # --- Main execution logic ---
    print("--- Running visualization.py demonstration ---")
    mock_history = create_mock_history_df()
    print("\nSample of generated mock data:")
    print(mock_history.head())
    
    print("\nPlotting correlation over time...")
    try:
        # Create a figure and axes for the plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Call the plotting function
        plot_correlation_over_time(
            history_df=mock_history,
            rating_col="rating",
            skill_col="true_skill",
            time_col="game_num",
            title="Demonstration: Correlation of Mock Ratings to True Skill",
            ax=ax
        )
        
        # FIX: Save the plot to a file instead of showing it
        output_filename = "correlation_demonstration.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        
        print(f"\nPlot saved successfully to '{output_filename}'.")

    except Exception as e:
        print(f"\nAn error occurred during plotting: {e}")
        print("Please ensure matplotlib and seaborn are installed (`pip install matplotlib seaborn`).")

    print("--- Demonstration finished ---")