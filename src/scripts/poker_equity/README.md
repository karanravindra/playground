# Poker Equity Calculator

This Python script calculates the preflop poker equity for a given hand against a specified number of opponents using a Monte Carlo simulation approach. It evaluates the probability of winning, tying, or losing as well as the frequency of each poker hand rank.

## Features

- **Detailed Probability Breakdown**: Outputs the probabilities of winning, tying, or losing against opponents.
- **Rank Breakdown**: Provides a detailed breakdown of the frequency of each hand rank for both the hero and opponents.
- **Full Poker Hand Evaluation**: Supports all standard poker hands, from high card up to a straight flush.

## Prerequisites

Ensure you have Python installed on your system. This script uses standard Python libraries only, so no additional installations are required.

## Usage

1. **Define Your Hand and Opponents**:
   - Set the `hero_cards` to the two cards in your hand.
   - Set `num_opponents` to the number of opponents you are facing.

2. **Run the Script**:
   - Execute the script to simulate the poker hands and calculate equity. Results will be displayed in the console.
   - You can modify the `iterations` parameter to increase or decrease the number of simulations for accuracy versus performance.

   Example usage in the script:

   ```python
   hero_cards = ["As", "9c"]  # Ace spades and 9 clubs
   num_opponents = 5 # 5 opponents
   results = simulate_preflop_equity(hero_cards, num_opponents)
   print(results)
    ```
