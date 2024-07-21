import random

# Constants for card and suit rankings
SUITS = "cdhs"  # Clubs, Diamonds, Hearts, Spades
RANKS = "23456789TJQKA"
CARD_RANK = {rank: i for i, rank in enumerate(RANKS)}


def create_deck():
    return [rank + suit for suit in SUITS for rank in RANKS]


def draw_hand(deck, num_cards):
    hand = random.sample(deck, num_cards)
    for card in hand:
        deck.remove(card)
    return hand


def hand_rank(hand):
    """Returns the rank of the hand: 0 - High Card, 1 - Pair, ..., 8 - Straight Flush"""
    ranks = [CARD_RANK[card[0]] for card in hand]
    suits = [card[1] for card in hand]
    is_flush = len(set(suits)) == 1
    is_straight = max(ranks) - min(ranks) == 4 and len(set(ranks)) == 5
    rank_counts = {rank: ranks.count(rank) for rank in ranks}
    count_values = list(rank_counts.values())
    if is_straight and is_flush:
        return 8  # Straight Flush
    elif 4 in count_values:
        return 7  # Four of a Kind
    elif sorted(count_values) == [2, 3]:
        return 6  # Full House
    elif is_flush:
        return 5  # Flush
    elif is_straight:
        return 4  # Straight
    elif 3 in count_values:
        return 3  # Three of a Kind
    elif count_values.count(2) == 2:
        return 2  # Two Pair
    elif 2 in count_values:
        return 1  # Pair
    return 0  # High Card


def simulate_preflop_equity(hero_hand, num_opponents, iterations=10000):
    wins, ties, losses = 0, 0, 0
    rank_count = {rank: 0 for rank in range(9)}
    opponent_rank_count = {rank: 0 for rank in range(9)}

    deck = create_deck()
    deck = [card for card in deck if card not in hero_hand]
    hero_rank = hand_rank(hero_hand)
    rank_count[hero_rank] += 1

    for _ in range(iterations):
        deck_copy = deck[:]
        opponent_hands = [draw_hand(deck_copy, 2) for _ in range(num_opponents)]
        opponent_ranks = [hand_rank(hand) for hand in opponent_hands]

        for rank in opponent_ranks:
            opponent_rank_count[rank] += 1

        best_opponent_rank = max(opponent_ranks)
        if hero_rank > best_opponent_rank:
            wins += 1
        elif hero_rank == best_opponent_rank:
            ties += 1
        else:
            losses += 1

    total = wins + ties + losses
    result = f"Results\nProbability breakdown:\nYou win: {wins/total:.1%}\nYou tie: {ties/total:.1%}\nYou lose: {losses/total:.1%}\n"
    result += "Rank breakdown:\nRank\tYou\tOpponents\n"
    hand_names = [
        "High Card",
        "Pair",
        "Two Pair",
        "Three of a Kind",
        "Straight",
        "Flush",
        "Full House",
        "Four of a Kind",
        "Straight Flush",
    ]
    for rank, name in enumerate(hand_names):
        result += f"{name}\t{rank_count[rank]/iterations:.1%}\t{opponent_rank_count[rank]/(iterations*num_opponents):.1%}\n"
    return result


# Example usage
hero_cards = ["As", "Ks"]  # Ace and King of spades
num_opponents = 1
results = simulate_preflop_equity(hero_cards, num_opponents)
print(results)
