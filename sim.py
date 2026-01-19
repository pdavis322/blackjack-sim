import random
import math
from typing import List, Optional, Tuple

RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
VALUES = {
    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
    'J': 10, 'Q': 10, 'K': 10, 'A': 11
}

# Configuration
NUM_DECKS = 6
PENETRATION = 0.75  
HIT_SOFT_17 = False  
BASE_BET = 10
WONG_OUT_THRESHOLD = -2

class Card:
    def __init__(self, rank: str):
        self.rank = rank
        self.value = VALUES[rank]

    def __repr__(self):
        return f"{self.rank}"

class Shoe:
    def __init__(self, num_decks: int):
        self.num_decks = num_decks
        self.cards = []
        self.running_count = 0
        self.reshuffle()

    def reshuffle(self):
        self.cards = []
        for _ in range(self.num_decks):
            for _ in range(4): 
                for rank in RANKS:
                    self.cards.append(Card(rank))
        random.shuffle(self.cards)
        self.starting_count = len(self.cards)
        self.running_count = 0

    def deal(self) -> Card:
        if not self.cards:
            self.reshuffle()
        card = self.cards.pop()
        
        # Update running count (Hi-Lo strategy)
        # 2-6: +1, 7-9: 0, 10-A: -1
        if card.rank in ['2', '3', '4', '5', '6']:
            self.running_count += 1
        elif card.rank in ['10', 'J', 'Q', 'K', 'A']:
            self.running_count -= 1
            
        return card

    def get_true_count(self) -> float:
        # rounding decks up for hi-lo
        decks_remaining = math.ceil(len(self.cards) / 52.0)
        
        # Avoid division by zero if shoe is empty (unlikely with penetration logic)
        if decks_remaining < 1:
            decks_remaining = 1.0
            
        return self.running_count / decks_remaining

    def needs_shuffle(self, penetration: float) -> bool:
        cards_dealt = self.starting_count - len(self.cards)
        return cards_dealt / self.starting_count >= penetration

class Hand:
    def __init__(self, bet: float, from_split: bool = False):
        self.cards: List[Card] = []
        self.bet = bet
        self.is_split_hand = from_split
        self.surrendered = False
        self.doubled = False
        self.stand = False

    def add_card(self, card: Card):
        self.cards.append(card)

    @property
    def value(self) -> int:
        val = sum(c.value for c in self.cards)
        aces = sum(1 for c in self.cards if c.rank == 'A')
        while val > 21 and aces > 0:
            val -= 10
            aces -= 1
        return val

    @property
    def is_soft(self) -> bool:
        val = sum(c.value for c in self.cards)
        aces = sum(1 for c in self.cards if c.rank == 'A')
        
        # Reduce aces from 11 to 1 until we're under 22
        while val > 21 and aces > 0:
            val -= 10
            aces -= 1
        
        # Hand is soft if we still have at least one ace counting as 11
        return aces > 0 and val <= 21        

    @property
    def is_blackjack(self) -> bool:
        return len(self.cards) == 2 and self.value == 21 and not self.is_split_hand

    @property
    def is_pair(self) -> bool:
        return len(self.cards) == 2 and self.cards[0].value == self.cards[1].value

    def __repr__(self):
        return f"[{','.join(str(c) for c in self.cards)}]({self.value})"

def get_basic_strategy_move(player_hand: Hand, dealer_up_card: Card, hit_soft_17: bool, can_split: bool = True, can_double: bool = True) -> str:
    # Moves: H=Hit, S=Stand, D=Double, P=Split, R=Surrender
    
    dealer_val = dealer_up_card.value
    
    # 1. Surrender logic (only first 2 cards)
    if len(player_hand.cards) == 2 and not player_hand.is_split_hand:

        is_pair_8s = (player_hand.is_pair and player_hand.cards[0].rank == '8')

        # H17 Surrender Tables
        if hit_soft_17:
            # 88 is the only pair that can surrender
            if is_pair_8s and can_split:
                if dealer_val == 11: return 'R'
            else:
                # Standard Surrender - Hard Totals Only
                if not player_hand.is_soft:
                    if player_hand.value == 17 and dealer_val == 11: return 'R'
                    if player_hand.value == 16 and dealer_val in [9, 10, 11]: return 'R'
                    if player_hand.value == 15 and dealer_val in [10, 11]: return 'R'

        else: 
            # S17 Surrender
            # 8s Special Case: Do not surrender 8s vs Ace (Split instead)
            if is_pair_8s and can_split:
                pass
            else:
                if player_hand.value == 16 and dealer_val in [9, 10, 11]: return 'R'
                if player_hand.value == 15 and dealer_val == 10: return 'R'
            
    # 2. Splits
    if player_hand.is_pair and can_split:
        rank = player_hand.cards[0].rank
        if rank == 'A': return 'P'
        if rank == '8': return 'P'
        if rank in ['2', '3']:
            # P if 2-7, else H. DAS allowed.
            if dealer_val <= 7: return 'P'
            return 'H'
        if rank == '4':
            # P only if 5 or 6 (assuming DAS)
            if dealer_val in [5, 6]: return 'P'
            return 'H'
        if rank == '5':
            # Never split 5s. Treat as 10.
            pass 
        if rank == '6':
            # P if 2-6
            if dealer_val <= 6: return 'P'
            return 'H'
        if rank == '7':
            # P if 2-7
            if dealer_val <= 7: return 'P'
            return 'H'
        if rank == '9':
            # P if 2-6, 8, 9. (Stand on 7, 10, A)
            if dealer_val <= 6 or dealer_val in [8, 9]: return 'P'
            return 'S' # Stand on 7, 10, A (18)
        # 10s never split outside of deviations
    
    # 3. Soft Totals
    if player_hand.is_soft:
        # We have an Ace counted as 11
        # Subtract 11 to get the "other" part.
        other_val = player_hand.value - 11
        
        if other_val >= 9: return 'S' # A,9 = 20 -> Stand
        if other_val == 8: # A,8 = 19
            if dealer_val == 6 and hit_soft_17:
                if can_double: return 'D'
                return 'S' # Fallback: Stand
            return 'S'
            
        if other_val == 7: # A,7 = 18
            # DS vs 2-6, S vs 7-8, H vs 9-A
            if dealer_val <= 6:
                if can_double: return 'D'
                return 'S' # Fallback: Stand
            if dealer_val <= 8: return 'S'
            return 'H'
            
        # A,6 (17) -> D vs 3-6
        if other_val == 6:
            if 3 <= dealer_val <= 6: return 'D' 
            return 'H'
            
        # A,4 and A,5 (15, 16) -> D vs 4-6
        if other_val in [4, 5]:
            if 4 <= dealer_val <= 6: return 'D'
            return 'H'
            
        # A,2 and A,3 (13, 14) -> D vs 5-6
        if other_val in [2, 3]:
            if 5 <= dealer_val <= 6: return 'D'
            return 'H'
            
    # 4. Hard Totals
    val = player_hand.value
    if val >= 17: return 'S'
    if val >= 13:
        # 13-16 stand vs 2-6, hit vs 7-A
        if dealer_val <= 6: return 'S'
        return 'H'
    if val == 12:
        # 12 stand vs 4-6, hit otherwise
        if 4 <= dealer_val <= 6: return 'S'
        return 'H'
    if val == 11: return 'H' if (not HIT_SOFT_17 and dealer_val == 11) else 'D'
    if val == 10:
        if dealer_val <= 9: return 'D'
        return 'H'
    if val == 9:
        if 3 <= dealer_val <= 6: return 'D'
        return 'H'
    
    return 'H'

def simulate_shoe(args):
    """
    Simulate one shoe until penetration is reached.
    Args is a tuple: (shoe_id, verbose)
    Returns: (shoe_id, total_profit, hands_played)
    """
    shoe_id, verbose = args
    shoe = Shoe(NUM_DECKS)
    total_profit = 0
    hands_played = 0
    
    while not shoe.needs_shuffle(PENETRATION):
        if shoe.get_true_count() < WONG_OUT_THRESHOLD:
             if verbose:
                 print(f"*** WONGING OUT at TC {shoe.get_true_count():.2f} ***")
             break

        profit = play_round(shoe, hands_played, verbose=verbose)
        total_profit += profit
        hands_played += 1
    
    return (shoe_id, total_profit, hands_played)

def play_round(shoe: Shoe, round_num: int, verbose: bool = True):
    """Play a single round. Set verbose=False to suppress output."""
    # Betting Logic based on True Count
    true_count = shoe.get_true_count()
    # 1 to 9 spread
    multiplier = max(1, min(9, math.floor(true_count) + 1))
    bet = BASE_BET * multiplier
    
    player_hand = Hand(bet)
    dealer_hand = Hand(0) 
    
    player_hand.add_card(shoe.deal())
    dealer_hand.add_card(shoe.deal())
    player_hand.add_card(shoe.deal())
    dealer_hand.add_card(shoe.deal())
    
    dealer_up = dealer_hand.cards[0]
    
    active_hands = [player_hand]
    
    dealer_has_bj = dealer_hand.is_blackjack
    
    if dealer_has_bj:
        if player_hand.is_blackjack:
            profit = 0
            result = "Push (BJ vs BJ)"
        else:
            profit = -player_hand.bet
            result = "Loss (Dealer BJ)"
        if verbose:
            print(f"Bet: {player_hand.bet} | Dealer: {dealer_hand} | Player: {player_hand} | Result: {result} | Profit: {profit}")
        return profit

    i = 0
    while i < len(active_hands):
        hand = active_hands[i]
        
        if hand.is_blackjack:
            hand.stand = True
            i += 1
            continue
            
        while not hand.stand and hand.value <= 21:
            if hand.is_split_hand and hand.cards[0].rank == 'A':
                if len(hand.cards) >= 2:
                   hand.stand = True
                   break
            
            can_split = len(active_hands) < 4
            can_double = len(hand.cards) == 2
            action = get_basic_strategy_move(hand, dealer_up, HIT_SOFT_17, can_split=can_split, can_double=can_double)
            
            if action == 'P':
                card1 = hand.cards[0]
                card2 = hand.cards[1]
                
                hand1 = Hand(hand.bet, from_split=True)
                hand1.add_card(card1)
                hand1.add_card(shoe.deal())
                
                hand2 = Hand(hand.bet, from_split=True)
                hand2.add_card(card2)
                hand2.add_card(shoe.deal())
                
                active_hands[i] = hand1
                active_hands.append(hand2)
                break
                
            elif action == 'D':
                 if len(hand.cards) == 2:
                     hand.bet *= 2
                     hand.add_card(shoe.deal())
                     hand.doubled = True
                     hand.stand = True
                 else:
                     hand.add_card(shoe.deal())
                     
            elif action == 'R':
                hand.surrendered = True
                hand.stand = True
                
            elif action == 'S':
                hand.stand = True
                
            elif action == 'H':
                hand.add_card(shoe.deal())
                if hand.value > 21:
                    hand.stand = True
        
        if active_hands[i].stand or active_hands[i].value > 21 or active_hands[i].surrendered:
            i += 1

    # Dealer Turn
    all_done = True
    for h in active_hands:
        if not h.surrendered and h.value <= 21:
            all_done = False
            break
            
    if not all_done:
        while dealer_hand.value < 17 or (HIT_SOFT_17 and dealer_hand.value == 17 and dealer_hand.is_soft):
            dealer_hand.add_card(shoe.deal())

    # Settlement
    total_profit = 0
    results_str = []
    
    for h in active_hands:
        p_profit = 0
        p_result = ""
        
        if h.surrendered:
            p_profit = -h.bet * 0.5
            p_result = "Surrender"
        elif h.value > 21:
            p_profit = -h.bet
            p_result = "Bust"
        elif h.is_blackjack:
            p_profit = h.bet * 1.5
            p_result = "Blackjack"
        else:
            if dealer_hand.value > 21:
                p_profit = h.bet
                p_result = "Win (Dealer Bust)"
            elif h.value > dealer_hand.value:
                p_profit = h.bet
                p_result = "Win"
            elif h.value < dealer_hand.value:
                p_profit = -h.bet
                p_result = "Loss"
            else:
                p_profit = 0
                p_result = "Push"
                
        total_profit += p_profit
        results_str.append(f"Hand: {h} ({p_result})")
    
    total_wager = sum(h.bet for h in active_hands)
    p_hands_str = " | ".join(results_str)
    
    if verbose:
        print(f"Bet: {total_wager} | Dealer: {dealer_hand} | Player: {p_hands_str} | Profit: {total_profit}")
    return total_profit

def main():
    """Single-threaded simulation (original behavior)."""
    shoe = Shoe(NUM_DECKS)
    total_cumulative_profit = 0
    
    for i in range(1, 21):
        if shoe.get_true_count() < WONG_OUT_THRESHOLD:
            print(f"*** Wonging Out! (TC: {shoe.get_true_count():.2f}) - Finding new table... ***")
            shoe.reshuffle()
            continue

        profit = play_round(shoe, i, verbose=True)
        total_cumulative_profit += profit
        
        if shoe.needs_shuffle(PENETRATION):
            shoe.reshuffle()
            print("--- SHUFFLE ---")
    
    print("-" * 40)
    print(f"Total Cumulative Profit: {total_cumulative_profit}")

def init_worker(config):
    global HIT_SOFT_17, NUM_DECKS, PENETRATION, BASE_BET, WONG_OUT_THRESHOLD
    HIT_SOFT_17 = config['h17']
    NUM_DECKS = config['decks']
    PENETRATION = config['pen']
    BASE_BET = config['min_bet']
    WONG_OUT_THRESHOLD = config['wong_out']

def main_parallel(num_shoes: int = 100, num_workers: int = None, config: dict = None):
    """
    Parallel simulation across multiple shoes.
     Each shoe is played until penetration is reached.
    
    Args:
        num_shoes: Number of shoes to simulate
        num_workers: Number of CPU cores to use (default: all available)
        config: Configuration dictionary
    """
    from multiprocessing import Pool, cpu_count
    import time
    
    if num_workers is None:
        num_workers = cpu_count()
    
    print(f"Starting parallel simulation:")
    print(f"  - Shoes: {num_shoes:,}")
    print(f"  - Config: {config}")
    print(f"  - Workers: {num_workers}")
    print("-" * 40)
    
    args = [(i, False) for i in range(num_shoes)]
    
    start_time = time.time()
    
    with Pool(num_workers, initializer=init_worker, initargs=(config,)) as pool:
        results = pool.map(simulate_shoe, args)
    
    elapsed = time.time() - start_time
    
    total_profit = sum(r[1] for r in results)
    total_hands = sum(r[2] for r in results)
    
    print(f"Simulation Complete!")
    print(f"  - Time: {elapsed:.2f} seconds")
    print(f"  - Hands/second: {total_hands / elapsed:,.0f}")
    print("-" * 40)
    print(f"Total Shoes: {num_shoes:,}")
    print(f"Total Hands Played: {total_hands:,}")
    print(f"Avg Hands/Shoe: {total_hands / num_shoes:.1f}")
    print(f"Total Profit: {total_profit:,.2f}")
    print(f"Average Profit/Hand: {total_profit / total_hands:.4f}")
    print(f"House Edge: {-total_profit / (total_hands * BASE_BET) * 100:.2f}%")

if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Blackjack Simulation")
    parser.add_argument("--parallel", nargs="?", const=1000, type=int, help="Run in parallel mode with optional num_shoes")
    parser.add_argument("--h17", action="store_true", help="Hit Soft 17 (flag)")
    parser.add_argument("--decks", type=int, default=6, help="Number of decks")
    parser.add_argument("--pen", type=float, default=0.75, help="Penetration (0.0-1.0)")
    parser.add_argument("--min_bet", type=int, default=10, help="Minimum bet")
    parser.add_argument("--wong_out", type=float, default=-2, help="Wong out threshold")
    
    args = parser.parse_args()

    # Update Globals
    HIT_SOFT_17 = args.h17
    NUM_DECKS = args.decks
    PENETRATION = args.pen
    BASE_BET = args.min_bet
    WONG_OUT_THRESHOLD = args.wong_out
    
    config = {
        'h17': HIT_SOFT_17,
        'decks': NUM_DECKS,
        'pen': PENETRATION,
        'min_bet': BASE_BET,
        'wong_out': WONG_OUT_THRESHOLD
    }
    
    if args.parallel:
        main_parallel(args.parallel, config=config)
    else:
        print(f"Running single-threaded with config: {config}")
        main()
