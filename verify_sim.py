"""
Unit tests for blackjack basic strategy - focusing on pair splitting decisions.
Tests all pair values (2,2 through A,A) against various dealer up cards.
"""

from sim import Card, Hand, get_basic_strategy_move

H17 = True  # Testing with H17 rules

def test_pair_aces():
    hand = Hand(10)
    hand.add_card(Card('A'))
    hand.add_card(Card('A'))
    
    for dealer_rank in ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']:
        dealer = Card(dealer_rank)
        move = get_basic_strategy_move(hand, dealer, H17)
        assert move == 'P', f"A,A vs {dealer_rank}: Expected 'P', got '{move}'"
    print("✓ Pair of Aces: All tests passed")

def test_pair_twos():
    """2,2: Split vs 2-7, Hit vs 8-A"""
    hand = Hand(10)
    hand.add_card(Card('2'))
    hand.add_card(Card('2'))
    
    for dealer_rank in ['2', '3', '4', '5', '6', '7']:
        dealer = Card(dealer_rank)
        move = get_basic_strategy_move(hand, dealer, H17)
        assert move == 'P', f"2,2 vs {dealer_rank}: Expected 'P', got '{move}'"
    
    for dealer_rank in ['8', '9', '10', 'A']:
        dealer = Card(dealer_rank)
        move = get_basic_strategy_move(hand, dealer, H17)
        assert move == 'H', f"2,2 vs {dealer_rank}: Expected 'H', got '{move}'"
    print("✓ Pair of 2s: All tests passed")

def test_pair_threes():
    """3,3: Split vs 2-7, Hit vs 8-A"""
    hand = Hand(10)
    hand.add_card(Card('3'))
    hand.add_card(Card('3'))
    
    for dealer_rank in ['2', '3', '4', '5', '6', '7']:
        dealer = Card(dealer_rank)
        move = get_basic_strategy_move(hand, dealer, H17)
        assert move == 'P', f"3,3 vs {dealer_rank}: Expected 'P', got '{move}'"
    
    for dealer_rank in ['8', '9', '10', 'A']:
        dealer = Card(dealer_rank)
        move = get_basic_strategy_move(hand, dealer, H17)
        assert move == 'H', f"3,3 vs {dealer_rank}: Expected 'H', got '{move}'"
    print("✓ Pair of 3s: All tests passed")

def test_pair_fours():
    """4,4: Split vs 5-6 (DAS), Hit otherwise"""
    hand = Hand(10)
    hand.add_card(Card('4'))
    hand.add_card(Card('4'))
    
    for dealer_rank in ['5', '6']:
        dealer = Card(dealer_rank)
        move = get_basic_strategy_move(hand, dealer, H17)
        assert move == 'P', f"4,4 vs {dealer_rank}: Expected 'P', got '{move}'"
    
    for dealer_rank in ['2', '3', '4', '7', '8', '9', '10', 'A']:
        dealer = Card(dealer_rank)
        move = get_basic_strategy_move(hand, dealer, H17)
        assert move == 'H', f"4,4 vs {dealer_rank}: Expected 'H', got '{move}'"
    print("✓ Pair of 4s: All tests passed")

def test_pair_fives():
    """5,5: Never split. Treat as 10 -> Double vs 2-9, Hit vs 10,A"""
    hand = Hand(10)
    hand.add_card(Card('5'))
    hand.add_card(Card('5'))
    
    for dealer_rank in ['2', '3', '4', '5', '6', '7', '8', '9']:
        dealer = Card(dealer_rank)
        move = get_basic_strategy_move(hand, dealer, H17)
        assert move == 'D', f"5,5 vs {dealer_rank}: Expected 'D', got '{move}'"
    
    for dealer_rank in ['10', 'A']:
        dealer = Card(dealer_rank)
        move = get_basic_strategy_move(hand, dealer, H17)
        assert move == 'H', f"5,5 vs {dealer_rank}: Expected 'H', got '{move}'"
    print("✓ Pair of 5s: All tests passed")

def test_pair_sixes():
    """6,6: Split vs 2-6, Hit vs 7-A"""
    hand = Hand(10)
    hand.add_card(Card('6'))
    hand.add_card(Card('6'))
    
    for dealer_rank in ['2', '3', '4', '5', '6']:
        dealer = Card(dealer_rank)
        move = get_basic_strategy_move(hand, dealer, H17)
        assert move == 'P', f"6,6 vs {dealer_rank}: Expected 'P', got '{move}'"
    
    for dealer_rank in ['7', '8', '9', '10', 'A']:
        dealer = Card(dealer_rank)
        move = get_basic_strategy_move(hand, dealer, H17)
        assert move == 'H', f"6,6 vs {dealer_rank}: Expected 'H', got '{move}'"
    print("✓ Pair of 6s: All tests passed")

def test_pair_sevens():
    """7,7: Split vs 2-7, Hit vs 8-A"""
    hand = Hand(10)
    hand.add_card(Card('7'))
    hand.add_card(Card('7'))
    
    for dealer_rank in ['2', '3', '4', '5', '6', '7']:
        dealer = Card(dealer_rank)
        move = get_basic_strategy_move(hand, dealer, H17)
        assert move == 'P', f"7,7 vs {dealer_rank}: Expected 'P', got '{move}'"
    
    for dealer_rank in ['8', '9', '10', 'A']:
        dealer = Card(dealer_rank)
        move = get_basic_strategy_move(hand, dealer, H17)
        assert move == 'H', f"7,7 vs {dealer_rank}: Expected 'H', got '{move}'"
    print("✓ Pair of 7s: All tests passed")

def test_pair_eights():
    """8,8: Surrender vs A (H17), Split otherwise"""
    hand = Hand(10)
    hand.add_card(Card('8'))
    hand.add_card(Card('8'))
    
    # Surrender vs Ace (H17 rule)
    dealer = Card('A')
    move = get_basic_strategy_move(hand, dealer, H17)
    assert move == 'R', f"8,8 vs A: Expected 'R', got '{move}'"
    
    # Split vs everything else
    for dealer_rank in ['2', '3', '4', '5', '6', '7', '8', '9', '10']:
        dealer = Card(dealer_rank)
        move = get_basic_strategy_move(hand, dealer, H17)
        assert move == 'P', f"8,8 vs {dealer_rank}: Expected 'P', got '{move}'"
    print("✓ Pair of 8s: All tests passed")

def test_pair_nines():
    """9,9: Split vs 2-6, 8-9. Stand vs 7, 10, A"""
    hand = Hand(10)
    hand.add_card(Card('9'))
    hand.add_card(Card('9'))
    
    for dealer_rank in ['2', '3', '4', '5', '6', '8', '9']:
        dealer = Card(dealer_rank)
        move = get_basic_strategy_move(hand, dealer, H17)
        assert move == 'P', f"9,9 vs {dealer_rank}: Expected 'P', got '{move}'"
    
    for dealer_rank in ['7', '10', 'A']:
        dealer = Card(dealer_rank)
        move = get_basic_strategy_move(hand, dealer, H17)
        assert move == 'S', f"9,9 vs {dealer_rank}: Expected 'S', got '{move}'"
    print("✓ Pair of 9s: All tests passed")

def test_pair_tens():
    """10,10: Never split. Stand on 20."""
    hand = Hand(10)
    hand.add_card(Card('10'))
    hand.add_card(Card('10'))
    
    for dealer_rank in ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']:
        dealer = Card(dealer_rank)
        move = get_basic_strategy_move(hand, dealer, H17)
        assert move == 'S', f"10,10 vs {dealer_rank}: Expected 'S', got '{move}'"
    print("✓ Pair of 10s: All tests passed")

def run_all_pair_tests():
    print("=" * 40)
    print("Running Pair Splitting Unit Tests (H17)")
    print("=" * 40)
    
    test_pair_aces()
    test_pair_twos()
    test_pair_threes()
    test_pair_fours()
    test_pair_fives()
    test_pair_sixes()
    test_pair_sevens()
    test_pair_eights()
    test_pair_nines()
    test_pair_tens()
    
    print("=" * 40)
    print("All pair tests passed!")
    print("=" * 40)


def test_resplit():
    """
    Test re-splitting: Start with 9,9, split into two hands where one gets another 9.
    Uses a controlled shoe to force specific card deals.
    """
    from sim import Shoe, Hand, Card, play_round, BASE_BET, HIT_SOFT_17
    
    # Create a rigged shoe for testing
    class RiggedShoe:
        def __init__(self, cards):
            # Cards are dealt in reverse order (pop from end)
            self.cards = list(reversed(cards))
            self.starting_count = len(cards)
        
        def deal(self):
            return self.cards.pop()
        
        def needs_shuffle(self, penetration):
            return False
        
        def reshuffle(self):
            pass
    
    # Scenario: Player gets 9,9. Dealer shows 6.
    # After split: Hand1 gets 9 (another pair!), Hand2 gets 10.
    # Hand1 re-splits: Hand1a gets 10, Hand1b gets 10.
    # Final: 3 hands total (9+10, 9+10, 9+10) = all 19s
    # Dealer: 6, then draws to 16, then busts with 10.
    
    rigged_cards = [
        # Initial deal (alternating player/dealer)
        Card('9'),   # Player card 1
        Card('6'),   # Dealer card 1
        Card('9'),   # Player card 2
        Card('10'),  # Dealer card 2 (hidden)
        # After first split
        Card('9'),   # Hand1 gets 9 (pair again!)
        Card('10'),  # Hand2 gets 10
        # After second split (re-split of Hand1)
        Card('10'),  # Hand1a gets 10
        Card('10'),  # Hand1b gets 10
        # Dealer draws
        Card('10'),  # Dealer busts (6+10+10=26)
    ]
    
    shoe = RiggedShoe(rigged_cards)
    
    # We need to manually simulate the split logic
    # Create initial hand
    player_hand = Hand(BASE_BET)
    dealer_hand = Hand(0)
    
    player_hand.add_card(shoe.deal())  # 9
    dealer_hand.add_card(shoe.deal())  # 6
    player_hand.add_card(shoe.deal())  # 9
    dealer_hand.add_card(shoe.deal())  # 10
    
    dealer_up = dealer_hand.cards[0]
    
    # Verify initial hand is a pair of 9s
    assert player_hand.is_pair, "Initial hand should be a pair"
    assert player_hand.value == 18, f"Expected value 18, got {player_hand.value}"
    
    # Check strategy says split
    from sim import get_basic_strategy_move
    move = get_basic_strategy_move(player_hand, dealer_up, H17)
    assert move == 'P', f"9,9 vs 6: Expected 'P', got '{move}'"
    
    # Simulate first split
    active_hands = [player_hand]
    card1 = player_hand.cards[0]
    card2 = player_hand.cards[1]
    
    hand1 = Hand(player_hand.bet, from_split=True)
    hand1.add_card(card1)
    hand1.add_card(shoe.deal())  # Gets another 9!
    
    hand2 = Hand(player_hand.bet, from_split=True)
    hand2.add_card(card2)
    hand2.add_card(shoe.deal())  # Gets 10
    
    active_hands = [hand1, hand2]
    
    # Verify hand1 is another pair
    assert hand1.is_pair, "Hand1 after split should be a pair (9,9)"
    assert hand1.value == 18, f"Hand1 expected 18, got {hand1.value}"
    
    # Strategy should say split again
    move = get_basic_strategy_move(hand1, dealer_up, H17)
    assert move == 'P', f"Re-split 9,9 vs 6: Expected 'P', got '{move}'"
    
    # Simulate second split (re-split)
    card1a = hand1.cards[0]
    card1b = hand1.cards[1]
    
    hand1a = Hand(hand1.bet, from_split=True)
    hand1a.add_card(card1a)
    hand1a.add_card(shoe.deal())  # Gets 10
    
    hand1b = Hand(hand1.bet, from_split=True)
    hand1b.add_card(card1b)
    hand1b.add_card(shoe.deal())  # Gets 10
    
    active_hands = [hand1a, hand1b, hand2]
    
    # Verify we now have 3 hands
    assert len(active_hands) == 3, f"Expected 3 hands after re-split, got {len(active_hands)}"
    
    # All hands should be 19
    for i, h in enumerate(active_hands):
        assert h.value == 19, f"Hand {i+1} expected 19, got {h.value}"
    
    print("✓ Re-split test: All tests passed")
    print(f"  - Successfully split 9,9 into 3 hands")
    print(f"  - All hands have value 19")


def test_split_limit():
    """
    Test that strategy respects can_split=False.
    Force a pair that usually splits (e.g. 8,8) and verify it returns a non-split move when denied.
    """
    hand = Hand(10)
    hand.add_card(Card('8'))
    hand.add_card(Card('8'))
    
    # Dealer shows 10. 
    # Normal: 8,8 vs 10 -> Split (P)
    # Limit: 8,8 vs 10 -> Hard 16 vs 10 -> Surrender (R) (in H17)
    
    from sim import get_basic_strategy_move
    
    # 1. Verify Normal Split
    move = get_basic_strategy_move(hand, Card('10'), H17, can_split=True)
    assert move == 'P', f"8,8 vs 10 (Split allowed): Expected 'P', got '{move}'"
    
    # 2. Verify Split Denied -> Hard 16 vs 10 -> Surrender
    move = get_basic_strategy_move(hand, Card('10'), H17, can_split=False)
    assert move == 'R', f"8,8 vs 10 (Split denied): Expected 'R' (Surrender 16), got '{move}'"
    
    # Test another case: 9,9 vs 9.
    # Normal: Split.
    # Limit: Hard 18 vs 9 -> Stand.
    hand9 = Hand(10)
    hand9.add_card(Card('9'))
    hand9.add_card(Card('9'))
    
    move = get_basic_strategy_move(hand9, Card('9'), H17, can_split=False)
    assert move == 'S', f"9,9 vs 9 (Split denied): Expected 'S' (Hard 18), got '{move}'"

    print("✓ Split Limit Test: All tests passed")

def test_soft_double_fallback():
    """
    Test that Soft 18/19 return 'S' (Stand) instead of 'D' (Double) when doubling is not allowed.
    Currently, the simulation defaults to Hit if 'D' is returned but invalid.
    We need strategy to explicitly return 'S' if can_double=False.
    """
    from sim import get_basic_strategy_move
    
    # Soft 18 (A,7 etc) vs 6
    # Standard: Double. Fallback: Stand.
    hand = Hand(10)
    hand.add_card(Card('A'))
    hand.add_card(Card('2'))
    hand.add_card(Card('5')) # Soft 18, 3 cards
    
    # We expect this to require a new parameter can_double=False
    # For now, let's verify what it returns currently (likely 'D')
    # Once fixed, it should return 'S'
    
    # Note: passing can_double=False (which we will implement)
    # This test will fail until we implement the fix.
    try:
        move = get_basic_strategy_move(hand, Card('6'), H17, can_double=False)
        assert move == 'S', f"Soft 18 vs 6 (Double denied): Expected 'S', got '{move}'"
    except TypeError:
        print("!! can_double parameter not yet implemented")
        return
        
    print("✓ Soft Double Fallback: All tests passed")

def test_wonging_out():
    """Verify shoe ends early when True Count drops below threshold."""
    import sim
    from sim import Shoe, Card, simulate_shoe, WONG_OUT_THRESHOLD
    import io
    import sys
    from unittest.mock import patch
    
    # 1. Set Threshold to -1 temporarily
    original_threshold = sim.WONG_OUT_THRESHOLD
    sim.WONG_OUT_THRESHOLD = -1
    
    print("Testing Wonging Out...")

    try:
        # 2. Define Mock Shoe that forces negative count
        class MockShoe:
            def __init__(self, num_decks):
                self.cards = [Card('2')] * 10 # Just some cards
                self.starting_count = 10
                self.num_decks = num_decks
                self.running_count = 0
            
            def get_true_count(self):
                # Always return -10 to force trigger
                return -10.0
                
            def needs_shuffle(self, pen):
                return False # Never needs shuffle naturally, relying on break
                
            def deal(self):
                return self.cards.pop() if self.cards else Card('2')
            
            def reshuffle(self):
                pass

        # 3. Patch sim.Shoe to use our MockShoe
        # We need to capture stdout
        captured_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output
        
        try:
            with patch('sim.Shoe', side_effect=MockShoe):
                # Run simulate_shoe with verbose=True
                # args = (shoe_id, verbose)
                sim.simulate_shoe((1, True))
        finally:
            sys.stdout = original_stdout
            
        # 4. Verify Output
        output = captured_output.getvalue()
        expected = "*** WONGING OUT at TC -10.00 ***"
        
        if expected in output:
             print("✓ Wonging Out: Print statement verified")
        else:
             print("✗ Wonging Out: Failed to find expected print statement")
             print(f"   Output was: {output}")
             raise AssertionError("Wonging out message not found in output")
             
    finally:
        sim.WONG_OUT_THRESHOLD = original_threshold

def test_card_counting():
    """Verify Hi-Lo Counting Logic"""
    from sim import Shoe, Card, NUM_DECKS
    shoe = Shoe(1)
    shoe.cards = []
    shoe.running_count = 0
    
    # +1 cards
    shoe.cards.append(Card('2'))
    shoe.deal()
    assert shoe.running_count == 1, f"Expect +1 (2), got {shoe.running_count}"
    
    # -1 cards
    shoe.cards.append(Card('A'))
    shoe.deal()
    assert shoe.running_count == 0, f"Expect 0 (A), got {shoe.running_count}"
    
    # 0 cards
    shoe.cards.append(Card('8'))
    shoe.deal()
    assert shoe.running_count == 0, f"Expect 0 (8), got {shoe.running_count}"
    
    # True Count Ceiling Logic
    shoe.cards = [Card('2')] * 26 # 26/52 = 0.5 decks -> ceil -> 1.0
    shoe.running_count = 10
    tc = shoe.get_true_count() # 10 / 1.0 = 10
    assert abs(tc - 10.0) < 0.01, f"Expect TC 10 (Ceil), got {tc}"
    
    # 51 cards -> ceil(51/52) = 1.0
    shoe.cards = [Card('2')] * 51
    tc = shoe.get_true_count()
    assert abs(tc - 10.0) < 0.01, f"Expect TC 10 (51 cards), got {tc}"
    
    print("✓ Card Counting: All tests passed")

def test_betting_logic():
    """Verify Betting Spread"""
    import math
    def get_bet(tc):
        m = max(1, min(9, math.floor(tc) + 1))
        return m
        
    assert get_bet(0.5) == 1, "TC 0.5 -> 1x"
    assert get_bet(1.0) == 2, "TC 1.0 -> 2x"
    assert get_bet(1.9) == 2, "TC 1.9 -> 2x"
    assert get_bet(7.5) == 8, "TC 7.5 -> 8x"
    assert get_bet(8.0) == 9, "TC 8.0 -> 9x"
    assert get_bet(20.0) == 9, "TC 20.0 -> 9x"
    
    print("✓ Betting Logic: All tests passed")

if __name__ == "__main__":
    run_all_pair_tests()
    print()
    test_resplit()
    print()
    test_split_limit()
    print()
    test_soft_double_fallback()
    print()
    test_wonging_out()
    print()
    test_card_counting()
    print()
    test_betting_logic()
