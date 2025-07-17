"""
Blackjack Basic Strategy Implementation
Based on blackjack-basic-strategy.md
"""

class BlackjackStrategy:
    def __init__(self):
        # Pair splitting table
        self.pair_splitting = {
            ('A', 'A'): ['Y'] * 10,  # Always split Aces
            ('10', '10'): ['N'] * 10,  # Never split 10s
            ('9', '9'): ['Y', 'Y', 'Y', 'Y', 'Y', 'N', 'Y', 'Y', 'N', 'N'],
            ('8', '8'): ['Y'] * 10,  # Always split 8s
            ('7', '7'): ['Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N', 'N'],
            ('6', '6'): ['N', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N', 'N', 'N'],
            ('5', '5'): ['N'] * 10,  # Never split 5s
            ('4', '4'): ['N'] * 10,  # Never split 4s
            ('3', '3'): ['N', 'N', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N', 'N'],
            ('2', '2'): ['N', 'N', 'Y', 'Y', 'Y', 'Y', 'N', 'N', 'N', 'N'],
        }
        
        # Soft totals (Ace counts as 11)
        self.soft_totals = {
            20: ['S'] * 10,  # A,9 - Always stand
            19: ['S', 'S', 'S', 'S', 'Ds', 'S', 'S', 'S', 'S', 'S'],  # A,8
            18: ['Ds', 'Ds', 'Ds', 'Ds', 'Ds', 'S', 'S', 'H', 'H', 'H'],  # A,7
            17: ['D', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # A,6
            16: ['H', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # A,5
            15: ['H', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # A,4
            14: ['H', 'H', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # A,3
            13: ['H', 'H', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],  # A,2
        }
        
        # Hard totals (no usable Ace)
        self.hard_totals = {
            21: ['S'] * 10,  # Blackjack
            20: ['S'] * 10,
            19: ['S'] * 10,
            18: ['S'] * 10,
            17: ['S'] * 10,
            16: ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],
            15: ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],
            14: ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],
            13: ['S', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],
            12: ['H', 'S', 'S', 'S', 'S', 'H', 'H', 'H', 'H', 'H'],
            11: ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'H'],
            10: ['D', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'H', 'H'],
            9: ['H', 'D', 'D', 'D', 'D', 'H', 'H', 'H', 'H', 'H'],
            8: ['H'] * 10,
            7: ['H'] * 10,
            6: ['H'] * 10,
            5: ['H'] * 10,
        }
        
        # Dealer upcard mapping (2-A maps to indices 0-9)
        self.dealer_index_map = {
            '2': 0, '3': 1, '4': 2, '5': 3, '6': 4,
            '7': 5, '8': 6, '9': 7, '10': 8, 'J': 8, 'Q': 8, 'K': 8, 'A': 9
        }
    
    def normalize_card(self, card):
        """Normalize card representation (J, Q, K -> 10)"""
        if card in ['J', 'Q', 'K']:
            return '10'
        return card
    
    def get_card_value(self, card):
        """Get numeric value of a card"""
        if card in ['J', 'Q', 'K']:
            return 10
        elif card == 'A':
            return 11  # We'll handle Ace as 1 vs 11 in hand evaluation
        else:
            return int(card)
    
    def evaluate_hand(self, cards):
        """
        Evaluate hand and return (total, is_soft, is_pair)
        is_soft = True if hand contains an Ace counted as 11
        is_pair = True if exactly 2 cards of same rank
        """
        # Check for pair first (only with exactly 2 cards)
        is_pair = False
        if len(cards) == 2:
            normalized_cards = [self.normalize_card(card) for card in cards]
            is_pair = normalized_cards[0] == normalized_cards[1]
        
        # Calculate hand value (always calculate, even for pairs)
        total = 0
        aces = 0
        
        for card in cards:
            if card == 'A':
                aces += 1
                total += 11
            else:
                total += self.get_card_value(card)
        
        # Handle Aces
        is_soft = False
        while total > 21 and aces > 0:
            total -= 10  # Convert Ace from 11 to 1
            aces -= 1
        
        # Check if we have a "soft" hand (Ace counted as 11)
        if aces > 0:
            is_soft = True
        
        return total, is_soft, is_pair
    
    def get_strategy_recommendation(self, player_cards, dealer_upcard):
        """
        Get the optimal strategy recommendation
        Returns: (action, explanation)
        """
        if len(player_cards) < 2:
            return "WAIT", "Need at least 2 cards to make a decision"
        
        dealer_index = self.dealer_index_map.get(dealer_upcard)
        if dealer_index is None:
            return "ERROR", f"Invalid dealer upcard: {dealer_upcard}"
        
        total, is_soft, is_pair = self.evaluate_hand(player_cards)
        
        # Handle pairs first (only for exactly 2 cards)
        if is_pair and len(player_cards) == 2:
            normalized_card = self.normalize_card(player_cards[0])
            pair_key = (normalized_card, normalized_card)
            if pair_key in self.pair_splitting:
                action = self.pair_splitting[pair_key][dealer_index]
                if action == 'Y':
                    return "SPLIT", f"Split {normalized_card}s vs dealer {dealer_upcard}"
                else:
                    # If we don't split, re-evaluate as regular hand
                    total, is_soft, _ = self.evaluate_hand(player_cards)
        
        # Handle soft totals
        if is_soft and total and total in self.soft_totals:
            action = self.soft_totals[total][dealer_index]
            return self._format_action(action, f"Soft {total}", dealer_upcard)
        
        # Handle hard totals
        if total and total in self.hard_totals:
            action = self.hard_totals[total][dealer_index]
            return self._format_action(action, f"Hard {total}", dealer_upcard)
        
        # Default for very low totals
        if total and total < 5:
            return "HIT", f"Always hit with {total}"
        
        return "ERROR", f"Unhandled situation: {player_cards} vs {dealer_upcard}"
    
    def _format_action(self, action, hand_desc, dealer_upcard):
        """Format action with explanation"""
        action_map = {
            'S': 'STAND',
            'H': 'HIT', 
            'D': 'DOUBLE',
            'Ds': 'DOUBLE_OR_STAND'
        }
        
        formatted_action = action_map.get(action, action)
        
        explanations = {
            'STAND': f"Stand with {hand_desc} vs dealer {dealer_upcard}",
            'HIT': f"Hit {hand_desc} vs dealer {dealer_upcard}",
            'DOUBLE': f"Double {hand_desc} vs dealer {dealer_upcard}",
            'DOUBLE_OR_STAND': f"Double if possible, otherwise stand with {hand_desc} vs dealer {dealer_upcard}"
        }
        
        explanation = explanations.get(formatted_action, f"{formatted_action} {hand_desc} vs dealer {dealer_upcard}")
        return formatted_action, explanation


# Quick test function
if __name__ == "__main__":
    strategy = BlackjackStrategy()
    
    # Test cases
    test_cases = [
        (['A', 'A'], '10'),  # Split Aces
        (['8', '8'], '6'),   # Split 8s
        (['10', '6'], '10'), # Hard 16 vs 10
        (['A', '7'], '9'),   # Soft 18 vs 9
        (['5', '6'], '5'),   # Hard 11 vs 5
    ]
    
    for player_cards, dealer_card in test_cases:
        action, explanation = strategy.get_strategy_recommendation(player_cards, dealer_card)
        print(f"Cards: {player_cards} vs {dealer_card} -> {action}: {explanation}") 