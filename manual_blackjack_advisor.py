#!/usr/bin/env python3
"""
Manual Blackjack Strategy Advisor
Simple terminal-based advisor where user types in their cards
"""

from blackjack_strategy import BlackjackStrategy


class ManualBlackjackAdvisor:
    def __init__(self):
        self.strategy = BlackjackStrategy()
        self.wins = 0
        self.losses = 0
        self.pushes = 0
        
        print("🃏 Manual Blackjack Strategy Advisor")
        print("=" * 50)
        print("💡 Tips:")
        print("   - Enter cards as: A, 2, 3, 4, 5, 6, 7, 8, 9, 10, J, Q, K")
        print("   - Type 'quit' at any prompt to exit")
        print("   - Press Ctrl+C to exit anytime")
        print()
    
    def get_valid_card(self, prompt: str) -> str:
        """Get a valid card input from user"""
        valid_cards = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        
        while True:
            card = input(prompt).strip().upper()
            
            if card in valid_cards:
                return card
            elif card in ['EXIT', 'QUIT', 'Q']:
                print("👋 Goodbye!")
                exit(0)
            else:
                print(f"❌ Invalid card: '{card}'")
                print(f"   Valid cards: {', '.join(valid_cards)}")
                print("   Or type 'quit' to exit")
    
    def get_player_cards(self) -> list:
        """Get player's initial 2 cards"""
        cards = []
        
        print("📝 Enter your cards:")
        
        # Always get exactly 2 cards for initial deal
        for i in range(2):
            card = self.get_valid_card(f"   Card {i+1}: ")
            cards.append(card)
        
        return cards
    
    def get_hand_result(self) -> str:
        """Ask user for the result of the hand"""
        while True:
            result = input("\n🎲 How did this hand end? (w)in, (l)oss, (p)ush/tie: ").strip().lower()
            
            if result in ['quit', 'q', 'exit']:
                print("👋 Goodbye!")
                exit(0)
            elif result in ['w', 'win']:
                self.wins += 1
                return 'WIN'
            elif result in ['l', 'loss', 'lose']:
                self.losses += 1
                return 'LOSS'
            elif result in ['p', 'push', 'tie']:
                self.pushes += 1
                return 'PUSH'
            else:
                print("   Please enter 'w' for win, 'l' for loss, or 'p' for push/tie")
    
    def display_session_stats(self):
        """Display current session statistics"""
        total_hands = self.wins + self.losses + self.pushes
        
        if total_hands == 0:
            return
            
        win_rate = (self.wins / (self.wins + self.losses) * 100) if (self.wins + self.losses) > 0 else 0
        
        print("\n" + "📊 SESSION STATISTICS")
        print("-" * 30)
        print(f"Hands played: {total_hands}")
        print(f"Wins: {self.wins}")
        print(f"Losses: {self.losses}")
        print(f"Pushes: {self.pushes}")
        print(f"Win rate: {win_rate:.1f}%")
        print("-" * 30)

    def ask_for_more_cards(self, cards: list, last_action: str) -> list:
        """Automatically ask for additional cards based on the recommendation"""
        if last_action == 'HIT':
            print(f"\n   Current cards: {cards}")
            card = self.get_valid_card(f"   Enter your new card: ")
            cards.append(card)
        elif last_action == 'DOUBLE':
            print(f"\n   Current cards: {cards}")
            card = self.get_valid_card(f"   Enter your one double-down card: ")
            cards.append(card)
            
        return cards
    
    def get_dealer_card(self) -> str:
        """Get dealer's upcard"""
        print("🏠 Dealer's face-up card:")
        return self.get_valid_card("   Dealer card: ")
    
    def run(self):
        """Run the advisor for multiple hands"""
        hand_count = 0
        
        while True:
            hand_count += 1
            print(f"\n🎲 HAND #{hand_count}")
            print("-" * 30)
            
            try:
                # Get initial 2 cards and dealer card
                player_cards = self.get_player_cards()
                dealer_card = self.get_dealer_card()
                
                # Loop for multiple recommendations (in case of hitting)
                while True:
                    # Get and display recommendation
                    action, explanation = self.strategy.get_strategy_recommendation(player_cards, dealer_card)
                    
                    # Calculate hand total for display
                    total, is_soft, is_pair = self.strategy.evaluate_hand(player_cards)
                    
                    print("\n" + "=" * 60)
                    print("🎯 STRATEGY RECOMMENDATION")
                    print("=" * 60)
                    print(f"👤 Your cards: {player_cards}")
                    print(f"🏠 Dealer upcard: {dealer_card}")
                    print(f"💡 RECOMMENDED ACTION: {action}")
                    print("=" * 60)
                    
                    # Follow the recommendation automatically
                    if action == 'HIT':
                        # Ask for new card and continue for another recommendation
                        player_cards = self.ask_for_more_cards(player_cards, action)
                        continue  # Get new recommendation with updated hand
                    elif action == 'DOUBLE':
                        # Ask for one card and end hand (double down ends the hand)
                        player_cards = self.ask_for_more_cards(player_cards, action)
                        print(f"\n   Final hand: {player_cards}")
                        break  # Hand is complete after double down
                    else:
                        # Action was STAND, SPLIT, etc. - hand is complete
                        break
                
                # Get hand result and update statistics
                hand_result = self.get_hand_result()
                print(f"✅ Hand result: {hand_result}")
                
                # Display session statistics
                self.display_session_stats()
                
                # Automatically continue to next hand (no prompt needed)
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    advisor = ManualBlackjackAdvisor()
    advisor.run()


if __name__ == "__main__":
    main() 