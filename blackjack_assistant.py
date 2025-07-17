#!/usr/bin/env python3
"""
Live Blackjack Strategy Assistant for Stake.us Casino
Main application that provides real-time strategy recommendations
"""

import time
import cv2
import numpy as np
from typing import List, Tuple
import argparse

from blackjack_strategy import BlackjackStrategy
from card_detector import CardDetector
from screen_capture import ScreenCapture


class BlackjackAssistant:
    def __init__(self, debug=False):
        self.debug = debug
        self.strategy = BlackjackStrategy()
        self.detector = CardDetector()
        self.capture = ScreenCapture()
        
        # State tracking
        self.last_player_cards = []
        self.last_dealer_cards = []
        self.last_recommendation = None
        self.recommendation_count = 0
        
        print("🃏 Blackjack Strategy Assistant Initialized")
        print("Strategy tables loaded from blackjack-basic-strategy.md")
    
    def setup_capture_region(self):
        """Setup the screen capture region interactively"""
        print("\n📷 Setting up screen capture region...")
        region = self.capture.select_capture_region_interactive()
        self.capture.set_capture_region(region)
        
        # Test the capture
        print("\n🔍 Testing capture region...")
        self.capture.show_live_preview(3)
        
        return region
    
    def detect_cards_from_current_screen(self) -> Tuple[List[str], List[str]]:
        """Capture screen and detect cards"""
        try:
            # Capture screenshot
            screenshot = self.capture.capture_screenshot()
            
            # Detect cards
            player_cards, dealer_cards = self.detector.detect_cards_from_screenshot(
                screenshot, debug=self.debug
            )
            
            if self.debug:
                # Save debug image
                regions = self.detector.detect_card_regions(screenshot)
                import time
                debug_filename = f"debug_{int(time.time())}.png"
                self.detector.save_debug_image(screenshot, regions, debug_filename)
            
            return player_cards, dealer_cards
            
        except Exception as e:
            print(f"❌ Error detecting cards: {e}")
            return [], []
    
    def has_cards_changed(self, player_cards: List[str], dealer_cards: List[str]) -> bool:
        """
        Check if the detected cards have changed since last check
        More sensitive change detection for immediate response to new cards
        """
        # Check if number of cards changed
        if len(player_cards) != len(self.last_player_cards) or len(dealer_cards) != len(self.last_dealer_cards):
            return True
        
        # Check if card values changed
        if player_cards != self.last_player_cards or dealer_cards != self.last_dealer_cards:
            return True
        
        # If this is the first detection (no previous cards), consider it a change
        if not self.last_player_cards and not self.last_dealer_cards and (player_cards or dealer_cards):
            return True
            
        return False
    
    def get_strategy_recommendation(self, player_cards: List[str], dealer_cards: List[str]) -> Tuple[str, str]:
        """Get strategy recommendation for current cards"""
        if not dealer_cards:
            return "WAIT", "Waiting for dealer card..."
        
        if len(player_cards) < 2:
            return "WAIT", "Waiting for at least 2 player cards..."
        
        dealer_upcard = dealer_cards[0]  # First (and only) dealer card is the upcard
        
        action, explanation = self.strategy.get_strategy_recommendation(player_cards, dealer_upcard)
        return str(action or "ERROR"), str(explanation or "Unknown situation")
    
    def display_recommendation(self, player_cards: List[str], dealer_cards: List[str], 
                             action: str, explanation: str):
        """Display the strategy recommendation"""
        self.recommendation_count += 1
        
        print("\n" + "="*60)
        print(f"🎯 RECOMMENDATION #{self.recommendation_count}")
        print("="*60)
        print(f"👤 Player Cards: {player_cards} (Total: {len(player_cards)} cards)")
        print(f"🏠 Dealer Upcard: {dealer_cards[0] if dealer_cards else 'Unknown'}")
        print(f"💡 RECOMMENDED ACTION: {action}")
        print(f"📝 Explanation: {explanation}")
        print("="*60)
        
        # Store current state
        self.last_player_cards = player_cards.copy()
        self.last_dealer_cards = dealer_cards.copy()
        self.last_recommendation = (action, explanation)
    
    def run_continuous_monitoring(self, check_interval: float = 2.0):
        """
        Main loop for continuous monitoring
        """
        print("\n🚀 Starting continuous monitoring...")
        print(f"⏱️  Checking for cards every {check_interval} seconds")
        print("📝 Will detect: Player cards (all face-up) + Dealer upcard (leftmost only)")
        print("🔄 Re-evaluates on any card change (new deal, hit, split)")
        print("⌨️  Press Ctrl+C to stop")
        print("\n" + "="*60)
        
        consecutive_no_cards = 0
        max_no_cards_warnings = 3
        
        try:
            while True:
                # Detect cards from current screen
                player_cards, dealer_cards = self.detect_cards_from_current_screen()
                
                if self.debug:
                    print(f"Debug: Detected P:{player_cards} D:{dealer_cards}")
                
                # Check if we detected any cards
                if not player_cards and not dealer_cards:
                    consecutive_no_cards += 1
                    if consecutive_no_cards <= max_no_cards_warnings:
                        print(f"⚠️  No cards detected (attempt {consecutive_no_cards}/{max_no_cards_warnings})")
                        if consecutive_no_cards == max_no_cards_warnings:
                            print("🔍 Ensure blackjack game is visible and cards are clearly shown")
                else:
                    consecutive_no_cards = 0  # Reset counter when cards are found
                    
                    # Check if cards have changed
                    if self.has_cards_changed(player_cards, dealer_cards):
                        # Get recommendation
                        action, explanation = self.get_strategy_recommendation(
                            player_cards, dealer_cards
                        )
                        
                        # Display recommendation
                        self.display_recommendation(
                            player_cards, dealer_cards, action, explanation
                        )
                
                # Wait before next check
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Error in monitoring: {e}")
            if self.debug:
                import traceback
                traceback.print_exc()
    
    def run_manual_check(self):
        """Single manual check of current screen"""
        print("\n🔍 Performing manual card detection...")
        print("📝 Looking for: Player cards (all face-up) + Dealer upcard (leftmost only)")
        
        player_cards, dealer_cards = self.detect_cards_from_current_screen()
        
        if not player_cards and not dealer_cards:
            print("❌ No cards detected. Please ensure:")
            print("   1. Stake.us blackjack game is visible")
            print("   2. Cards are clearly visible on screen")
            print("   3. Capture region includes the card areas")
            print("   4. Player has at least 2 cards dealt")
            print("   5. Dealer has at least 1 face-up card")
            return
        
        if not dealer_cards:
            print("⚠️  No dealer upcard detected")
        
        if len(player_cards) < 2:
            print(f"⚠️  Only {len(player_cards)} player card(s) detected - need at least 2")
        
        action, explanation = self.get_strategy_recommendation(player_cards, dealer_cards)
        self.display_recommendation(player_cards, dealer_cards, action, explanation)
    
    def test_detection_on_screenshot(self, image_path: str):
        """Test card detection on a saved screenshot"""
        try:
            screenshot = cv2.imread(image_path)
            if screenshot is None:
                print(f"❌ Could not load image: {image_path}")
                return
            
            print(f"🔍 Testing detection on: {image_path}")
            
            player_cards, dealer_cards = self.detector.detect_cards_from_screenshot(
                screenshot, debug=True
            )
            
            print(f"Player cards: {player_cards}")
            print(f"Dealer cards: {dealer_cards}")
            
            if player_cards or dealer_cards:
                action, explanation = self.get_strategy_recommendation(player_cards, dealer_cards)
                print(f"Recommendation: {action} - {explanation}")
            
        except Exception as e:
            print(f"❌ Error testing on screenshot: {e}")


def main():
    parser = argparse.ArgumentParser(description='Live Blackjack Strategy Assistant')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--interval', type=float, default=2.0, 
                       help='Check interval in seconds (default: 2.0)')
    parser.add_argument('--manual', action='store_true', 
                       help='Run single manual check instead of continuous monitoring')
    parser.add_argument('--test-image', type=str, 
                       help='Test detection on a specific image file')
    parser.add_argument('--setup-only', action='store_true',
                       help='Only setup capture region and exit')
    
    args = parser.parse_args()
    
    # Create assistant
    assistant = BlackjackAssistant(debug=args.debug)
    
    # Test image mode
    if args.test_image:
        assistant.test_detection_on_screenshot(args.test_image)
        return
    
    # Setup capture region
    assistant.setup_capture_region()
    
    if args.setup_only:
        print("✅ Setup complete. Region configured.")
        return
    
    # Run based on mode
    if args.manual:
        assistant.run_manual_check()
    else:
        assistant.run_continuous_monitoring(args.interval)


if __name__ == "__main__":
    main() 