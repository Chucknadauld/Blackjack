#!/usr/bin/env python3
"""
Template Builder Utility for Blackjack Card Detection
Helps users manually create templates from debug images
"""

import cv2
import numpy as np
import os
from typing import List, Tuple
import argparse
from card_detector import CardDetector


class TemplateBuilder:
    def __init__(self):
        self.detector = CardDetector()
        self.current_image = None
        self.regions = []
        
    def load_debug_image(self, image_path: str) -> bool:
        """Load a debug image for template creation"""
        try:
            self.current_image = cv2.imread(image_path)
            if self.current_image is None:
                print(f"❌ Could not load image: {image_path}")
                return False
                
            print(f"✅ Loaded image: {image_path}")
            
            # Detect card regions automatically
            self.regions = self.detector.detect_card_regions(self.current_image, debug=True)
            print(f"Found {len(self.regions)} potential card regions")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return False
    
    def show_regions(self):
        """Display the image with detected regions marked"""
        if self.current_image is None:
            print("❌ No image loaded")
            return
            
        display_img = self.current_image.copy()
        
        for i, (x, y, w, h) in enumerate(self.regions):
            # Draw rectangle around detected card
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Add label
            cv2.putText(display_img, f"Region {i}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Resize for display if too large
        height, width = display_img.shape[:2]
        if height > 800 or width > 1200:
            scale = min(800/height, 1200/width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_img = cv2.resize(display_img, (new_width, new_height))
        
        cv2.imshow('Card Regions - Press any key to continue', display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def extract_region(self, region_index: int) -> np.ndarray:
        """Extract a specific region for template creation"""
        if region_index >= len(self.regions):
            print(f"❌ Invalid region index: {region_index}")
            return None
            
        x, y, w, h = self.regions[region_index]
        card_region = self.current_image[y:y+h, x:x+w]
        return card_region
    
    def show_region_detail(self, region_index: int):
        """Show a detailed view of a specific region"""
        card_region = self.extract_region(region_index)
        if card_region is None:
            return
            
        # Show the full card
        cv2.imshow(f'Region {region_index} - Full Card', card_region)
        
        # Show the template area (top-left 40%)
        height, width = card_region.shape[:2]
        template_area = card_region[:int(height * 0.4), :int(width * 0.4)]
        cv2.imshow(f'Region {region_index} - Template Area', template_area)
        
        print(f"Region {region_index}: {self.regions[region_index]}")
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def create_template(self, region_index: int, card_value: str) -> bool:
        """Create a template from a specific region"""
        card_region = self.extract_region(region_index)
        if card_region is None:
            return False
            
        success = self.detector.create_template_for_card(card_region, card_value)
        if success:
            print(f"✅ Created template for '{card_value}' from region {region_index}")
        else:
            print(f"❌ Failed to create template for '{card_value}'")
            
        return success
    
    def interactive_template_creation(self):
        """Interactive mode for creating templates"""
        if self.current_image is None:
            print("❌ No image loaded")
            return
            
        print("\n🎯 Interactive Template Creation")
        print("Available commands:")
        print("  'show' - Show all detected regions")
        print("  'detail <region>' - Show detailed view of a region")
        print("  'create <region> <card_value>' - Create template from region")
        print("  'stats' - Show current template statistics")
        print("  'quit' - Exit interactive mode")
        print()
        
        while True:
            try:
                command = input("Template Builder> ").strip().lower()
                
                if command == 'quit':
                    break
                elif command == 'show':
                    self.show_regions()
                elif command == 'stats':
                    stats = self.detector.get_template_stats()
                    print(f"Templates: {stats['template_count']}")
                    print(f"Available cards: {stats['available_cards']}")
                elif command.startswith('detail '):
                    try:
                        region_idx = int(command.split()[1])
                        self.show_region_detail(region_idx)
                    except (IndexError, ValueError):
                        print("❌ Usage: detail <region_number>")
                elif command.startswith('create '):
                    try:
                        parts = command.split()
                        region_idx = int(parts[1])
                        card_value = parts[2].upper()
                        self.create_template(region_idx, card_value)
                    except (IndexError, ValueError):
                        print("❌ Usage: create <region_number> <card_value>")
                        print("   Example: create 0 K")
                else:
                    print("❌ Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break


def main():
    parser = argparse.ArgumentParser(description='Template Builder for Card Detection')
    parser.add_argument('image_path', help='Path to debug image for template creation')
    parser.add_argument('--batch', action='store_true', 
                       help='Batch mode with predefined card values')
    
    args = parser.parse_args()
    
    builder = TemplateBuilder()
    
    # Load the debug image
    if not builder.load_debug_image(args.image_path):
        return
    
    # Show current template stats
    stats = builder.detector.get_template_stats()
    print(f"\n📊 Current Template Library:")
    print(f"   Templates: {stats['template_count']}")
    print(f"   Available cards: {stats['available_cards']}")
    
    if args.batch:
        # Batch mode - try to create templates for common cards
        print("\n🔄 Batch Mode: Attempting to create templates...")
        
        # Define expected cards (you can modify this based on your game)
        expected_cards = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        
        builder.show_regions()
        
        print("\nBatch template creation - you'll need to identify each region:")
        for i, region in enumerate(builder.regions):
            print(f"\nRegion {i}:")
            builder.show_region_detail(i)
            
            card_value = input(f"What card is in region {i}? (or 'skip'): ").strip().upper()
            if card_value != 'SKIP' and card_value:
                builder.create_template(i, card_value)
    else:
        # Interactive mode
        builder.interactive_template_creation()
    
    # Final stats
    final_stats = builder.detector.get_template_stats()
    print(f"\n📊 Final Template Library:")
    print(f"   Templates: {final_stats['template_count']}")
    print(f"   Available cards: {final_stats['available_cards']}")


if __name__ == "__main__":
    main() 