"""
Card Detection Module for Blackjack Strategy Assistant
Uses OpenCV Template Matching for accurate card detection
"""

import cv2
import numpy as np
import os
from PIL import Image
from typing import List, Tuple, Optional, Dict
import json


class TemplateCardDetector:
    def __init__(self):
        # Template storage
        self.templates = {}
        self.template_dir = "card_templates"
        self.confidence_threshold = 0.8  # 80% confidence minimum
        
        # Ensure template directory exists
        os.makedirs(self.template_dir, exist_ok=True)
        
        # Load existing templates if available
        self.load_templates()
    
    def save_templates(self):
        """Save template metadata to JSON file"""
        metadata = {
            'template_count': len(self.templates),
            'available_cards': list(self.templates.keys()),
            'confidence_threshold': self.confidence_threshold
        }
        
        metadata_path = os.path.join(self.template_dir, 'templates_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_templates(self):
        """Load card templates from the template directory"""
        try:
            # Load template images
            for filename in os.listdir(self.template_dir):
                if filename.endswith('.png') and not filename.startswith('debug'):
                    card_value = filename.replace('.png', '')
                    template_path = os.path.join(self.template_dir, filename)
                    template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                    
                    if template_img is not None:
                        self.templates[card_value] = template_img
                        
            print(f"✅ Loaded {len(self.templates)} card templates: {list(self.templates.keys())}")
                        
        except Exception as e:
            print(f"⚠️  Could not load templates: {e}")
    
    def create_template_from_region(self, card_image: np.ndarray, card_value: str) -> bool:
        """
        Create and save a template from a card region
        This is used to build the template library
        """
        try:
            # Preprocess the card image for template creation
            if len(card_image.shape) == 3:
                gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = card_image
            
            # Focus on the top-left corner where the card value is located
            height, width = gray.shape
            # Extract top-left 40% of the card (where rank/value typically appears)
            template_region = gray[:int(height * 0.4), :int(width * 0.4)]
            
            # Enhance contrast for better template matching
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(template_region)
            
            # Resize to standard template size
            template_size = (60, 80)  # Standard template dimensions
            template = cv2.resize(enhanced, template_size)
            
            # Save template
            template_path = os.path.join(self.template_dir, f"{card_value}.png")
            cv2.imwrite(template_path, template)
            
            # Store in memory
            self.templates[card_value] = template
            
            print(f"✅ Created template for card: {card_value}")
            self.save_templates()
            return True
            
        except Exception as e:
            print(f"❌ Error creating template for {card_value}: {e}")
            return False
    
    def match_card_template(self, card_image: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Match a card image against all templates using normalized cross-correlation
        Returns (card_value, confidence) or (None, 0.0) if no good match
        """
        try:
            # Check if this is a face-down card first
            if self.is_face_down_card(card_image):
                return None, 0.0
                
            # Preprocess the card image
            if len(card_image.shape) == 3:
                gray = cv2.cvtColor(card_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = card_image
            
            # Focus on top-left corner for value detection
            height, width = gray.shape
            value_region = gray[:int(height * 0.4), :int(width * 0.4)]
            
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(value_region)
            
            best_match_value = None
            best_confidence = 0.0
            
            # Try each template
            for card_value, template in self.templates.items():
                # Resize card region to match template size if needed
                template_height, template_width = template.shape
                resized_region = cv2.resize(enhanced, (template_width, template_height))
                
                # Perform template matching using normalized cross-correlation
                result = cv2.matchTemplate(resized_region, template, cv2.TM_CCOEFF_NORMED)
                
                # Get the maximum correlation value
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                # Check if this is the best match so far
                if max_val > best_confidence:
                    best_confidence = max_val
                    best_match_value = card_value
            
            # Only return match if confidence is above threshold
            if best_confidence >= self.confidence_threshold:
                return best_match_value, best_confidence
            else:
                return None, best_confidence
                
        except Exception as e:
            print(f"❌ Template matching error: {e}")
            return None, 0.0
    
    def is_face_down_card(self, card_image: np.ndarray) -> bool:
        """
        Detect if a card is face-down (blue with "Stake" branding)
        """
        try:
            # Convert to HSV for better blue detection
            hsv = cv2.cvtColor(card_image, cv2.COLOR_BGR2HSV)
            
            # Define blue color range for Stake cards
            lower_blue = np.array([90, 50, 50])
            upper_blue = np.array([130, 255, 255])
            
            # Create mask for blue pixels
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            
            # Calculate percentage of blue pixels
            blue_pixels = np.sum(blue_mask > 0)
            total_pixels = card_image.shape[0] * card_image.shape[1]
            blue_percentage = blue_pixels / total_pixels
            
            # If more than 40% of the card is blue, it's likely face-down
            return blue_percentage > 0.4
            
        except Exception as e:
            # If detection fails, assume it's face-up to be safe
            return False


class CardDetector:
    def __init__(self):
        # Initialize the template-based detector
        self.template_detector = TemplateCardDetector()
    
    def extract_card_value_from_image(self, card_image: np.ndarray) -> Optional[str]:
        """
        Extract card value using template matching
        """
        # Use template matching for card detection
        card_value, confidence = self.template_detector.match_card_template(card_image)
        
        if card_value is not None:
            print(f"✅ Template match: {card_value} (confidence: {confidence:.2f})")
            return card_value
        
        print(f"⚠️  No template match found (best confidence: {confidence:.2f})")
        return None
    
    def create_template_for_card(self, card_image: np.ndarray, card_value: str) -> bool:
        """
        Helper method to create a template from a detected card region
        """
        return self.template_detector.create_template_from_region(card_image, card_value)
    
    def get_template_stats(self) -> Dict:
        """Get information about available templates"""
        return {
            'template_count': len(self.template_detector.templates),
            'available_cards': list(self.template_detector.templates.keys()),
            'confidence_threshold': self.template_detector.confidence_threshold
        }

    def detect_card_regions(self, screenshot: np.ndarray, debug=False) -> List[Tuple[int, int, int, int]]:
        """
        Detect potential card regions in the screenshot
        Returns list of (x, y, width, height) tuples
        """
        # Convert to grayscale
        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Multiple detection approaches for maximum sensitivity
        all_contours = []
        
        # Method 1: Multiple Canny edge detection with different thresholds
        for low, high in [(10, 40), (15, 60), (30, 100), (50, 150)]:
            edges = cv2.Canny(blurred, low, high)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(contours)
            
            # Also try with RETR_TREE to catch internal contours
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(contours)
        
        # Method 2: Adaptive thresholding
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours)
        
        # Method 3: Different adaptive threshold
        adaptive_thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
        contours, _ = cv2.findContours(adaptive_thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours)
        
        # Method 4: Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        morph = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours)
        
        card_regions = []
        
        for contour in all_contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on size and aspect ratio
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            
            # Very lenient criteria to catch all possible cards
            if (0.1 < aspect_ratio < 4.0 and  # Extremely flexible aspect ratio
                200 < area < 200000 and       # Very wide area range
                w > 12 and h > 15):           # Very low minimum dimensions
                
                # Check for duplicates/overlaps with much higher tolerance
                is_duplicate = False
                for existing_x, existing_y, existing_w, existing_h in card_regions:
                    # Calculate overlap
                    overlap_x = max(0, min(x + w, existing_x + existing_w) - max(x, existing_x))
                    overlap_y = max(0, min(y + h, existing_y + existing_h) - max(y, existing_y))
                    overlap_area = overlap_x * overlap_y
                    
                    # Only skip if almost identical (98% overlap)
                    if overlap_area > 0.98 * min(area, existing_w * existing_h):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    card_regions.append((x, y, w, h))
        
        # Try to split large regions that might contain multiple cards
        split_regions = []
        for x, y, w, h in card_regions:
            area = w * h
            aspect_ratio = w / h
            
            # If region is very large or wide, try to split it
            if area > 15000 or aspect_ratio > 1.8:
                # Try vertical split for side-by-side cards
                if w > h * 1.5:  # Wide region, likely side-by-side cards
                    # Split into left and right halves
                    left_w = w // 2
                    right_w = w - left_w
                    
                    # Add left half
                    if left_w > 12 and h > 15:
                        split_regions.append((x, y, left_w, h))
                    
                    # Add right half  
                    if right_w > 12 and h > 15:
                        split_regions.append((x + left_w, y, right_w, h))
                else:
                    # Keep original if splitting doesn't make sense
                    split_regions.append((x, y, w, h))
            else:
                split_regions.append((x, y, w, h))
        
        # Use split regions if we got more cards
        if len(split_regions) > len(card_regions):
            card_regions = split_regions
        
        # Sort by area (largest first) to prioritize well-defined cards
        card_regions.sort(key=lambda region: region[2] * region[3], reverse=True)
        
        # Limit to reasonable number of cards (max 12 to avoid too much noise)
        card_regions = card_regions[:12]
        
        if debug:
            print(f"Found {len(card_regions)} potential card regions")
            for i, (x, y, w, h) in enumerate(card_regions):
                print(f"  Region {i}: ({x}, {y}) {w}x{h}, aspect: {w/h:.2f}, area: {w*h}")
        
        return card_regions
    
    def extract_cards_from_regions(self, screenshot: np.ndarray, regions: List[Tuple[int, int, int, int]]) -> List[str]:
        """
        Extract card values from detected regions using template matching
        """
        cards = []
        
        for i, (x, y, w, h) in enumerate(regions):
            # Extract card region
            card_image = screenshot[y:y+h, x:x+w]
            
            # Try to detect the card value using template matching
            card_value = self.extract_card_value_from_image(card_image)
            
            if card_value:
                cards.append(card_value)
        
        return cards
    
    def detect_cards_from_screenshot(self, screenshot: np.ndarray, debug=False) -> Tuple[List[str], List[str]]:
        """
        Main method to detect player and dealer cards from screenshot
        Returns (player_cards, dealer_cards)
        
        Uses screen position to determine player vs dealer cards:
        - Top half of screen = Dealer cards (filter out face-down cards)
        - Bottom half of screen = Player cards (filter out face-down cards)
        """
        # Detect all card regions
        regions = self.detect_card_regions(screenshot, debug=debug)
        
        # Get screen height to determine midpoint
        screen_height = screenshot.shape[0]
        screen_midpoint = screen_height // 2
        
        # Separate regions into dealer (top) and player (bottom) based on Y coordinate
        dealer_regions = []
        player_regions = []
        
        for x, y, w, h in regions:
            card_center_y = y + h // 2
            if card_center_y < screen_midpoint:
                dealer_regions.append((x, y, w, h))
            else:
                player_regions.append((x, y, w, h))
        
        # Sort regions left to right within each group
        dealer_regions.sort(key=lambda region: region[0])  # Sort by X coordinate
        player_regions.sort(key=lambda region: region[0])  # Sort by X coordinate
        
        # Process ALL dealer cards and filter out face-down cards
        dealer_cards = []
        for x, y, w, h in dealer_regions:
            card_image = screenshot[y:y+h, x:x+w]
            
            # Check if it's a face-down card first
            if not self.template_detector.is_face_down_card(card_image):
                # It's face-up, try to extract the value
                card_value = self.extract_card_value_from_image(card_image)
                if card_value:
                    dealer_cards.append(card_value)
                    if debug:
                        print(f"✅ Dealer card detected: {card_value}")
                else:
                    if debug:
                        print(f"⚠️  Dealer face-up card detected but couldn't identify value")
            else:
                if debug:
                    print(f"🔵 Dealer face-down card ignored")
        
        # Process ALL player cards and filter out face-down cards  
        player_cards = []
        for x, y, w, h in player_regions:
            card_image = screenshot[y:y+h, x:x+w]
            
            # Check if it's a face-down card first
            if not self.template_detector.is_face_down_card(card_image):
                # It's face-up, try to extract the value
                card_value = self.extract_card_value_from_image(card_image)
                if card_value:
                    player_cards.append(card_value)
                    if debug:
                        print(f"✅ Player card detected: {card_value}")
                else:
                    if debug:
                        print(f"⚠️  Player face-up card detected but couldn't identify value")
            else:
                if debug:
                    print(f"🔵 Player face-down card ignored")
        
        if debug:
            print(f"Screen height: {screen_height}, midpoint: {screen_midpoint}")
            print(f"Dealer regions: {dealer_regions}")
            print(f"Player regions: {player_regions}")
            print(f"Dealer cards (face-up only): {dealer_cards}")
            print(f"Player cards (face-up only): {player_cards}")
        
        return player_cards, dealer_cards
    
    def save_debug_image(self, screenshot: np.ndarray, regions: List[Tuple[int, int, int, int]], filename: str = "debug_detection.png"):
        """
        Save screenshot with detected regions marked for debugging
        """
        import os
        
        # Ensure debug_images directory exists
        debug_dir = "debug_images"
        os.makedirs(debug_dir, exist_ok=True)
        
        debug_img = screenshot.copy()
        
        for i, (x, y, w, h) in enumerate(regions):
            # Draw rectangle around detected card
            cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Add label
            cv2.putText(debug_img, f"Card {i}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Save to debug_images folder
        debug_path = os.path.join(debug_dir, filename)
        cv2.imwrite(debug_path, debug_img)
        print(f"Debug image saved as {debug_path}")


# Test function
if __name__ == "__main__":
    import pyautogui
    
    detector = CardDetector()
    
    print("Template-based card detector initialized.")
    print(f"Available templates: {detector.get_template_stats()}")
    print("To test, take a screenshot of a blackjack game and this will try to detect cards.")
    print("Press Ctrl+C to exit.")
    
    try:
        # Take a screenshot
        screenshot = pyautogui.screenshot()
        screenshot_np = np.array(screenshot)
        screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        
        # Detect cards
        player_cards, dealer_cards = detector.detect_cards_from_screenshot(screenshot_cv, debug=True)
        
        print(f"Player cards: {player_cards}")
        print(f"Dealer cards: {dealer_cards}")
        
        # Save debug image
        regions = detector.detect_card_regions(screenshot_cv)
        detector.save_debug_image(screenshot_cv, regions)
        
    except KeyboardInterrupt:
        print("\nExiting...") 