"""
Screen Capture Module for Blackjack Strategy Assistant
Handles real-time screen capture with region selection
"""

import cv2
import numpy as np
import pyautogui
import mss
from typing import Tuple, Optional
import time

class ScreenCapture:
    def __init__(self):
        # Disable pyautogui failsafe for smoother operation
        pyautogui.FAILSAFE = False
        
        # Initialize MSS for faster screenshots
        self.sct = mss.mss()
        
        # Capture region (will be set by user)
        self.capture_region = None
        
    def get_screen_dimensions(self) -> Tuple[int, int]:
        """Get full screen dimensions"""
        return pyautogui.size()
    
    def select_capture_region_interactive(self) -> Tuple[int, int, int, int]:
        """
        Interactive region selection using mouse
        Returns (x, y, width, height)
        """
        print("\n" + "="*50)
        print("INTERACTIVE REGION SELECTION")
        print("="*50)
        print("Instructions:")
        print("1. Position your browser with Stake.us blackjack table visible")
        print("2. Click and drag to select the area containing the cards")
        print("3. Try to include both player and dealer card areas")
        print("4. Press ENTER after selection or ESC to use full screen")
        print("="*50)
        
        try:
            # Take a screenshot for selection
            screenshot = pyautogui.screenshot()
            screenshot_np = np.array(screenshot)
            screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
            
            # Create window for selection
            cv2.namedWindow('Select Blackjack Table Region', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Select Blackjack Table Region', 1200, 800)
            
            # Variables for mouse selection
            selecting = False
            start_point: Optional[Tuple[int, int]] = None
            end_point: Optional[Tuple[int, int]] = None
            current_img = screenshot_cv.copy()
            
            def mouse_callback(event, x, y, flags, param):
                nonlocal selecting, start_point, end_point, current_img
                
                if event == cv2.EVENT_LBUTTONDOWN:
                    selecting = True
                    start_point = (x, y)
                    end_point = (x, y)
                
                elif event == cv2.EVENT_MOUSEMOVE and selecting and start_point:
                    end_point = (x, y)
                    current_img = screenshot_cv.copy()
                    cv2.rectangle(current_img, start_point, end_point, (0, 255, 0), 2)
                    cv2.imshow('Select Blackjack Table Region', current_img)
                
                elif event == cv2.EVENT_LBUTTONUP and start_point:
                    selecting = False
                    end_point = (x, y)
                    current_img = screenshot_cv.copy()
                    cv2.rectangle(current_img, start_point, end_point, (0, 255, 0), 2)
                    cv2.imshow('Select Blackjack Table Region', current_img)
            
            cv2.setMouseCallback('Select Blackjack Table Region', mouse_callback)
            cv2.imshow('Select Blackjack Table Region', screenshot_cv)
            
            print("Click and drag to select region, then press ENTER or ESC...")
            
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 13:  # Enter key
                    break
                elif key == 27:  # ESC key
                    start_point = None
                    end_point = None
                    break
            
            cv2.destroyAllWindows()
            
            if start_point is not None and end_point is not None:
                x1, y1 = start_point
                x2, y2 = end_point
                
                # Ensure positive width and height
                x = min(x1, x2)
                y = min(y1, y2)
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                
                print(f"Selected region: ({x}, {y}) {width}x{height}")
                return (x, y, width, height)
            else:
                print("No region selected, using full screen")
                width, height = self.get_screen_dimensions()
                return (0, 0, width, height)
                
        except Exception as e:
            print(f"Error in region selection: {e}")
            print("Using full screen as fallback")
            width, height = self.get_screen_dimensions()
            return (0, 0, width, height)
    
    def set_capture_region(self, region: Tuple[int, int, int, int]):
        """Set the capture region (x, y, width, height)"""
        self.capture_region = region
        print(f"Capture region set to: {region}")
    
    def capture_region_fast(self) -> np.ndarray:
        """Fast screenshot capture using MSS"""
        if not self.capture_region:
            # Capture full screen if no region set
            monitor = self.sct.monitors[1]  # Primary monitor
        else:
            x, y, width, height = self.capture_region
            monitor = {"top": y, "left": x, "width": width, "height": height}
        
        # Capture screenshot
        screenshot = self.sct.grab(monitor)
        
        # Convert to numpy array
        img = np.array(screenshot)
        
        # Convert from BGRA to BGR (remove alpha channel)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        return img
    
    def capture_region_pyautogui(self) -> np.ndarray:
        """Fallback screenshot capture using pyautogui"""
        if not self.capture_region:
            screenshot = pyautogui.screenshot()
        else:
            x, y, width, height = self.capture_region
            screenshot = pyautogui.screenshot(region=(x, y, width, height))
        
        # Convert to numpy array and BGR
        screenshot_np = np.array(screenshot)
        screenshot_cv = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
        
        return screenshot_cv
    
    def capture_screenshot(self, use_fast=True) -> np.ndarray:
        """
        Capture screenshot of the selected region
        """
        try:
            if use_fast:
                return self.capture_region_fast()
            else:
                return self.capture_region_pyautogui()
        except Exception as e:
            print(f"Fast capture failed: {e}, falling back to pyautogui")
            return self.capture_region_pyautogui()
    
    def save_screenshot(self, filename: str = "blackjack_screenshot.png") -> bool:
        """Save current screenshot to file"""
        try:
            screenshot = self.capture_screenshot()
            cv2.imwrite(filename, screenshot)
            print(f"Screenshot saved as {filename}")
            return True
        except Exception as e:
            print(f"Failed to save screenshot: {e}")
            return False
    
    def show_live_preview(self, duration: int = 10):
        """
        Show live preview of the capture region for testing
        """
        print(f"Showing live preview for {duration} seconds...")
        print("Press 'q' to quit early")
        
        start_time = time.time()
        
        while (time.time() - start_time) < duration:
            screenshot = self.capture_screenshot()
            
            # Resize for display if too large
            height, width = screenshot.shape[:2]
            if width > 1200 or height > 800:
                scale = min(1200/width, 800/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                screenshot = cv2.resize(screenshot, (new_width, new_height))
            
            cv2.imshow('Live Preview - Press q to quit', screenshot)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            time.sleep(0.1)  # 10 FPS
        
        cv2.destroyAllWindows()
        print("Live preview ended")


# Test function
if __name__ == "__main__":
    capture = ScreenCapture()
    
    print("Screen Capture Module Test")
    print("1. Select capture region")
    print("2. Show live preview")
    print("3. Save test screenshot")
    
    # Interactive region selection
    region = capture.select_capture_region_interactive()
    capture.set_capture_region(region)
    
    # Show live preview
    capture.show_live_preview(5)
    
    # Save test screenshot
    capture.save_screenshot("test_capture.png") 