# Blackjack tips and tricks (Stake.us):

- **Never take insurance.** It's not worth it.
- Stick to the table. Don’t chase losses.
- Blackjack is mostly luck, but correct play keeps losses as low as possible.
- 8 decks are used.
- Dealer stands on all 17s (soft and hard).
- You can **hit**, **stand**, **split pairs**, or **double down**.
- Doubling after a split is **not allowed**.
- Re-splitting is **not allowed**.
- If you split Aces, you get one card per Ace—no more hits.
- Insurance is offered when the dealer shows an Ace (not recommended).
- No surrender option.
- Payouts: 1:1 for a regular win, 3:2 for a natural blackjack.
- House edge: ~0.57% if you use perfect strategy.

# Blackjack Strategy Assistant Usage Guide

## Quick Start

1. **Install Dependencies** (already done):
   ```bash
   source .venv/bin/activate
   pip3 install -r requirements.txt
   ```

2. **Run the Assistant**:
   ```bash
   python3 blackjack_assistant.py
   ```

3. **Setup Capture Region**:
   - The app will show your screen
   - Click and drag to select the area containing the blackjack table
   - Include both player and dealer card areas
   - Press ENTER to confirm

4. **Start Playing**:
   - Open Stake.us blackjack in your browser
   - The assistant will automatically detect cards and provide recommendations

## Usage Modes

### Continuous Monitoring (Default)
```bash
python3 blackjack_assistant.py
```
- Automatically monitors your screen every 2 seconds
- Provides recommendations when new cards are detected
- Press Ctrl+C to stop

### Manual Check
```bash
python3 blackjack_assistant.py --manual
```
- Performs a single card detection check
- Useful for testing or occasional use

### Debug Mode
```bash
python3 blackjack_assistant.py --debug
```
- Shows detailed detection information
- Saves debug images showing detected regions
- Useful for troubleshooting detection issues

### Custom Check Interval
```bash
python3 blackjack_assistant.py --interval 1.0
```
- Changes monitoring frequency (default: 2.0 seconds)
- Lower values = faster detection, higher CPU usage

## Testing & Validation

### Run Unit Tests
```bash
python3 test_blackjack_assistant.py
```

### Test Strategy Logic Only
```bash
python3 blackjack_strategy.py
```

### Test Card Detection Only
```bash
python3 card_detector.py
```

### Test Screen Capture Only
```bash
python3 screen_capture.py
```

## Strategy Reference

The assistant follows basic blackjack strategy exactly as defined in `blackjack-basic-strategy.md`:

- **SPLIT**: Always split Aces and 8s, never split 10s
- **DOUBLE**: Double on 11 vs most dealer cards, soft hands vs weak dealer cards
- **HIT**: Hit hard 16 vs 10, soft totals vs strong dealer cards
- **STAND**: Stand on hard 17+, soft 19+, weak dealer cards vs stiff hands

## Troubleshooting

### Cards Not Detected
1. **Check Capture Region**: Run with `--setup-only` to reconfigure
2. **Improve Lighting**: Ensure cards are clearly visible
3. **Use Debug Mode**: Run with `--debug` to see what's being detected
4. **Check Card Size**: Cards should be reasonably large on screen

### Wrong Recommendations
1. **Verify Card Detection**: Use debug mode to confirm correct cards detected
2. **Check Strategy Chart**: Compare with `blackjack-basic-strategy.md`
3. **Run Tests**: Execute `python3 test_blackjack_assistant.py`

### Performance Issues
1. **Increase Interval**: Use `--interval 3.0` for slower checking
2. **Reduce Capture Region**: Select smaller area around just the cards
3. **Close Other Applications**: Free up system resources

## Command Line Options

```
--debug              Enable debug mode with detailed output
--interval SECONDS   Set monitoring interval (default: 2.0)
--manual            Run single check instead of continuous
--setup-only        Only configure capture region and exit
--test-image PATH   Test detection on a saved screenshot
```

## Example Workflow

1. **First Time Setup**:
   ```bash
   python3 blackjack_assistant.py --setup-only
   ```

2. **Start Playing**:
   ```bash
   python3 blackjack_assistant.py
   ```

3. **If Issues, Debug**:
   ```bash
   python3 blackjack_assistant.py --debug --manual
   ```

The system is designed to be simple and focused - it only detects cards and provides optimal strategy recommendations based on proven basic strategy mathematics. 