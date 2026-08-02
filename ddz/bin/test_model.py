"""
Test modified model - verify bomb can beat any card
"""
import torch
import numpy as np
from model import DouDiZhuAgent
from card_utils import CardPattern, get_legal_plays


def test_bomb_recognition():
    """Test if AI recognizes bomb can beat any card"""
    print("="*60)
    print("Test: Can AI recognize bomb can beat any card?")
    print("="*60)
    
    # Create Agent
    agent = DouDiZhuAgent(position=0)
    
    # Test Scene 1: Hand has bomb, opponent plays single card
    print("\nScene 1: Hand 4444555566667777, opponent plays 9")
    hand = list('4444555566667777')
    agent.init_play_state(hand, 0)
    
    # Opponent plays 9
    last_play = CardPattern('9')
    legal_plays = get_legal_plays(hand, last_play)
    
    print(f"  Legal plays count: {len(legal_plays)}")
    print(f"  Has bomb: {any(CardPattern(p).is_bomb for p in legal_plays if p != 'PASS')}")
    
    # AI decision
    suggestion = agent.select_play(legal_plays, epsilon=0.0, last_play=last_play)
    print(f"  AI suggestion: {suggestion}")
    
    # Verify suggestion
    if suggestion == "PASS":
        print("  [FAIL] AI suggests PASS but bomb is available!")
    elif CardPattern(suggestion).is_bomb:
        print("  [PASS] AI chooses to play bomb")
    else:
        print(f"  [PASS] AI chooses {suggestion} (non-bomb can beat)")
    
    # Test Scene 2: Only bomb can beat
    print("\nScene 2: Hand 4444KKKAA2, opponent plays K")
    hand2 = list('4444KKKAA2')
    agent2 = DouDiZhuAgent(position=0)
    agent2.init_play_state(hand2, 0)
    
    last_play2 = CardPattern('K')
    legal_plays2 = get_legal_plays(hand2, last_play2)
    
    print(f"  Legal plays count: {len(legal_plays2)}")
    
    # Check if non-bomb can beat K
    non_bombs_can_beat = []
    for p in legal_plays2:
        if p == 'PASS':
            continue
        pat = CardPattern(p)
        if not pat.is_bomb and pat.can_beat(last_play2):
            non_bombs_can_beat.append(p)
    
    print(f"  Non-bombs that can beat: {non_bombs_can_beat}")
    
    suggestion2 = agent2.select_play(legal_plays2, epsilon=0.0, last_play=last_play2)
    print(f"  AI suggestion: {suggestion2}")
    
    if not non_bombs_can_beat and CardPattern(suggestion2).is_bomb:
        print("  [PASS] No non-bomb can beat, AI plays bomb")
    elif non_bombs_can_beat and suggestion2 in non_bombs_can_beat:
        print("  [PASS] Non-bomb available, AI plays non-bomb")
    elif suggestion2 == "PASS" and not non_bombs_can_beat:
        print("  [FAIL] AI should play bomb instead of PASS!")
    
    # Test Scene 3: Opponent plays bomb
    print("\nScene 3: Hand 5555666677778888, opponent plays 9999 (bomb)")
    hand3 = list('5555666677778888')
    agent3 = DouDiZhuAgent(position=0)
    agent3.init_play_state(hand3, 0)
    
    last_play3 = CardPattern('9999')  # Bomb
    legal_plays3 = get_legal_plays(hand3, last_play3)
    
    print(f"  Legal plays count: {len(legal_plays3)}")
    
    # Check for bigger bombs
    bigger_bombs = []
    for p in legal_plays3:
        if p == 'PASS':
            continue
        pat = CardPattern(p)
        if pat.is_bomb and pat.can_beat(last_play3):
            bigger_bombs.append(p)
    
    print(f"  Bombs that can beat: {bigger_bombs}")
    
    suggestion3 = agent3.select_play(legal_plays3, epsilon=0.0, last_play=last_play3)
    print(f"  AI suggestion: {suggestion3}")
    
    if bigger_bombs and CardPattern(suggestion3).is_bomb:
        print("  [PASS] AI plays bigger bomb")
    elif not bigger_bombs and suggestion3 == "PASS":
        print("  [PASS] No bigger bomb, AI passes")
    
    print("\n" + "="*60)
    print("Test completed")
    print("="*60)


if __name__ == "__main__":
    test_bomb_recognition()
