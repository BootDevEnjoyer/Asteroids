# Action Space Design Issue

## The Problem

The enemy still spirals even after fixing the state space. Why?

**Current action application:**
```python
target_angle = angle_to_player + (network_output * 90 deg)
```

The formula already points at the player. The network just adds an offset.

## Why This Is Wrong

```
STATE:  "Player is 30 deg to your left!"   (useful info)
           |
           v
       [NETWORK]
           |
           v
ACTION: "Add X degrees to the perfect direction"

Optimal output: X = 0 (always)
```

The network doesn't need to learn anything - the answer is baked into the action formula.

## The Fix

Change action to control turning relative to current heading:

```python
# Before (answer given):
target_angle = angle_to_player + adjustment

# After (must learn):
target_angle = current_angle + (turn_decision * turn_rate)
```

Now the network must USE the state information to decide which way to turn.

## Summary

| Component | Should Contain |
|-----------|---------------|
| **State** | Where is the player? (observations) |
| **Network** | Process observations, decide action |
| **Action** | Raw motor output (turn left/right) |

Don't pre-compute the answer in the action mechanism.

