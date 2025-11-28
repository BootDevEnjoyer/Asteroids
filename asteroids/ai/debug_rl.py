"""
RL Debug Module - Dumps diagnostic data for training analysis.

Separate from training_log.json, this tracks per-step granular data
to diagnose issues like multi-agent buffer corruption.
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import deque


class RLDebugger:
    """Collects and dumps RL diagnostic data to debug folder."""
    
    def __init__(self, output_dir: str = "debug"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.debug_file = self.output_dir / "rl_debug.json"
        self.session_start = time.time()
        
        # Per-enemy tracking to detect interleaving
        self.enemy_step_counts: Dict[int, int] = {}
        self.buffer_contributions: List[Dict[str, Any]] = []
        
        # Corruption detection
        self.interleave_events: List[Dict[str, Any]] = []
        self.last_enemy_id: Optional[int] = None
        
        # Episode boundary tracking
        self.episode_boundaries: List[Dict[str, Any]] = []
        
        # Rolling stats
        self.recent_rewards = deque(maxlen=50)
        self.recent_distances = deque(maxlen=50)
        
        self._enabled = True
        self._step_counter = 0
        
    def disable(self):
        """Disable debugging (for production)."""
        self._enabled = False
        
    def log_step(self, enemy_id: int, state: list, action: float, 
                 reward: float, distance: float):
        """Log a single RL step with enemy attribution."""
        if not self._enabled:
            return
            
        self._step_counter += 1
        
        # Track per-enemy contributions
        if enemy_id not in self.enemy_step_counts:
            self.enemy_step_counts[enemy_id] = 0
        self.enemy_step_counts[enemy_id] += 1
        
        # Detect interleaving (different enemy than last step)
        if self.last_enemy_id is not None and self.last_enemy_id != enemy_id:
            self.interleave_events.append({
                'step': self._step_counter,
                'from_enemy': self.last_enemy_id,
                'to_enemy': enemy_id,
                'time': time.time() - self.session_start
            })
        self.last_enemy_id = enemy_id
        
        # Track recent stats
        self.recent_rewards.append(reward)
        self.recent_distances.append(distance)
        
        # Sample buffer contributions (every 10th step to limit size)
        if self._step_counter % 10 == 0:
            self.buffer_contributions.append({
                'step': self._step_counter,
                'enemy_id': enemy_id,
                'action': round(action, 4),
                'reward': round(reward, 2),
                'distance': round(distance, 1)
            })
    
    def log_episode_end(self, enemy_id: int, total_reward: float, 
                        success: bool, episode_length: int):
        """Log episode boundary for corruption analysis."""
        if not self._enabled:
            return
            
        self.episode_boundaries.append({
            'step': self._step_counter,
            'enemy_id': enemy_id,
            'total_reward': round(total_reward, 2),
            'success': success,
            'episode_length': episode_length,
            'buffer_size_at_end': self._step_counter,
            'time': time.time() - self.session_start
        })
        
    def get_corruption_summary(self) -> Dict[str, Any]:
        """Generate summary of potential training corruption."""
        total_interleaves = len(self.interleave_events)
        enemies_active = len(self.enemy_step_counts)
        
        # Calculate interleave rate
        interleave_rate = 0.0
        if self._step_counter > 1:
            interleave_rate = total_interleaves / (self._step_counter - 1)
        
        return {
            'total_steps': self._step_counter,
            'enemies_active': enemies_active,
            'enemy_step_distribution': dict(self.enemy_step_counts),
            'total_interleaves': total_interleaves,
            'interleave_rate': round(interleave_rate, 4),
            'corruption_risk': 'HIGH' if interleave_rate > 0.3 else 
                              'MEDIUM' if interleave_rate > 0.1 else 'LOW',
            'avg_recent_reward': round(sum(self.recent_rewards) / max(1, len(self.recent_rewards)), 2),
            'avg_recent_distance': round(sum(self.recent_distances) / max(1, len(self.recent_distances)), 1)
        }
    
    def print_status(self):
        """Print concise debug status to console."""
        if not self._enabled:
            return
            
        summary = self.get_corruption_summary()
        print(f"[RL-DEBUG] Steps: {summary['total_steps']}, "
              f"Enemies: {summary['enemies_active']}, "
              f"Interleaves: {summary['total_interleaves']} ({summary['interleave_rate']:.1%}), "
              f"Risk: {summary['corruption_risk']}")
        
        if summary['enemies_active'] > 0:
            dist = summary['enemy_step_distribution']
            print(f"           Distribution: {dist}")
    
    def dump(self):
        """Write all debug data to file."""
        if not self._enabled:
            return
            
        data = {
            'session_duration': time.time() - self.session_start,
            'corruption_summary': self.get_corruption_summary(),
            'episode_boundaries': self.episode_boundaries[-100:],  # Last 100
            'interleave_events': self.interleave_events[-200:],    # Last 200
            'buffer_samples': self.buffer_contributions[-500:]      # Last 500
        }
        
        with open(self.debug_file, 'w') as f:
            json.dump(data, f, indent=2)


# Global debugger instance
_debugger: Optional[RLDebugger] = None


def get_debugger() -> RLDebugger:
    """Get or create the global RL debugger."""
    global _debugger
    if _debugger is None:
        _debugger = RLDebugger()
    return _debugger


def debug_step(enemy_id: int, state: list, action: float, 
               reward: float, distance: float):
    """Convenience function to log a step."""
    get_debugger().log_step(enemy_id, state, action, reward, distance)


def debug_episode_end(enemy_id: int, total_reward: float, 
                      success: bool, episode_length: int):
    """Convenience function to log episode end."""
    get_debugger().log_episode_end(enemy_id, total_reward, success, episode_length)


def debug_print():
    """Print current debug status."""
    get_debugger().print_status()


def debug_dump():
    """Dump debug data to file."""
    get_debugger().dump()

