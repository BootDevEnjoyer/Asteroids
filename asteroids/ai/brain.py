import torch
import torch.nn as nn
import torch.optim as optim
import pygame
import math
import os
import json
import time
from collections import deque
from asteroids.core.constants import *

# global shared brain instance for all AI enemies
_global_brain = None

def get_global_brain():
    """get or create the global AI brain instance"""
    global _global_brain
    if _global_brain is None:
        _global_brain = EnemyBrain()
        print("Created new global AI brain")
    return _global_brain

def save_global_brain():
    """save the global brain to disk"""
    global _global_brain
    if _global_brain is not None:
        _global_brain.save_model()

def reset_global_brain():
    """archive existing model/logs and create fresh brain instance"""
    global _global_brain
    import shutil
    from datetime import datetime
    
    # create archive directory
    archive_dir = "archive"
    os.makedirs(archive_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived_files = []
    
    # archive model files
    model_files = [
        "models/ai_enemy_brain.pth",
        "models/ai_enemy_brain_backup.pth"
    ]
    for model_file in model_files:
        if os.path.exists(model_file):
            base_name = os.path.basename(model_file).replace(".pth", "")
            archive_path = os.path.join(archive_dir, f"{base_name}_{timestamp}.pth")
            shutil.move(model_file, archive_path)
            archived_files.append(archive_path)
    
    # archive debug log
    debug_file = "debug/rl_debug.json"
    if os.path.exists(debug_file):
        archive_path = os.path.join(archive_dir, f"rl_debug_{timestamp}.json")
        shutil.move(debug_file, archive_path)
        archived_files.append(archive_path)
    
    # archive training log
    log_file = "logs/training_log.json"
    if os.path.exists(log_file):
        archive_path = os.path.join(archive_dir, f"training_log_{timestamp}.json")
        shutil.move(log_file, archive_path)
        archived_files.append(archive_path)
    
    # reset global brain instance
    _global_brain = EnemyBrain()
    
    print(f"AI Brain reset! Archived {len(archived_files)} files to {archive_dir}/")
    for f in archived_files:
        print(f"  - {f}")
    
    return _global_brain, archived_files

class TrainingLogger:
    """logs training progress to files for analysis"""
    def __init__(self):
        self.log_file = "logs/training_log.json"
        self.session_start = time.time()
        os.makedirs("logs", exist_ok=True)
        
    def log_episode(self, episode_data):
        """log episode data to file"""
        try:
            # load existing log or create new
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    log_data = json.load(f)
            else:
                log_data = {'sessions': []}
            
            # add timestamp and session info
            episode_data['timestamp'] = time.time()
            episode_data['session_time'] = time.time() - self.session_start
            
            # create new session if this is the first episode
            if not log_data['sessions'] or len(log_data['sessions'][-1].get('episodes', [])) == 0:
                log_data['sessions'].append({
                    'start_time': self.session_start,
                    'episodes': []
                })
            
            # add episode to current session
            log_data['sessions'][-1]['episodes'].append(episode_data)
            
            # keep only last 1000 episodes to prevent file bloat
            if len(log_data['sessions'][-1]['episodes']) > 1000:
                log_data['sessions'][-1]['episodes'] = log_data['sessions'][-1]['episodes'][-1000:]
            
            # save back to file
            with open(self.log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
                
        except Exception as e:
            print(f"Failed to log episode: {e}")

class EnemyBrain(nn.Module):
    """neural network implementing policy gradient reinforcement learning for enemy movement"""
    def __init__(self, input_size=5, hidden_size=64, output_size=1, session_stats=None):
        super().__init__()
        
        # policy network outputs target angle adjustment using actor-critic architecture
        self.policy_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
            nn.Tanh()  # output: target_angle_adjustment (-1 to 1)
        )
        
        # value network estimates state value for advantage calculation
        self.value_network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
        # optimizers with learning rate for stable convergence
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=0.003)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=0.003)
        
        # episode storage for policy gradient calculation
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_values = []
        
        # training metrics for performance tracking
        self.training_steps = 0
        self.total_reward = 0.0
        self.episode_rewards_history = deque(maxlen=100)
        self.success_history = deque(maxlen=50)
        self.phase_success_history = deque(maxlen=20)
        self.current_loss = 0.0
        
        # comprehensive training statistics
        self.training_start_time = time.time()
        self.total_training_time = 0.0
        self.best_episode_reward = 0.0
        self.consecutive_successes = 0
        self.max_consecutive_successes = 0
        self.phase_advancement_count = 0
        
        # exploration parameters for epsilon-greedy-like behavior
        self.exploration_noise = 0.3
        self.noise_decay = 0.9995
        self.min_noise = 0.02
        
        # progressive training phase system
        self.training_phase = 1
        self.phase_episode_count = 0
        self.phase_success_rate = 0.0
        self.episodes_in_current_phase = 0
        
        # adaptive learning rate scheduling
        self.initial_lr = 0.003
        self.current_lr = 0.003
        self.lr_decay = 0.9999
        self.min_lr = 0.0001
        
        # model persistence paths
        self.model_path = "models/ai_enemy_brain.pth"
        self.backup_path = "models/ai_enemy_brain_backup.pth"
        self.training_logger = TrainingLogger()
        
        # reference to session stats for updating display
        self.session_stats = session_stats
        
        self.load_model()
    
    def set_session_stats(self, session_stats):
        """set reference to session statistics for live updates"""
        self.session_stats = session_stats
        
    def forward(self, state):
        """forward pass returning target angle adjustment and state value"""
        angle_adjustment = self.policy_network(state)
        value = self.value_network(state)
        return angle_adjustment, value
    
    def get_action(self, state_tensor):
        """select action using policy network with exploration noise"""
        with torch.no_grad():
            angle_adjustment, value = self.forward(state_tensor)
            
            # add exploration noise during training for policy exploration
            if self.training:
                noise = torch.normal(0, self.exploration_noise, size=angle_adjustment.shape)
                angle_adjustment = angle_adjustment + noise
                angle_adjustment = torch.clamp(angle_adjustment, -1.0, 1.0)
            
            # store for policy gradient calculation
            self.episode_states.append(state_tensor.clone())
            self.episode_actions.append(angle_adjustment.clone())
            self.episode_values.append(value.clone())
        
        return angle_adjustment
    
    def store_reward(self, reward):
        """store reward for current step in episode buffer"""
        self.episode_rewards.append(reward)
        self.total_reward += reward
    
    def end_episode(self, episode_total_reward, success=False):
        """train networks using policy gradient with advantage estimation"""
        if len(self.episode_states) == 0:
            return
            
        # update success tracking for phase progression
        self.success_history.append(1.0 if success else 0.0)
        self.phase_episode_count += 1
        self.episodes_in_current_phase += 1
        
        # track consecutive successes for phase advancement
        if success:
            self.consecutive_successes += 1
            self.max_consecutive_successes = max(self.max_consecutive_successes, self.consecutive_successes)
        else:
            self.consecutive_successes = 0
        
        # update best reward tracking
        self.best_episode_reward = max(self.best_episode_reward, episode_total_reward)
        
        # calculate success rate for current phase
        if len(self.success_history) >= 10:
            self.phase_success_rate = sum(list(self.success_history)[-10:]) / 10.0
        
        # ensure episode data consistency
        min_length = min(len(self.episode_states), len(self.episode_actions), 
                        len(self.episode_rewards), len(self.episode_values))
        
        # truncate all lists to same length if needed
        self.episode_states = self.episode_states[:min_length]
        self.episode_actions = self.episode_actions[:min_length]
        self.episode_rewards = self.episode_rewards[:min_length]
        self.episode_values = self.episode_values[:min_length]
        
        # convert episode data to tensors
        states = torch.stack(self.episode_states)
        actions = torch.stack(self.episode_actions) 
        rewards = torch.tensor(self.episode_rewards, dtype=torch.float32)
        values = torch.stack(self.episode_values).squeeze()
        
        # ensure values tensor has correct dimensions
        if values.dim() == 0:
            values = values.unsqueeze(0)
        
        # calculate discounted returns using temporal difference
        returns = torch.zeros_like(rewards)
        running_return = 0
        gamma = 0.95  # discount factor for future rewards
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
        
        # normalize returns for stable training
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # ensure tensor dimension compatibility
        if values.shape[0] != returns.shape[0]:
            print(f"Warning: Tensor size mismatch - values: {values.shape}, returns: {returns.shape}")
            min_size = min(values.shape[0], returns.shape[0])
            values = values[:min_size]
            returns = returns[:min_size]
        
        # calculate advantages for policy gradient
        advantages = returns - values
        
        # forward pass for current policy
        action_predictions, value_predictions = self.forward(states)
        
        # policy gradient loss using advantage weighting
        action_loss = torch.mean(torch.sum((action_predictions - actions.detach())**2, dim=1) * (-advantages.detach()))
        
        # value function loss for baseline estimation
        value_predictions_flat = value_predictions.squeeze()
        if value_predictions_flat.dim() == 0:
            value_predictions_flat = value_predictions_flat.unsqueeze(0)
        value_loss = nn.MSELoss()(value_predictions_flat, returns.detach())
        
        # update policy network
        self.policy_optimizer.zero_grad()
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.policy_optimizer.step()
        
        # update value network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1.0)
        self.value_optimizer.step()
        
        # update training metrics
        self.current_loss = action_loss.item() + value_loss.item()
        self.episode_rewards_history.append(episode_total_reward)
        self.training_steps += 1
        
        # decay exploration noise over time
        if self.exploration_noise > self.min_noise:
            self.exploration_noise *= self.noise_decay
        
        # decay learning rate for convergence
        if self.current_lr > self.min_lr:
            self.current_lr *= self.lr_decay
            for param_group in self.policy_optimizer.param_groups:
                param_group['lr'] = self.current_lr
            for param_group in self.value_optimizer.param_groups:
                param_group['lr'] = self.current_lr
        
        # log episode data for analysis
        episode_data = {
            'episode': self.training_steps,
            'phase': self.training_phase,
            'reward': episode_total_reward,
            'success': success,
            'success_rate': self.phase_success_rate,
            'exploration_noise': self.exploration_noise,
            'learning_rate': self.current_lr,
            'loss': self.current_loss,
            'episode_length': len(self.episode_states)
        }
        self.training_logger.log_episode(episode_data)
        
        # auto-save periodically with higher frequency for successful episodes
        save_interval = 10 if success else 25
        if self.training_steps % save_interval == 0:
            self.save_model()
        
        # clear episode buffers
        self.episode_states.clear()
        self.episode_actions.clear() 
        self.episode_rewards.clear()
        self.episode_values.clear()
    
    def should_advance_phase(self):
        """determine if AI should advance to next training phase based on performance"""
        if self.episodes_in_current_phase < 30:
            return False
        
        # Advance when success rate exceeds 70%
        return self.get_success_rate() > 0.7
    
    def advance_phase(self):
        """advance to next training phase with increased difficulty"""
        if self.training_phase < 3:
            prev_success_rate = self.get_success_rate()
            self.training_phase += 1
            self.phase_episode_count = 0
            self.episodes_in_current_phase = 0
            self.consecutive_successes = 0
            self.phase_advancement_count += 1
            
            # increase exploration when advancing phases
            self.exploration_noise = min(0.2, self.exploration_noise * 1.5)
            
            print(f"AI Advanced to Training Phase {self.training_phase}!")
            print(f"   Success rate in previous phase: {prev_success_rate:.1%}")
            print(f"   Phase advancements so far: {self.phase_advancement_count}")
            
            # save milestone checkpoint
            milestone_path = f"ai_brain_phase_{self.training_phase}_milestone.pth"
            self.save_model(milestone_path)
    
    def get_average_reward(self):
        """calculate average reward over recent episodes"""
        if len(self.episode_rewards_history) == 0:
            return 0.0
        return float(sum(self.episode_rewards_history) / len(self.episode_rewards_history))
    
    def get_success_rate(self):
        """calculate recent success rate for performance monitoring"""
        if len(self.success_history) == 0:
            return 0.0
        return float(sum(self.success_history) / len(self.success_history))
    
    def get_training_summary(self):
        """generate comprehensive training summary for analysis"""
        training_time = time.time() - self.training_start_time + self.total_training_time
        hours = training_time / 3600
        
        return {
            'total_episodes': self.training_steps,
            'training_phase': self.training_phase,
            'success_rate': self.get_success_rate(),
            'best_reward': self.best_episode_reward,
            'training_hours': hours,
            'episodes_per_hour': self.training_steps / max(0.01, hours),
            'max_consecutive_successes': self.max_consecutive_successes,
            'phase_advancements': self.phase_advancement_count,
            'exploration_noise': self.exploration_noise,
            'learning_rate': self.current_lr
        }
    
    def save_model(self, path=None):
        """save trained model and training state to disk"""
        save_path = path or self.model_path
        
        # ensure models directory exists
        os.makedirs(os.path.dirname(save_path) or "models", exist_ok=True)
        
        try:
            # create backup of existing model
            if os.path.exists(self.model_path) and save_path == self.model_path:
                import shutil
                shutil.copy2(self.model_path, self.backup_path)
            
            checkpoint = {
                'policy_network_state_dict': self.policy_network.state_dict(),
                'value_network_state_dict': self.value_network.state_dict(),
                'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
                'value_optimizer_state_dict': self.value_optimizer.state_dict(),
                'training_steps': self.training_steps,
                'total_reward': self.total_reward,
                'exploration_noise': self.exploration_noise,
                'training_phase': self.training_phase,
                'phase_episode_count': self.phase_episode_count,
                'episodes_in_current_phase': self.episodes_in_current_phase,
                'episode_rewards': list(self.episode_rewards_history),
                'success_history': list(self.success_history),
                'best_episode_reward': self.best_episode_reward,
                'consecutive_successes': self.consecutive_successes,
                'max_consecutive_successes': self.max_consecutive_successes,
                'phase_advancement_count': self.phase_advancement_count,
                'total_training_time': self.total_training_time + (time.time() - self.training_start_time),
                'current_lr': self.current_lr,
                'save_timestamp': time.time(),
                'training_summary': self.get_training_summary()
            }
            torch.save(checkpoint, save_path)
            
            # print save confirmation with summary
            if save_path == self.model_path:
                summary = self.get_training_summary()
                print(f"💾 AI Brain saved! Episode {summary['total_episodes']}, Phase {summary['training_phase']}")
                print(f"   Success: {summary['success_rate']:.1%}, Best: {summary['best_reward']:.1f}, Hours: {summary['training_hours']:.1f}")
                
        except Exception as e:
            print(f"Failed to save AI brain: {e}")
    
    def load_model(self):
        """load previously trained model and training state from disk"""
        if os.path.exists(self.model_path):
            try:
                print(f"=== LOADING AI BRAIN FROM {self.model_path} ===")
                checkpoint = torch.load(self.model_path, map_location='cpu')
                
                # verify model architecture compatibility
                policy_state_dict = checkpoint['policy_network_state_dict']
                expected_input_size = list(self.policy_network[0].parameters())[0].shape[1]
                actual_input_size = list(policy_state_dict.values())[0].shape[1]
                
                if expected_input_size != actual_input_size:
                    print(f"⚠️  Model architecture mismatch!")
                    print(f"   Expected input size: {expected_input_size}, Found: {actual_input_size}")
                    print(f"   Creating fresh model with enhanced features...")
                    return
                
                self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
                self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
                self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
                self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
                self.training_steps = checkpoint['training_steps']
                self.total_reward = checkpoint['total_reward']
                self.exploration_noise = checkpoint['exploration_noise']
                self.training_phase = checkpoint.get('training_phase', 1)
                self.phase_episode_count = checkpoint.get('phase_episode_count', 0)
                self.episodes_in_current_phase = checkpoint.get('episodes_in_current_phase', 0)
                self.episode_rewards_history = deque(checkpoint['episode_rewards'], maxlen=100)
                self.success_history = deque(checkpoint.get('success_history', []), maxlen=50)
                self.best_episode_reward = checkpoint.get('best_episode_reward', 0.0)
                self.consecutive_successes = checkpoint.get('consecutive_successes', 0)
                self.max_consecutive_successes = checkpoint.get('max_consecutive_successes', 0)
                self.phase_advancement_count = checkpoint.get('phase_advancement_count', 0)
                self.total_training_time = checkpoint.get('total_training_time', 0.0)
                self.current_lr = checkpoint.get('current_lr', self.initial_lr)
                
                # update optimizer learning rates
                for param_group in self.policy_optimizer.param_groups:
                    param_group['lr'] = self.current_lr
                for param_group in self.value_optimizer.param_groups:
                    param_group['lr'] = self.current_lr
                
                print(f"[OK] AI brain successfully loaded!")
                summary = self.get_training_summary()
                print(f"   Phase {summary['training_phase']}, Episodes: {summary['total_episodes']}")
                print(f"   Success: {summary['success_rate']:.1%}, Training: {summary['training_hours']:.1f}h")
                print(f"   Best reward: {summary['best_reward']:.1f}, Max streak: {summary['max_consecutive_successes']}")
                print(f"   Exploration: {self.exploration_noise:.3f}, LR: {self.current_lr:.6f}")
                print(f"=== END LOAD ===")
            except Exception as e:
                print(f"[ERROR] Failed to load AI brain: {e}")
                print("Starting with fresh neural network with enhanced features")
                # attempt to load backup
                if os.path.exists(self.backup_path):
                    print("Attempting to load backup...")
                    try:
                        checkpoint = torch.load(self.backup_path, map_location='cpu')
                        # verify architecture compatibility for backup
                        policy_state_dict = checkpoint['policy_network_state_dict']
                        expected_input_size = list(self.policy_network[0].parameters())[0].shape[1]
                        actual_input_size = list(policy_state_dict.values())[0].shape[1]
                        
                        if expected_input_size == actual_input_size:
                            # load essential data only
                            self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
                            self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
                            print("[OK] Backup loaded successfully!")
                        else:
                            print("[ERROR] Backup also has architecture mismatch, starting fresh")
                    except Exception:
                        print("[ERROR] Backup also failed, starting fresh with enhanced model")
        else:
            print(f"[NEW] No saved AI brain found at {self.model_path}")
            print("Starting with fresh neural network with enhanced features")


class GameStateCollector:
    """collects and normalizes game state for neural network input"""
    
    @staticmethod
    def collect_state(enemy, player, asteroid_group=None, enemy_group=None):
        """
        collect normalized game state for reinforcement learning
        
        returns: torch.tensor with 5 normalized values:
        [0]: angle to player (normalized to -1,1)
        [1]: distance to player (normalized 0-1) 
        [2]: current movement speed (normalized 0-1)
        [3]: alignment with target direction (-1 to 1)
        [4]: relative angle - signed direction to turn (-1 to 1)
        """
        
        # calculate direct distance to player without screen wrapping
        direction_to_player = player.position - enemy.position
        distance_to_player = direction_to_player.length()
        
        # calculate angle to player (absolute, in world coordinates)
        if distance_to_player > 0:
            angle_to_player = math.atan2(direction_to_player.y, direction_to_player.x)
            # normalize angle from [-pi, pi] to [-1, 1]
            normalized_angle = angle_to_player / math.pi
        else:
            angle_to_player = 0.0
            normalized_angle = 0.0
        
        # normalize distance (0 = touching, 1 = far away)
        max_distance = 800.0
        normalized_distance = min(1.0, distance_to_player / max_distance)
        
        # normalize current speed
        enemy_velocity = getattr(enemy, 'velocity', pygame.Vector2(0, 0))
        current_speed = enemy_velocity.length()
        normalized_speed = min(1.0, current_speed / (ENEMY_SPEED * 2))
        
        # calculate movement efficiency: alignment between current and target direction
        alignment = 0.0
        if distance_to_player > 0 and current_speed > 0:
            target_direction = direction_to_player.normalize()
            current_direction = enemy_velocity.normalize()
            alignment = target_direction.dot(current_direction)  # -1 to 1
        
        # calculate signed relative angle: which way should the enemy turn?
        # positive = player is to the left (turn left), negative = player is to the right
        relative_angle = 0.0
        if hasattr(enemy, 'current_angle'):
            relative_angle = angle_to_player - enemy.current_angle
            # wrap to [-pi, pi]
            while relative_angle > math.pi:
                relative_angle -= 2 * math.pi
            while relative_angle < -math.pi:
                relative_angle += 2 * math.pi
            # normalize to [-1, 1]
            relative_angle = relative_angle / math.pi
        
        # create state tensor with normalized features
        state = torch.tensor([
            normalized_angle,     # where is the player? (-1 to 1)
            normalized_distance,  # how far is the player? (0 to 1)  
            normalized_speed,     # how fast am I moving? (0 to 1)
            alignment,            # am I moving toward the player? (-1 to 1)
            relative_angle        # which way should I turn? (-1 to 1)
        ], dtype=torch.float32)
        
        return state


class AIMetricsDisplay:
    """handles display of AI learning metrics and training progress"""
    
    def __init__(self):
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)
        
    def draw_metrics(self, screen, brain):
        """render AI training progress and phase information on screen"""
        # background panel positioned on right side
        panel_width = 350
        panel_height = 220
        panel_x = SCREEN_WIDTH - panel_width - 10
        panel_y = 50
        
        # semi-transparent background
        panel_surface = pygame.Surface((panel_width, panel_height))
        panel_surface.set_alpha(190)
        panel_surface.fill((20, 20, 40))
        screen.blit(panel_surface, (panel_x, panel_y))
        
        # border with phase-specific color coding
        phase_colors = {1: (100, 255, 100), 2: (255, 200, 100), 3: (255, 100, 100)}
        border_color = phase_colors.get(brain.training_phase, (100, 150, 255))
        pygame.draw.rect(screen, border_color, 
                        (panel_x, panel_y, panel_width, panel_height), 3)
        
        # title with current training phase
        phase_names = {1: "Stationary", 2: "Slow Moving", 3: "Advanced"}
        phase_name = phase_names.get(brain.training_phase, "Unknown")
        title_text = self.font.render(f"AI Phase {brain.training_phase}: {phase_name}", True, border_color)
        screen.blit(title_text, (panel_x + 10, panel_y + 5))
        
        # metrics with vertical spacing
        y_offset = panel_y + 35
        line_height = 20
        
        # success rate with color coding
        success_rate = brain.get_success_rate()
        success_color = "green" if success_rate > 0.6 else "yellow" if success_rate > 0.3 else "white"
        success_text = self.small_font.render(f"Success Rate: {success_rate:.1%}", True, success_color)
        screen.blit(success_text, (panel_x + 10, y_offset))
        y_offset += line_height
        
        # phase progress and consecutive successes
        phase_text = self.small_font.render(f"Phase Episodes: {brain.episodes_in_current_phase}", True, "white")
        screen.blit(phase_text, (panel_x + 10, y_offset))
        y_offset += line_height
        
        streak_text = self.small_font.render(f"Current Streak: {brain.consecutive_successes}", True, "white")
        screen.blit(streak_text, (panel_x + 10, y_offset))
        y_offset += line_height
        
        # total training episodes
        steps_text = self.small_font.render(f"Total Episodes: {brain.training_steps}", True, "white")
        screen.blit(steps_text, (panel_x + 10, y_offset))
        y_offset += line_height
        
        # reward metrics
        avg_reward = brain.get_average_reward()
        reward_text = self.small_font.render(f"Avg Reward: {avg_reward:.1f}", True, "white")
        screen.blit(reward_text, (panel_x + 10, y_offset))
        y_offset += line_height
        
        best_text = self.small_font.render(f"Best Reward: {brain.best_episode_reward:.1f}", True, "white")
        screen.blit(best_text, (panel_x + 10, y_offset))
        y_offset += line_height
        
        # learning parameters
        noise_text = self.small_font.render(f"Exploration: {brain.exploration_noise:.3f}", True, "white")
        screen.blit(noise_text, (panel_x + 10, y_offset))
        y_offset += line_height
        
        lr_text = self.small_font.render(f"Learning Rate: {brain.current_lr:.4f}", True, "white")
        screen.blit(lr_text, (panel_x + 10, y_offset))
        y_offset += line_height
        
        # training time display
        training_time = time.time() - brain.training_start_time + brain.total_training_time
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        time_text = self.small_font.render(f"Training Time: {hours:02d}h {minutes:02d}m", True, "white")
        screen.blit(time_text, (panel_x + 10, y_offset)) 