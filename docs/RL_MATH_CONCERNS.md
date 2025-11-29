# Reinforcement Learning Math Concerns

This note catalogs issues in the current reinforcement-learning pipeline (`asteroids/ai/brain.py` and `asteroids/entities/enemy.py`). It is written for mathematically trained readers; brief explanations of reinforcement-learning (RL) terminology are included where needed.

## 1. Policy Gradient Has the Wrong Sign and Target

**Context.** In policy-gradient RL, the policy network outputs an action parameter `μ(s)` for each state `s`. During training we sample an action `a` (here, `a = μ(s) + ε` with Gaussian noise `ε`) and adjust the policy so that actions with positive **advantage** `A` become more likely. The standard actor loss is proportional to `-A * log π(a|s)` (stochastic) or `-A * Q(s, μ(s))` (deterministic).

**Implementation.** The code instead defines:

```300:305:asteroids/ai/brain.py
action_loss = torch.mean(torch.sum((action_predictions - actions.detach())**2, dim=1)
                         * (-advantages.detach()))
```

`action_predictions` is `μ(s)`; `actions` is the noisy sample `a`. Taking derivatives shows the gradient step becomes `μ ← μ + 2η(μ - a)A`. Thus:

- If `A > 0` (action was better than predicted), `(μ - a)A > 0`, so μ is pushed away from `a`.
- If `A < 0`, μ is nudged toward the action that underperformed.

This is the opposite of the intended behavior. Moreover, because `a` already includes exploration noise `ε`, the model is regressing toward / away from a random perturbation rather than the noiseless policy output.

**Why it matters.** The update no longer maximizes expected reward; it behaves like weighted behavior cloning with the wrong sign. Even if training appears to improve, the learning signal is mathematically inconsistent, so performance is fragile.

## 2. Returns Are Normalized per Episode but Values Are Not

**Context.** The **return** `R_t` is the discounted sum of future rewards from time `t`. Advantage is `A_t = R_t - V(s_t)`, where `V` is the critic’s value estimate. Both terms must share the same scale.

**Implementation.**

```277:288:asteroids/ai/brain.py
returns = ...
for t in reversed(range(len(rewards))):
    running_return = rewards[t] + gamma * running_return
    returns[t] = running_return
if len(returns) > 1:
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)
advantages = returns - values
```

`returns` are normalized to zero mean and unit variance *within each episode*, but the stored `values` tensor is left on the original reward scale. The critic is still trained against the normalized targets, but future episodes use different scaling, so the value function chases a moving target that generally has mean 0 and variance 1 regardless of true reward magnitudes.

**Why it matters.** The critic cannot represent the expected return in physical units; it only learns to predict a z-scored quantity. Consequently, advantage estimates lose their meaning (they mix normalized returns with unnormalized predictions), making the policy updates noisy and biased. Phase comparisons (e.g., “Phase 1 vs Phase 3 reward”) also become statistically meaningless because every episode is rescaled independently.

## 3. Exploration Noise Is Always Active

**Context.** During evaluation we normally disable exploration noise so we can observe the deterministic policy.

**Implementation.**

```210:220:asteroids/ai/brain.py
if self.training:
    noise = torch.normal(0, self.exploration_noise, size=angle_adjustment.shape)
    angle_adjustment = torch.clamp(angle_adjustment + noise, -1.0, 1.0)
```

`torch.nn.Module.training` defaults to `True`, and the brain object is never switched to evaluation mode (`self.eval()`) for showcase runs. Therefore noise is added in every call, including showcases or performance measurement.

**Why it matters.** Reported behavior and statistics mix policy quality with random jitter. Senior reviewers would expect a deterministic evaluation path, otherwise the measured success rates are not reproducible.

## 4. Policy Update Ignores Action Probabilities

Even disregarding the sign bug, the actor loss only regresses toward the sampled action. There is no parameterization of the action distribution (no log-prob, no variance term, no entropy). A stochastic policy gradient should update proportional to `∇θ log πθ(a|s) * A`. A deterministic policy gradient should use `∇θ μθ(s) * ∇a Q(s, a)`. Neither form appears; thus the learning rule has no theoretical guarantee of improving expected reward.

## 5. Curriculum Advancement Lacks Statistical Confidence

Phase changes occur when:

```365:372:asteroids/ai/brain.py
if self.episodes_in_current_phase < 30:
    return False
return self.get_success_rate() > 0.7
```

`get_success_rate()` averages a deque of recent outcomes without confidence intervals or hypothesis testing. A brief lucky streak can trigger a premature phase jump, while the model’s true success probability might still be below the threshold.

**Why it matters.** Curriculum learning relies on stable estimates of competence. Without variance estimates (e.g., Wilson intervals or Bayesian posteriors), the system can oscillate between phases or solidify incorrect beliefs about readiness.

---

### Summary
The current implementation deviates from standard RL mathematics in several fundamental ways (policy gradient direction, advantage scaling, and action distribution modeling). These issues explain inconsistencies in training outcomes and would be flagged immediately. Addressing them requires re-deriving the actor–critic losses, ensuring consistent scaling between returns and values, gating exploration noise, and adding statistically sound criteria for curriculum progression.

