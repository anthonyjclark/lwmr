# Notes

## Reward Functions

Some notes on reward function design for waypoint navigation.

```python
        # ----------------------------
        # Tunable reward constants
        # ----------------------------
        R_CRASH = -500.0

        R_PROGRESS_FWD = 20.0  # reward per meter of forward path progress
        R_PROGRESS_BACK = 40.0  # penalty per meter of backward path progress

        R_WAYPOINT = 150.0  # bonus per waypoint reached
        R_FINISH = 1000.0  # terminal completion bonus

        R_TIME = -0.01  # per-step penalty
        R_ACTION = -0.001  # small effort penalty

        R_CTE = -0.10  # cross-track penalty coefficient
        CTE_CLIP = 2.0  # cap cross-track penalty magnitude

        # Optional anti-stall term.
        R_STALL = -0.02
        MIN_SPEED = 0.05

        # ----------------------------
        # Terminal failure
        # ----------------------------
        if crashed:
            return R_CRASH, {"event": "crash"}, True

        # ----------------------------
        # Already finished
        # ----------------------------
        if self.waypoint_index >= len(self.waypoints):
            return R_FINISH, {"event": "finish"}, True

        # pos = self.pos.astype(np.float32)
        action = np.asarray(action, dtype=np.float32)

        target = self.waypoints[self.waypoint_index].astype(np.float32)
        dist_to_target = float(np.linalg.norm(target - pos))

        # ----------------------------
        # Segment start/end
        # ----------------------------
        # For the first segment, use the true episode start if available.
        # Otherwise default to origin.
        if self.waypoint_index == 0:
            if hasattr(self, "start_pos"):
                a = self.start_pos.astype(np.float32)
            else:
                a = np.zeros(2, dtype=np.float32)
        else:
            a = self.waypoints[self.waypoint_index - 1].astype(np.float32)

        b = target
        ab = b - a
        seg_len = float(np.linalg.norm(ab)) + 1e-8
        ab_hat = ab / seg_len

        # Progress along current segment, in meters.
        raw_segment_progress = float(np.dot(pos - a, ab_hat))
        segment_progress = float(np.clip(raw_segment_progress, 0.0, seg_len))

        # ----------------------------
        # 1. Forward path progress
        # ----------------------------
        if not hasattr(self, "prev_segment_progress") or self.prev_segment_progress is None:
            delta_progress = 0.0
        else:
            delta_progress = segment_progress - self.prev_segment_progress

        self.prev_segment_progress = segment_progress

        if delta_progress >= 0.0:
            progress_r = R_PROGRESS_FWD * delta_progress
        else:
            progress_r = R_PROGRESS_BACK * delta_progress

        # ----------------------------
        # 2. Waypoint hit / finish
        # ----------------------------
        hit_r = 0.0
        finish_r = 0.0
        terminated = False

        if dist_to_target < self.HIT_RADIUS:
            hit_r = R_WAYPOINT
            self.waypoint_index += 1

            # Reset progress baseline for the next segment.
            self.prev_segment_progress = None

            info["event"] = "waypoint_hit"

            if self.waypoint_index >= len(self.waypoints):
                finish_r = R_FINISH
                terminated = True
                info["event"] = "finish"

        # ----------------------------
        # 3. Time penalty
        # ----------------------------
        time_r = R_TIME

        # ----------------------------
        # 4. Action penalty
        # ----------------------------
        action_r = R_ACTION * float(np.sum(np.square(action)))

        # ----------------------------
        # 5. Cross-track penalty, clipped
        # ----------------------------
        # Important: clipped so that a successful but imperfect trajectory
        # cannot get destroyed by one large CTE term.
        cte = float(self._cross_track_error(pos))
        cte_abs_clipped = min(abs(cte), CTE_CLIP)
        cte_r = R_CTE * cte_abs_clipped

        # ----------------------------
        # 6. Small anti-stall penalty
        # ----------------------------
        if hasattr(self, "vel_world"):
            speed = float(np.linalg.norm(self.vel_world))
        else:
            speed = 0.0

        if not terminated and dist_to_target > self.HIT_RADIUS and speed < MIN_SPEED:
            stall_r = R_STALL
        else:
            stall_r = 0.0

        # ----------------------------
        # Total
        # ----------------------------
        reward = progress_r + hit_r + finish_r + time_r + action_r + cte_r + stall_r

        info.update(
            {
                "dist": dist_to_target,
                "waypoint_index": self.waypoint_index,
                "raw_segment_progress": raw_segment_progress,
                "segment_progress": segment_progress,
                "delta_progress": delta_progress,
                "progress_r": progress_r,
                "hit_r": hit_r,
                "finish_r": finish_r,
                "time_r": time_r,
                "action_r": action_r,
                "cte": cte,
                "cte_r": cte_r,
                "stall_r": stall_r,
                "speed": speed,
                "reward": float(reward),
            }
        )

        return float(reward), info, terminated
```

```python

    W_PROGRESS = 1.0
    W_HIT = 10.0
    W_CTE = 0.1
    W_HEADING = 0.05
    W_ACTION = 0.01
    W_TIME = 0.002
    W_CRASH = 50.0
    W_FINISH = 100.0

    def _compute_reward2(self, action, crashed, pos, yaw, info):

        if crashed:
            info["event"] = "crash"
            return -self.W_CRASH, True

        if self.waypoint_index >= len(self.waypoints):
            info["event"] = "finish"
            return self.W_FINISH, True

        target = self.waypoints[self.waypoint_index]
        dist = float(np.linalg.norm(target - pos))

        # 1. Potential-based progress
        if self.prev_dist is None:
            progress_r = 0.0
        else:
            progress_r = self.W_PROGRESS * (self.prev_dist - dist)
        self.prev_dist = dist

        # 2. Waypoint hit
        hit_r = 0.0
        if dist < self.HIT_RADIUS:
            hit_r = self.W_HIT
            self.waypoint_index += 1
            self.prev_dist = None
            info["event"] = "waypoint_hit"

        # 3. Cross-track penalty
        cte_r = -self.W_CTE * abs(self._cross_track_error(pos))

        # 4. Heading alignment toward current target
        if self.waypoint_index < len(self.waypoints):
            tgt = self.waypoints[self.waypoint_index]
            desired = np.arctan2(tgt[1] - pos[1], tgt[0] - pos[0])
            err = np.arctan2(np.sin(desired - yaw), np.cos(desired - yaw))
            heading_r = self.W_HEADING * np.cos(err)
        else:
            heading_r = 0.0

        # 5. Action effort
        action_r = -self.W_ACTION * float(np.sum(np.square(action)))

        # 6. Time
        time_r = -self.W_TIME

        reward = progress_r + hit_r + cte_r + heading_r + action_r + time_r

        terminated = self.waypoint_index >= len(self.waypoints)
        if terminated:
            reward += self.W_FINISH
            info["event"] = "finish"

        info["dist"] = dist
        info["waypoint_index"] = self.waypoint_index

        return reward, terminated
```
