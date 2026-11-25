"""Team-safe starts and optional swept peer termination for the 2024 crossing example."""
from types import FunctionType

import numpy as np
import torch


MIN_CENTER_M = 0.20
PAIRS = np.array(((0, 1), (0, 2), (1, 2)))


def team_minimum(positions):
    points = np.asarray(positions, dtype=np.float64)
    differences = points[..., PAIRS[:, 0], :] - points[..., PAIRS[:, 1], :]
    return np.linalg.norm(differences, axis=-1).min(axis=-1)


def swept_contacts(samples):
    """Return separation squared per interval, scene and pair for linear relative motion."""
    points = np.asarray(samples, dtype=np.float64)
    relative = points[:, :, PAIRS[:, 0], :] - points[:, :, PAIRS[:, 1], :]
    start = relative[:-1]
    change = np.diff(relative, axis=0)
    length2 = np.einsum("...i,...i->...", change, change)
    projection = -np.einsum("...i,...i->...", start, change)
    fraction = np.divide(projection, length2, out=np.zeros_like(length2), where=length2 > 0)
    closest = start + np.clip(fraction, 0., 1.)[..., None] * change
    return np.einsum("...i,...i->...", closest, closest)


class SafeReset:
    """Reject joint proposals before the author applies positions or renders observations."""

    def __init__(self, *args, **kwargs):
        self.accepted_teams = 0
        self.rejected_teams = 0
        self.min_accepted_m = float("inf")
        super().__init__(*args, **kwargs)

    def reset_agents(self, indices=None, state=None):
        explicit = state is not None
        count = self.dynamics.num
        ids = np.arange(count) if indices is None else np.atleast_1d(np.asarray(indices, dtype=int))
        if ids.ndim != 1 or len(ids) != len(set(ids.tolist())) or np.any((ids < 0) | (ids >= count)):
            raise ValueError("Reset requires distinct valid agent indices")
        state = state if explicit else self._generate_state(ids)
        scenes, inverse = np.unique(ids // 3, return_inverse=True)
        existing = self.dynamics.position.detach().cpu().numpy().reshape(-1, 3, 3)
        candidates = existing[scenes].astype(np.float64)
        proposed = state[0] if indices is None else state[0].to(self.dynamics.position.dtype)
        candidates[inverse, ids % 3] = proposed.detach().cpu().numpy()
        if not np.isfinite(candidates).all():
            raise ValueError("Non-finite coordinates in team reset")

        minima = team_minimum(candidates)
        rejected = 0
        for group in np.flatnonzero(minima < MIN_CENTER_M):
            if explicit:
                raise ValueError(f"Explicit reset overlaps peers in scene {scenes[group]}")
            rows = np.flatnonzero(inverse == group)
            movable = ids[rows] % 3
            fixed = sorted(set(range(3)).difference(movable))
            if len(fixed) == 2 and np.linalg.norm(np.diff(candidates[group, fixed], axis=0)) < MIN_CENTER_M:
                raise ValueError("Two unreset peers already overlap; reset their team together")
            for _ in range(4096):
                rejected += 1
                new = self._generate_state(ids[rows])
                for initial, replacement in zip(state, new):
                    initial[rows] = replacement
                candidates[group, movable] = state[0][rows].detach().cpu().numpy()
                minima[group] = team_minimum(candidates[group])
                if minima[group] >= MIN_CENTER_M:
                    break
            else:
                raise RuntimeError(f"No legal joint start after 4096 fresh proposals in scene {scenes[group]}")
        result = super().reset_agents(indices=indices, state=state)
        self.accepted_teams += len(scenes)
        self.rejected_teams += rejected
        if len(minima):
            self.min_accepted_m = min(self.min_accepted_m, float(minima.min()))
        return result


class PeerEpisode:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.num_agent_per_scene != 3:
            raise ValueError("This experiment requires a three-drone team per scene")
        self.last_peer_contact = np.zeros(self.num_scene, dtype=bool)
        self.last_peer_substep = np.zeros(self.num_scene, dtype=int)
        self.last_peer_pair = np.zeros((self.num_scene, 2), dtype=int)
        self.last_peer_min_m = np.full(self.num_scene, np.inf)
        self._peer_agents = torch.zeros(self.num_agent, dtype=torch.bool)
        if self.peer_termination:
            self._record_substeps()

    def spawn_stats(self):
        return dict(accepted_teams=self.envs.accepted_teams, rejected_teams=self.envs.rejected_teams,
                    min_accepted_m=self.envs.min_accepted_m)

    def _record_substeps(self):
        dynamics = self.envs.dynamics
        intervals = dynamics._interval_steps
        samples = np.empty((intervals + 1, self.num_scene, 3, 3), dtype=np.float64)
        original_motors = dynamics._run_motors
        original_step = dynamics.step
        sample_index = 0

        def copy_position(index):
            samples[index] = dynamics.position.detach().cpu().numpy().reshape(self.num_scene, 3, 3)

        def motors(*args, **kwargs):
            nonlocal sample_index
            # This hook is called exactly once at the start of each original integrator iteration.
            copy_position(sample_index)
            sample_index += 1
            return original_motors(*args, **kwargs)

        def step(action):
            nonlocal sample_index
            sample_index = 0
            result = original_step(action)
            if sample_index != intervals:
                raise RuntimeError(f"Captured {sample_index} substeps, expected {intervals}")
            copy_position(intervals)
            self._set_contacts(swept_contacts(samples))
            return result

        dynamics._run_motors = motors
        dynamics.step = step

    def _set_contacts(self, distance2):
        distance_m = np.sqrt(distance2)
        # Match the independent geometry audit's numerical equality at tangency in metres.
        hits = distance_m < MIN_CENTER_M - 1e-12
        at_interval = hits.any(axis=2)
        self.last_peer_contact = at_interval.any(axis=0)
        first = np.argmax(at_interval, axis=0)
        self.last_peer_substep = np.where(self.last_peer_contact, first + 1, 0)
        pair = hits[first, np.arange(self.num_scene)].argmax(axis=1)
        self.last_peer_pair = np.where(self.last_peer_contact[:, None], PAIRS[pair] + 1, 0)
        self.last_peer_min_m = distance_m.min(axis=(0, 2))
        involved = np.zeros((self.num_scene, 3), dtype=bool)
        hit_pairs = hits.any(axis=0)
        for k, (i, j) in enumerate(PAIRS):
            involved[:, i] |= hit_pairs[:, k]
            involved[:, j] |= hit_pairs[:, k]
        self._peer_agents = torch.from_numpy(involved.reshape(-1))

    def get_success(self):
        result = super().get_success()
        if self.peer_termination:
            return result & ~self._peer_agents
        return result

    @property
    def is_collision(self):
        value = super().is_collision
        return value | self._peer_agents if self.peer_termination else value

    def reset(self):
        obs = super().reset()
        self._peer_agents.zero_()
        self.last_peer_contact.fill(False)
        self.last_peer_substep.fill(0)
        self.last_peer_pair.fill(0)
        self.last_peer_min_m.fill(np.inf)
        return obs

    def reset_by_id(self, indices):
        result = super().reset_by_id(indices)
        self._peer_agents.view(-1, 3)[torch.as_tensor(indices) // 3] = False
        return result

    def step(self, action, is_test=False):
        observations, rewards, dones, infos = super().step(action, is_test=is_test)
        for scene in np.flatnonzero(dones.reshape(-1, 3)[:, 0]):
            contact = bool(self.last_peer_contact[scene])
            for info in infos[3 * scene:3 * scene + 3]:
                info["peer_contact"] = contact
                if contact:
                    info["TimeLimit.truncated"] = False
                    info["peer_substep"] = int(self.last_peer_substep[scene])
                    info["peer_pair"] = self.last_peer_pair[scene].tolist()
                    info["peer_min_m"] = float(self.last_peer_min_m[scene])
        return observations, rewards, dones, infos


def install(rl, terminate: bool):
    """Select a local class for this official entry point, leaving original modules untouched."""
    from envs.droneEnv import DroneEnvsBase
    from envs.multiDroneGymEnv import MultiDroneGymEnvBase
    from envs.MultiNavigationEnv import MultiNavigationEnv

    class ValidStartDroneEnvs(SafeReset, DroneEnvsBase):
        pass

    # The upstream gym constructor hardcodes DroneEnvsBase. Reuse its function with a
    # private dependency binding so constructor-time reset is safe even for the very first pose.
    fn = MultiDroneGymEnvBase.__init__
    bindings = dict(fn.__globals__, DroneEnvsBase=ValidStartDroneEnvs)

    class ValidStartGym(MultiDroneGymEnvBase):
        __init__ = FunctionType(fn.__code__, bindings, fn.__name__, fn.__defaults__, fn.__closure__)

    class Crossing(PeerEpisode, MultiNavigationEnv, ValidStartGym):
        peer_termination = terminate

    rl.MultiNavigationEnv = Crossing
    return Crossing
