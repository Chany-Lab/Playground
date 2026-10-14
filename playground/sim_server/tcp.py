"""TCP jog for keyboard teleoperation (Isaac Sim only: PhysX tensor API + torch).

A twist given in the TCP (tool0) frame — [vx, vy, vz] m/s and [wx, wy, wz] deg/s — becomes joint
speeds by damped least squares on the TCP link's Jacobian; the app then moves the drive targets at
those speeds (scene.Drives.jog). The TCP is a link of the robot plus an offset in that link's frame
(robot.yaml tcp: link, offset, joints).
"""

import math

import omni.physics.tensors as physics_tensors
import torch

DAMPING = 0.05  # damped least squares λ: keeps speeds bounded near singularities


def _rotation(q: torch.Tensor) -> torch.Tensor:
    """3×3 rotation of a quaternion (x, y, z, w)."""
    x, y, z, w = q.tolist()
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float64,
    )


def _skew(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.tolist()
    return torch.tensor([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=torch.float64)


class TcpJog:
    def __init__(self, link: str, offset=(0.0, 0.0, 0.0), joints=None, root: str = "/World/Robot"):
        self.link, self.root = link, root
        self.offset = torch.tensor([float(v) for v in offset], dtype=torch.float64)
        self.joints = list(joints) if joints else None
        self.sim = self.view = None

    def _ready(self) -> bool:
        """Tensor views exist only while the simulation plays; rebuilt after a stop / play."""
        if self.view is not None and self.sim.is_valid:
            return True
        try:
            self.sim = physics_tensors.create_simulation_view("torch")
            self.view = self.sim.create_articulation_view(self.root)
            meta = self.view.shared_metatype
            self.link_index = meta.link_names.index(self.link)
            names = list(meta.dof_names)
            self.names = [n for n in names if self.joints is None or n in self.joints]
            self.cols = [names.index(n) for n in self.names]
            self.fixed = meta.fixed_base
            return True
        except Exception:
            self.sim = self.view = None
            return False

    def _frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        """TCP position (world, m) and rotation."""
        tf = self.view.get_link_transforms()[0, self.link_index].to(torch.float64).cpu()
        rot = _rotation(tf[3:7])
        return tf[0:3] + rot @ self.offset, rot

    def pose(self) -> list[float] | None:
        """[x, y, z (m), roll, pitch, yaw (deg)] of the TCP, or None before the simulation plays."""
        if not self._ready():
            return None
        pos, r = self._frame()
        roll = math.atan2(r[2, 1], r[2, 2])
        pitch = math.asin(max(-1.0, min(1.0, -float(r[2, 0]))))
        yaw = math.atan2(r[1, 0], r[0, 0])
        return [float(v) for v in pos] + [math.degrees(a) for a in (roll, pitch, yaw)]

    def joint_speeds(self, twist: list[float]) -> dict[str, float]:
        """Joint → degrees per second that move the TCP by twist (tool frame); {} when not ready."""
        if not self._ready():
            return {}
        _, rot = self._frame()
        v = rot @ torch.tensor(twist[0:3], dtype=torch.float64)
        w = rot @ torch.tensor([math.radians(a) for a in twist[3:6]], dtype=torch.float64)
        row = self.link_index - 1 if self.fixed else self.link_index
        jac = self.view.get_jacobians()[0, row].to(torch.float64).cpu()[:, self.cols]
        # Linear rows are for the link origin; move them to the TCP offset
        jac[0:3] -= _skew(rot @ self.offset) @ jac[3:6]
        target = torch.cat([v, w])
        lam = DAMPING**2 * torch.eye(6, dtype=torch.float64)
        dq = jac.T @ torch.linalg.solve(jac @ jac.T + lam, target)
        return {n: math.degrees(float(s)) for n, s in zip(self.names, dq)}
