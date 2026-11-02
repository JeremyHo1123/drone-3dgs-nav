"""
Geometry and frame conversions for the orbit task (no torch; used by the inference code and the flight node).

Frames:
    ENU     MAVROS local frame (east, north, up): /mavros/local_position/pose, velocity_local
    scene   frame of the training scene sunny_baseball_field, defined by the ArUco tag on the ground
            during capture (z up, ground at z = 0)
    body    FLU (forward, left, up); quaternions (x, y, z, w) rotate body -> world

The orbit used in training (envs/drone_vla_orbit.py in grad_nav with the simulator patch):
    box centre (scene) ORBIT_CENTER = (-0.077, 0.575), radius 2.5 m, flight-controller height 1.0 m, 0.5 m/s
    start bearing theta (direction from the box centre to the drone; scene x axis = 0 deg, counter-clockwise
    positive), nose pointing at the box: psi = theta + 180 deg
    task 0 "CLOCKWISE" has direction -1 (theta decreasing), task 1 "COUNTERCLOCKWISE" +1
    The nose always points at the box, so the accumulated nose yaw = the angle orbited (same direction)
"""
import collections
import math
from dataclasses import dataclass

ORBIT_CENTER_SCENE = (-0.077, 0.575)
ORBIT_RADIUS = 2.5          # m, training value
ORBIT_HEIGHT = 1.0          # m, flight controller above ground (scene z)
FC_HEIGHT = 0.313           # m, flight controller above ground when the drone stands on the ground
# During capture the ArUco tag lay at the scene origin. Bearing of the tag seen from the box centre (for finding the start point on site)
TAG_BEARING_DEG = math.degrees(math.atan2(-ORBIT_CENTER_SCENE[1], -ORBIT_CENTER_SCENE[0])) % 360.0  # 277.6


@dataclass
class SceneState:
    z: float
    quat: tuple          # (x, y, z, w)
    vel: tuple           # (vx, vy, vz)


def wrap(a):
    """Wrap an angle to (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def quat_mul(a, b):  # (x,y,z,w) Hamilton product
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


def yaw_of(q):
    x, y, z, w = q
    return math.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))


def heading_xy(q):
    """Direction of the body x axis (the nose) projected on the ground, normalised.
    Formula for formula the same as proper_heading_xy() in envs/drone_vla_orbit.py (including the +1e-8)."""
    x, y, z, w = q
    fx = 1 - 2 * (y * y + z * z)
    fy = 2 * (x * y + w * z)
    n = math.hypot(fx, fy) + 1e-8
    return fx / n, fy / n


def tilt_quat(q):
    """Attitude in the v2 observation: the quaternion with yaw removed, q_yaw(psi)^-1 * q, with w >= 0.
    Formula for formula the same as tilt_quat() in envs/drone_vla_orbit.py.
    Independent of world yaw: ENU and the scene frame differ only by a rotation about z, so an ENU or a
    scene quaternion gives the same result; q and -q also give the same result (w >= 0 at the end)."""
    x, y, z, w = q
    half = 0.5 * math.atan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))
    qt = quat_mul((0.0, 0.0, -math.sin(half), math.cos(half)), q)
    return tuple(-c for c in qt) if qt[3] < 0 else qt


def heading_frame_vel(vel, q):
    """Velocity in the v2 observation: world velocity rotated into the heading frame (forward, left, up),
    yaw only, no tilt. Formula for formula the same as to_heading_frame(v, proper_heading_xy(q)) in
    envs/drone_vla_orbit.py. Independent of world yaw (velocity and nose rotate together)."""
    hx, hy = heading_xy(q)
    vx, vy, vz = vel
    return (hx * vx + hy * vy, -hy * vx + hx * vy, vz)


def tilt_deg(q):
    """Angle between body z and world z, degrees: cos = 1 - 2(x^2 + y^2) (unit quaternion)."""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if not n > 0.0:
        return float("nan")
    c = 1.0 - 2.0 * (x * x + y * y) / n
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def scene_state_from_enu(pos_enu, quat_enu, vel_enu, ground_z_enu, yaw_offset, fc_height=FC_HEIGHT):
    """MAVROS (ENU, body FLU) -> scene frame.

    pos_enu / quat_enu / vel_enu: /mavros/local_position/pose and /mavros/local_position/velocity_local
    ground_z_enu: local z while standing on the ground before take-off
    yaw_offset: psi_enu - psi_scene (yaw of the same nose direction in ENU and in the scene), see yaw_offset_from_start()
    fc_height: the flight controller (body origin) is 0.313 m above ground when the drone stands on the ground
    """
    c, s = math.cos(-yaw_offset), math.sin(-yaw_offset)
    qz = (0.0, 0.0, math.sin(-yaw_offset / 2), math.cos(-yaw_offset / 2))
    q = quat_mul(qz, quat_enu)
    vx, vy, vz = vel_enu
    return SceneState(z=pos_enu[2] - ground_z_enu + fc_height, quat=q,
                      vel=(c * vx - s * vy, s * vx + c * vy, vz))


def yaw_offset_from_start(quat_enu, start_deg):
    """Call with the drone on the orbit start point (scene bearing start_deg), nose pointing at the box.
    In the scene this nose direction has yaw start_deg + 180 deg."""
    return yaw_offset_from_yaw(yaw_of(quat_enu), start_deg)


def yaw_offset_from_yaw(yaw_enu, start_deg):
    return wrap(yaw_enu - (math.radians(start_deg) + math.pi))


def circle_center_enu(x, y, yaw_enu, radius=ORBIT_RADIUS):
    """Drone on the circle, nose pointing at the box: the box centre is radius metres straight ahead."""
    return x + radius * math.cos(yaw_enu), y + radius * math.sin(yaw_enu)


def circular_mean(angles):
    s = sum(math.sin(a) for a in angles)
    c = sum(math.cos(a) for a in angles)
    return math.atan2(s, c)


def task_direction(task_name):
    """Task string -> orbit direction (+1 counter-clockwise, -1 clockwise), as in DroneVLAOrbitEnv.TASKS."""
    t = task_name.upper()
    if "COUNTERCLOCKWISE" in t:
        return 1
    if "CLOCKWISE" in t:
        return -1
    raise ValueError(f"cannot tell the orbit direction from the task name: {task_name!r}")


class OrbitProgress:
    """Angle orbited, from the accumulated nose yaw (the nose always points at the box, so the two are equal).

    update(yaw, t) returns the angle accumulated in the commanded direction (radians, negative when going the
    other way). The yaw change per step is far below pi (simulator limit 0.5 rad/s x 0.05 s), so wrapping
    step-by-step differences never jumps."""

    def __init__(self, yaw0, direction, t0, keep_s=60.0):
        self.prev = float(yaw0)
        self.dir = int(direction)
        self.acc = 0.0
        self.t0 = float(t0)
        self.keep_s = float(keep_s)
        self.hist = collections.deque()     # (t, progress)
        self.hist.append((self.t0, 0.0))

    def update(self, yaw, t):
        if math.isfinite(yaw):
            self.acc += wrap(yaw - self.prev)
            self.prev = yaw
        p = self.dir * self.acc
        self.hist.append((t, p))
        while self.hist and t - self.hist[0][0] > self.keep_s:
            self.hist.popleft()
        return p

    @property
    def progress(self):
        return self.dir * self.acc

    def gained(self, window_s, t):
        """Angle gained in the last window_s seconds (radians); None while the record is shorter than that."""
        if t - self.t0 < window_s:
            return None
        ref = None
        for tt, pp in self.hist:
            if tt <= t - window_s:
                ref = pp
            else:
                break
        if ref is None:
            ref = self.hist[0][1]
        return self.progress - ref
