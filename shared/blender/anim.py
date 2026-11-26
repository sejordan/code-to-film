"""Shared tools for animating a shot: timing curves, the frame walk and the
camera. A film's own shot files import what they need from here.

Times are seconds from the start of the shot.
"""
import math

from mathutils import Vector

import core


def ease(t, a, b):
    """0 before a, 1 after b, smooth between."""
    return core.smooth((t - a) / (b - a))


def ramp(t, a, b):
    """0 before a, 1 after b, a straight line between."""
    return core.clamp01((t - a) / (b - a))


def coast(u):
    """Push off from rest, then coast to a stop."""
    u = core.clamp01(u)
    return 1.0 - (1.0 - u ** 1.5) ** 2


def wiggle(t, a, b, cycles):
    """A few swings between a and b that start and end at rest."""
    if t <= a or t >= b:
        return 0.0
    u = (t - a) / (b - a)
    return math.sin(math.tau * cycles * u) * math.sin(math.pi * u)


def vlerp(a, b, s):
    return Vector(a).lerp(Vector(b), s)


def frames(shot):
    """Every frame of the shot plus one spare at each end, for motion blur.
    Gives (Blender frame number, seconds into the shot)."""
    for i in range(-1, shot["frames"] + 1):
        yield shot["block_start"] + i, i / core.FPS


class Cam:
    """The shot's camera: where it is, what it looks at, what is sharp.
    st is what the set's build() returned; its "cam" is the camera."""

    def __init__(self, st, lens=50.0, fstop=4.0):
        self.cam = st["cam"]
        self.cam.data.lens = lens
        self.cam.data.dof.aperture_fstop = fstop
        self.prev = None

    def key(self, f, pos, target, focus=None):
        pos, target = Vector(pos), Vector(target)
        self.prev = core.aim(self.cam, pos, target, self.prev)
        core.key(self.cam, f, "location", "rotation_quaternion")
        fwd = (target - pos).normalized()
        point = target if focus is None else Vector(focus)
        self.cam.data.dof.focus_distance = max(0.03, (point - pos).dot(fwd))
        self.cam.data.keyframe_insert("dof.focus_distance", frame=f)
