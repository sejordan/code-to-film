"""The table: a marble, a domino, and the light over them.

build(sc) makes the set at real size, in metres, and returns its parts.
animate(sc, st, shot) keys one shot's motion. trace(shot) reports, per
frame, what the sound needs to follow the picture (shared/blender/motion.py
collects it into out/motion.json).
"""
import math

import bpy
from mathutils import Vector

import core
from anim import Cam, coast, ease, frames

SAMPLES = 96                    # Cycles samples per pixel for this set

R = 0.012                       # the marble's radius
DOMINO = (0.012, 0.042, 0.085)  # thickness (along the marble's path), width, height
DOMINO_X = 0.10                 # the domino's centre
TOUCH = DOMINO_X - DOMINO[0] / 2 - R     # the marble's centre when it touches the domino
HIT = 1.3                       # seconds into S02 when it does
SPEED = 0.1                     # metres a second


def marble_x(shot, t):
    """Where the marble's centre is, along the table, t seconds into a shot."""
    if shot["name"] == "S01":
        return -0.65 + SPEED * t
    if t <= HIT:
        return TOUCH - SPEED * (HIT - t)
    return TOUCH - 0.035 * coast((t - HIT) / 1.6)      # knocked back, rolling to a stop


def tip(shot, t):
    """How far the domino has fallen, in radians: 0 standing, pi/2 flat."""
    if shot["name"] != "S02" or t <= HIT:
        return 0.0
    u = min(1.0, (t - HIT) / 0.42)
    fall = (math.pi / 2) * u * u                       # gravity: slow, then fast
    land = t - HIT - 0.42
    if land > 0.0:                                     # one small bounce as it lands
        fall -= 0.06 * math.sin(math.pi * min(1.0, land / 0.18))
    return fall


def build(sc):
    world = bpy.data.worlds.new("room")
    sc.world = world
    world.use_nodes = True
    bg = next(n for n in world.node_tree.nodes if n.type == "BACKGROUND")
    bg.inputs["Color"].default_value = (0.20, 0.23, 0.28, 1.0)
    bg.inputs["Strength"].default_value = 0.35

    key = bpy.data.lights.new("key", "AREA")
    key.energy, key.size, key.color = 60.0, 0.6, (1.0, 0.93, 0.82)
    ob = bpy.data.objects.new("key", key)
    sc.collection.objects.link(ob)
    ob.rotation_mode = "QUATERNION"
    core.aim(ob, (-0.5, -0.6, 0.9), (0.0, 0.0, 0.0))

    wood = core.plain("wood", (0.42, 0.25, 0.13), rough=0.45)
    core.box(sc, "table", (1.6, 0.9, 0.04), (0.0, 0.0, -0.02), wood, bevel=0.004)
    glass = core.plain("glass", (0.15, 0.45, 0.85), rough=0.05)
    marble = core.ball(sc, "marble", R, (0.0, 0.0, R), glass)
    marble.rotation_mode = "XYZ"
    stripe = core.cyl(sc, "stripe", R * 1.004, R * 0.5, (0.0, 0.0, 0.0), core.plain("stripe", (0.95, 0.95, 0.9), rough=0.2),
                      axis="X", parent=marble)
    ivory = core.plain("ivory", (0.9, 0.87, 0.8), rough=0.35)
    hinge = core.empty(sc, "hinge", (DOMINO_X + DOMINO[0] / 2, 0.0, 0.0))     # the far bottom edge it falls over
    core.box(sc, "domino", DOMINO, (-DOMINO[0] / 2, 0.0, DOMINO[2] / 2), ivory, bevel=0.0015, parent=hinge)
    cam = core.new_cam(sc, "main")
    return dict(cam=cam, marble=marble, hinge=hinge, stripe=stripe)


def animate(sc, st, shot):
    sc.frame_start = shot["block_start"]
    sc.frame_end = shot["block_start"] + shot["frames"] - 1
    cam = Cam(st, lens=35.0 if shot["name"] == "S01" else 50.0, fstop=2.8)
    for f, t in frames(shot):
        x = marble_x(shot, t)
        st["marble"].location = (x, 0.0, R)
        st["marble"].rotation_euler = (0.0, x / R, 0.0)          # rolling: turned by distance over radius
        core.key(st["marble"], f, "location", "rotation_euler")
        st["hinge"].rotation_euler = (0.0, tip(shot, t), 0.0)
        core.key(st["hinge"], f, "rotation_euler")
        if shot["name"] == "S01":
            follow = Vector((x + 0.04, 0.0, 0.015))
            cam.key(f, follow + Vector((0.05, -0.42, 0.16)), follow, focus=(x, 0.0, R))
        else:
            push = ease(t, 0.0, shot["seconds"]) * 0.03
            cam.key(f, (0.02 + push, -0.24 + push, 0.035), (0.075, 0.0, 0.03), focus=(TOUCH, 0.0, R))


def trace(shot):
    """Per frame: how far the marble has rolled, and how far the domino has fallen."""
    ts = [i / core.FPS for i in range(shot["frames"])]
    return dict(x=[round(marble_x(shot, t), 5) for t in ts], tip=[round(tip(shot, t), 4) for t in ts])
