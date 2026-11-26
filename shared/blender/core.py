"""Shared tools for every set: render settings, camera, materials, shapes.

Everything is built at real size, in metres.
"""
import json
import math
import os
import warnings

import bmesh
import bpy
from mathutils import Matrix, Vector

warnings.simplefilter("ignore", DeprecationWarning)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASSETS = os.path.join(ROOT, "assets", "polyhaven")
FPS = 24            # render.py and motion.py set this from the film being made


# ---------------------------------------------------------------- scene

def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    return bpy.context.scene


def pick_device(sc, want="AUTO"):
    """Use the GPU: OptiX or CUDA on rented NVIDIA cards, Metal on a Mac."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    order = ["OPTIX", "CUDA", "METAL"] if want == "AUTO" else [want]
    for kind in order:
        if kind == "CPU":
            break
        try:
            prefs.compute_device_type = kind
            prefs.get_devices()
        except Exception:
            continue
        gpus = [d for d in prefs.devices if d.type == kind]
        if gpus:
            for d in prefs.devices:
                d.use = d.type == kind
            sc.cycles.device = "GPU"
            return kind + ": " + ", ".join(d.name for d in gpus)
    sc.cycles.device = "CPU"
    return "CPU"


def setup_render(sc, samples=160, scale=100, device="AUTO"):
    r = sc.render
    r.engine = "CYCLES"
    r.resolution_x, r.resolution_y = 3840, 2160
    r.resolution_percentage = scale
    r.fps = FPS
    cy = sc.cycles
    cy.samples = samples
    cy.use_adaptive_sampling = True
    cy.adaptive_threshold = 0.02
    cy.use_denoising = True
    cy.denoiser = "OPENIMAGEDENOISE"
    try:
        cy.denoising_input_passes = "RGB_ALBEDO_NORMAL"
        cy.denoising_prefilter = "ACCURATE"
    except Exception:
        pass
    # Blender denoises on the CPU unless told otherwise. On the first cloud run
    # that was most of each 4K frame: the GPU traced for about 5 s, then sat
    # idle while the host CPU denoised for 6 to 22 s, depending on the host.
    try:
        cy.denoising_use_gpu = True
    except Exception:
        pass
    cy.max_bounces = 7
    cy.diffuse_bounces = 4
    cy.glossy_bounces = 4
    cy.transmission_bounces = 6
    cy.transparent_max_bounces = 12
    cy.sample_clamp_indirect = 6.0
    cy.sample_clamp_direct = 0.0
    cy.caustics_reflective = False
    cy.caustics_refractive = False
    cy.blur_glossy = 0.6
    cy.filter_width = 1.5
    # Persistent data keeps the scene loaded between frames, which is faster,
    # but in Blender 5.3 an object renders dark on the first frame after it
    # stops moving (its motion-blur data goes stale). Seen on a toy's
    # body in an early test. Off until that is fixed: every frame is built fresh.
    r.use_persistent_data = False
    r.use_motion_blur = True
    r.motion_blur_shutter = 0.5          # a 180 degree film shutter
    try:
        r.motion_blur_position = "CENTER"
    except Exception:
        pass
    vs = sc.view_settings
    vs.view_transform = "AgX"
    vs.look = "AgX - Medium High Contrast"
    vs.exposure = 0.0
    r.image_settings.file_format = "JPEG"
    r.image_settings.quality = 96
    r.film_transparent = False
    return pick_device(sc, device)


def new_cam(sc, name, lens=50.0, fstop=4.0):
    cd = bpy.data.cameras.new("CAM_" + name)
    cam = bpy.data.objects.new("CAM_" + name, cd)
    sc.collection.objects.link(cam)
    cd.sensor_width = 36.0
    cd.lens = lens
    cd.dof.use_dof = True
    cd.dof.aperture_fstop = fstop
    cd.dof.aperture_blades = 0
    cd.clip_start = 0.005
    cd.clip_end = 400.0
    cam.rotation_mode = "QUATERNION"
    sc.camera = cam
    return cam


def aim(cam, pos, target, prev=None):
    """Place the camera and point it at a target. Rotation is a quaternion
    kept on the same side as the previous frame's, so it can never flip
    between two frames and smear the picture through motion blur."""
    pos, target = Vector(pos), Vector(target)
    q = (target - pos).to_track_quat("-Z", "Y")
    if prev is not None and q.dot(prev) < 0.0:
        q.negate()
    cam.location = pos
    cam.rotation_quaternion = q
    return q


# ---------------------------------------------------------------- easing

def clamp01(u):
    return 0.0 if u < 0.0 else 1.0 if u > 1.0 else u


def smooth(u):
    u = clamp01(u)
    return u * u * (3.0 - 2.0 * u)


def bell(u):
    """A smooth bump: 0 at u = -1 and u = 1, 1 at u = 0."""
    if u <= -1.0 or u >= 1.0:
        return 0.0
    return 0.5 * (1.0 + math.cos(math.pi * u))


# ---------------------------------------------------------------- materials

def new_mat(name):
    m = bpy.data.materials.new(name)
    if m.node_tree is None or not any(n.type == "BSDF_PRINCIPLED" for n in m.node_tree.nodes):
        m.use_nodes = True
    nt = m.node_tree
    bsdf = next(n for n in nt.nodes if n.type == "BSDF_PRINCIPLED")
    return m, nt, bsdf


def scan(name, asset, size=None, rough_mul=1.0, normal=1.0, tint=None, grain=0.0, turn=0.0):
    """A photo-scanned Poly Haven surface, box-projected in object space at
    its real-world size, so no object needs UVs. turn spins the picture
    about the up axis, to run a wood grain the other way."""
    d = os.path.join(ASSETS, asset)
    with open(os.path.join(d, "info.json")) as f:
        info = json.load(f)
    size = size or info.get("size_m", 1.0)
    m, nt, bsdf = new_mat(name)
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0 / size,) * 3
    mp.inputs["Rotation"].default_value = (0.0, 0.0, turn)
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])

    def tex(fname, space):
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(os.path.join(d, fname), check_existing=True)
        n.image.colorspace_settings.name = space
        n.projection = "BOX"
        n.projection_blend = 0.25
        nt.links.new(mp.outputs["Vector"], n.inputs["Vector"])
        return n

    col = tex("diff.jpg", "sRGB")
    if tint is not None:
        mix = nt.nodes.new("ShaderNodeMixRGB")
        mix.blend_type = "MULTIPLY"
        mix.inputs["Fac"].default_value = 1.0
        mix.inputs["Color2"].default_value = (*tint, 1.0)
        nt.links.new(col.outputs["Color"], mix.inputs["Color1"])
        nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
    else:
        nt.links.new(col.outputs["Color"], bsdf.inputs["Base Color"])
    rough = tex("rough.jpg", "Non-Color")
    if rough_mul != 1.0:
        mul = nt.nodes.new("ShaderNodeMath")
        mul.operation = "MULTIPLY"
        mul.use_clamp = True
        mul.inputs[1].default_value = rough_mul
        nt.links.new(rough.outputs["Color"], mul.inputs[0])
        nt.links.new(mul.outputs["Value"], bsdf.inputs["Roughness"])
    else:
        nt.links.new(rough.outputs["Color"], bsdf.inputs["Roughness"])
    nor = tex("nor_gl.jpg", "Non-Color")
    nm = nt.nodes.new("ShaderNodeNormalMap")
    nm.inputs["Strength"].default_value = normal
    nt.links.new(nor.outputs["Color"], nm.inputs["Color"])
    if grain > 0.0:
        # a scan runs out of detail when the camera is centimetres away, so
        # surfaces seen that close get a fine procedural grain on top
        gm = nt.nodes.new("ShaderNodeMapping")
        gm.inputs["Scale"].default_value = (60.0, 1400.0, 1400.0)
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Detail"].default_value = 4.0
        bp = nt.nodes.new("ShaderNodeBump")
        bp.inputs["Strength"].default_value = grain
        bp.inputs["Distance"].default_value = 0.0002
        nt.links.new(tc.outputs["Object"], gm.inputs["Vector"])
        nt.links.new(gm.outputs["Vector"], noise.inputs["Vector"])
        nt.links.new(noise.outputs["Fac"], bp.inputs["Height"])
        nt.links.new(nm.outputs["Normal"], bp.inputs["Normal"])
        nt.links.new(bp.outputs["Normal"], bsdf.inputs["Normal"])
    else:
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def paint(name, color, rough=0.48, coat=0.25, coat_rough=0.3, bump_scale=900.0, bump=0.06):
    """Painted toy: a slightly uneven gloss paint under a clear coat."""
    m, nt, bsdf = new_mat(name)
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Coat Weight"].default_value = coat
    bsdf.inputs["Coat Roughness"].default_value = coat_rough
    if bump > 0.0:
        tc = nt.nodes.new("ShaderNodeTexCoord")
        noise = nt.nodes.new("ShaderNodeTexNoise")
        noise.inputs["Scale"].default_value = bump_scale
        noise.inputs["Detail"].default_value = 3.0
        bp = nt.nodes.new("ShaderNodeBump")
        bp.inputs["Strength"].default_value = bump
        bp.inputs["Distance"].default_value = 0.0002
        nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
        nt.links.new(noise.outputs["Fac"], bp.inputs["Height"])
        nt.links.new(bp.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def paper(name, color=(0.86, 0.82, 0.72)):
    """The edge of a block of pages: off-white with fine horizontal layers."""
    m, nt, bsdf = new_mat(name)
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.85
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (2.0, 2.0, 5000.0)
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 1.0
    bp = nt.nodes.new("ShaderNodeBump")
    bp.inputs["Strength"].default_value = 0.5
    bp.inputs["Distance"].default_value = 0.0002
    nt.links.new(tc.outputs["Object"], mp.inputs["Vector"])
    nt.links.new(mp.outputs["Vector"], noise.inputs["Vector"])
    nt.links.new(noise.outputs["Fac"], bp.inputs["Height"])
    nt.links.new(bp.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def plain(name, color, rough=0.6, metallic=0.0, emission=None, strength=0.0):
    m, nt, bsdf = new_mat(name)
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    bsdf.inputs["Metallic"].default_value = metallic
    if emission is not None:
        bsdf.inputs["Emission Color"].default_value = (*emission, 1.0)
        bsdf.inputs["Emission Strength"].default_value = strength
    return m


# ---------------------------------------------------------------- shapes

def _finish(sc, name, bm, mat, loc, parent, bevel, segments, smooth_shade=True, rot=None):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if smooth_shade:
        me.polygons.foreach_set("use_smooth", [True] * len(me.polygons))
        me.update()
    ob = bpy.data.objects.new(name, me)
    sc.collection.objects.link(ob)
    ob.location = loc
    if rot is not None:
        ob.rotation_euler = rot
    if mat is not None:
        me.materials.append(mat)
    if parent is not None:
        ob.parent = parent
    if bevel and bevel > 0.0:
        mod = ob.modifiers.new("bevel", "BEVEL")
        mod.width = bevel
        mod.segments = segments
        mod.limit_method = "ANGLE"
        mod.angle_limit = math.radians(40.0)
        mod.harden_normals = True
    else:
        try:
            me.set_sharp_from_angle(angle=math.radians(40.0))
        except Exception:
            pass
    return ob


def box(sc, name, size, loc, mat=None, bevel=0.002, parent=None, segments=3, rot=None):
    """A box of the given size centred on loc. No razor-sharp corners."""
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector(size), verts=bm.verts)
    return _finish(sc, name, bm, mat, loc, parent, bevel, segments, rot=rot)


def box_between(sc, name, lo, hi, mat=None, bevel=0.002, parent=None):
    lo, hi = Vector(lo), Vector(hi)
    return box(sc, name, hi - lo, (lo + hi) / 2.0, mat, bevel, parent)


def cyl(sc, name, r, depth, loc, mat=None, axis="Z", segs=64, r2=None, bevel=0.0005,
        parent=None, rot=None, segments=3):
    """A cylinder or cone (r2 is the far-end radius) along the given axis."""
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segs,
                          radius1=r, radius2=r if r2 is None else r2, depth=depth)
    if axis == "X":
        bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(math.radians(90), 3, "Y"), verts=bm.verts)
    elif axis == "Y":
        bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=Matrix.Rotation(math.radians(-90), 3, "X"), verts=bm.verts)
    return _finish(sc, name, bm, mat, loc, parent, bevel, segments, rot=rot)


def ball(sc, name, r, loc, mat=None, scale=(1, 1, 1), segs=64, parent=None, rot=None):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=segs, v_segments=segs // 2, radius=r)
    bmesh.ops.scale(bm, vec=Vector(scale), verts=bm.verts)
    return _finish(sc, name, bm, mat, loc, parent, 0.0, 0, rot=rot)


def lathe(sc, name, profile, loc, mat=None, steps=72, parent=None):
    """Spin a profile of (radius, height) points around the Z axis."""
    bm = bmesh.new()
    verts = [bm.verts.new((r, 0.0, z)) for r, z in profile]
    edges = [bm.edges.new((a, b)) for a, b in zip(verts, verts[1:])]
    bmesh.ops.spin(bm, geom=verts + edges, cent=(0, 0, 0), axis=(0, 0, 1),
                   angle=math.tau, steps=steps, use_duplicate=False)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return _finish(sc, name, bm, mat, loc, parent, 0.0, 0)


def empty(sc, name, loc=(0, 0, 0), parent=None):
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_size = 0.01
    sc.collection.objects.link(ob)
    ob.location = loc
    if parent is not None:
        ob.parent = parent
    return ob


def key(ob, frame, *paths):
    for p in paths:
        ob.keyframe_insert(p, frame=frame)
