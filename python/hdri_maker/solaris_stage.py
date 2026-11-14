"""
Houdini 20+ Solaris (/stage) LOPs integration for HDRI Maker.

Creates and updates domelight::2.0 and optional hotspot companion lights
on /stage with proper Lat/Long 32-bit float EXR wiring for Karma and Redshift.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .hdri_generator import ExtractedAreaLightInfo
from .lighting_analyzer import HotspotLight, LightingProfile

try:
    import hou
    HAS_HOU = True
except ImportError:
    hou = None
    HAS_HOU = False


def _ensure_stage_context():
    """Return the /stage LOP network node."""
    if not HAS_HOU:
        raise RuntimeError("Houdini (hou) is not available in this Python session.")
    stage = hou.node("/stage")
    if stage is None:
        stage = hou.node("/").createNode("lopnet", "stage")
    return stage


def _resolve_valid_parent_lop(stage_net, parent_node):
    """Return `parent_node` if still alive in Houdini, else fallback to `/stage/hdri_maker_domelight`."""
    if parent_node is not None:
        try:
            parent_node.path()
            if parent_node.name() == "hdri_maker_lookdev":
                ins = parent_node.inputs()
                if ins and ins[0] is not None:
                    return ins[0]
            return parent_node
        except Exception:
            pass
    try:
        return stage_net.node("hdri_maker_domelight")
    except Exception:
        return None


def _find_tail_light_lop(stage_net, fallback_parent=None, exclude_node=None):
    """
    Find the last HDRI Maker light LOP in the `/stage` stream so that
    `hdri_maker_lookdev` is always wired at the very end after all lights.
    """
    tail = _resolve_valid_parent_lop(stage_net, fallback_parent)
    try:
        for prefix in ("hdri_maker_area_", "hdri_maker_key_"):
            for k in range(1, 65):
                n = stage_net.node(f"{prefix}{k}")
                if n is not None and n != exclude_node:
                    tail = n
        if tail is None or tail.name() == "hdri_maker_domelight":
            key_area = stage_net.node("hdri_maker_key_area")
            if key_area is not None and key_area != exclude_node:
                tail = key_area
    except Exception:
        pass

    visited = set()
    while tail is not None:
        try:
            t_path = tail.path()
        except Exception:
            break
        if t_path in visited:
            break
        visited.add(t_path)
        next_light = None
        try:
            for out_node in tail.outputs():
                if (
                    out_node is not None
                    and out_node != exclude_node
                    and out_node.name() != "hdri_maker_lookdev"
                    and out_node.name().startswith("hdri_maker_")
                ):
                    next_light = out_node
                    break
        except Exception:
            break
        if next_light is not None:
            tail = next_light
        else:
            break
    return tail


def _rewire_lookdev_to_tail(stage_net, tail_light_node, set_display_flag: bool = True) -> bool:
    """
    If `/stage/hdri_maker_lookdev` exists, ensure it is wired at the very end
    of the stream after `tail_light_node` (and any downstream light LOPs) and
    carries the display flag when `set_display_flag` is True.
    """
    try:
        lookdev = stage_net.node("hdri_maker_lookdev")
    except Exception:
        lookdev = None
    if lookdev is None:
        return False
    tail = _find_tail_light_lop(stage_net, fallback_parent=tail_light_node, exclude_node=lookdev)
    if tail is not None and tail != lookdev:
        try:
            lookdev.setInput(0, tail)
            lookdev.moveToGoodPosition()
        except Exception:
            pass
    if set_display_flag:
        try:
            lookdev.setDisplayFlag(True)
        except Exception:
            pass
    return True


def _set_first_parm(node, candidates: Tuple[str, ...], value) -> bool:
    """Set the first matching parameter name found on a Solaris LOP node."""
    for name in candidates:
        parm = node.parm(name)
        if parm is not None:
            parm.set(value)
            return True
    return False


def _set_first_parm_tuple(node, candidates: Tuple[str, ...], values: Tuple[float, ...]) -> bool:
    """Set the first matching tuple parameter found on a Solaris LOP node."""
    for name in candidates:
        pt = node.parmTuple(name)
        if pt is not None:
            pt.set(values)
            return True
    return False


def _set_parm_control(node, control_candidates: Tuple[str, ...], mode: str = "set") -> None:
    """Ensure Solaris LOP parameter action/control is set to 'set' (not 'none')."""
    for name in control_candidates:
        parm = node.parm(name)
        if parm is not None:
            try:
                parm.set(mode)
            except Exception:
                pass
            return


def _create_lop(stage, type_candidates: Tuple[str, ...], name: str):
    """Find existing LOP node by name or create using the first valid node type."""
    existing = stage.node(name)
    if existing is not None:
        return existing
    for ntype in type_candidates:
        try:
            return stage.createNode(ntype, name)
        except hou.OperationFailed:
            continue
    raise RuntimeError(f"Could not create any of LOP types {type_candidates} in {stage.path()}")


def create_or_update_domelight(
    hdri_exr_path: str,
    profile: Optional[LightingProfile] = None,
    node_name: str = "hdri_maker_domelight",
    prim_path: str = "/lights/hdri_maker_dome",
    exposure: float = 0.0,
    intensity: float = 1.0,
    rotation_y_deg: float = 0.0,
    enable_color_temp: bool = False,
    set_display_flag: bool = True,
):
    """
    Create or update a Solaris Dome Light LOP on /stage pointing to the 32-bit EXR.

    Compatible with Houdini 20.0, 20.5, and 21.0+ (Karma & Redshift Solaris).
    """
    stage = _ensure_stage_context()
    dome = _create_lop(stage, ("domelight::2.0", "domelight"), node_name)

    # Set USD primitive path
    _set_first_parm(dome, ("primpath",), prim_path)

    # Set texture file & latlong format
    _set_parm_control(dome, ("xn__inputstexturefile_control_shbh", "xn__inputstexturefile_control_r3a"))
    _set_first_parm(
        dome,
        ("xn__inputstexturefile_r3ah", "xn__inputstexturefile_r3a", "inputs:texture:file", "texturefile"),
        hdri_exr_path,
    )

    _set_parm_control(dome, ("xn__inputstextureformat_control_1kbh", "xn__inputstextureformat_control_t3a"))
    _set_first_parm(
        dome,
        ("xn__inputstextureformat_06ah", "xn__inputstextureformat_t3a", "inputs:texture:format", "textureformat"),
        "latlong",
    )

    # Intensity & Exposure
    _set_parm_control(dome, ("xn__inputsintensity_control_jeb", "xn__inputsintensity_control_i5a"))
    _set_first_parm(
        dome,
        ("xn__inputsintensity_i0a", "xn__inputsintensity_i5a", "inputs:intensity", "intensity"),
        float(intensity),
    )

    _set_parm_control(dome, ("xn__inputsexposure_control_wcb", "xn__inputsexposure_control_e5a"))
    _set_first_parm(
        dome,
        ("xn__inputsexposure_vya", "xn__inputsexposure_e5a", "inputs:exposure", "exposure"),
        float(exposure),
    )

    # Azimuth orientation
    _set_first_parm(dome, ("ry",), float(rotation_y_deg))

    # Optional Color Temperature metadata on dome light
    if profile is not None and enable_color_temp:
        _set_parm_control(dome, ("xn__inputsenableColorTemperature_control_pzb", "xn__inputsenableColorTemperature_control_r6a"))
        _set_first_parm(
            dome,
            ("xn__inputsenableColorTemperature_omb", "xn__inputsenableColorTemperature_r6a", "enableColorTemperature"),
            1,
        )
        _set_parm_control(dome, ("xn__inputscolorTemperature_control_xpb", "xn__inputscolorTemperature_control_v6a"))
        _set_first_parm(
            dome,
            ("xn__inputscolorTemperature_wcb", "xn__inputscolorTemperature_v6a", "colorTemperature"),
            float(profile.overall_cct_kelvin),
        )

    dome.moveToGoodPosition()
    try:
        first_area = stage.node("hdri_maker_area_1") or stage.node("hdri_maker_key_area") or stage.node("hdri_maker_key_1")
        if first_area is not None and not first_area.inputs():
            first_area.setInput(0, dome)
    except Exception:
        pass
    if not _rewire_lookdev_to_tail(stage, dome, set_display_flag=set_display_flag):
        if set_display_flag:
            try:
                dome.setDisplayFlag(True)
            except Exception:
                pass

    return dome


def create_or_update_area_light_from_hotspot(
    info: ExtractedAreaLightInfo,
    light_tex_exr_path: str,
    parent_dome=None,
    node_name: str = "hdri_maker_key_area",
    prim_path: str = "/lights/hdri_maker_key_area",
    distance: float = 20.0,
    intensity: float = 1.0,
    exposure: float = 0.0,
    set_display_flag: bool = True,
):
    """
    Create or update ONE Solaris Area Light (light::2.0 rect light) positioned
    on the sphere at the extracted hotspot's exact (azimuth, elevation), aimed
    at the origin, sized to match the selected angular span, and textured with
    the baked 32-bit float EXR light texture.
    """
    stage = _ensure_stage_context()
    light = _create_lop(stage, ("light::2.0", "light"), node_name)

    _set_first_parm(light, ("primpath",), prim_path)
    # Always create UsdLuxRectLight so USD/Karma/Redshift support inputs:texture:file;
    # when info.shape == "circle", the baked EXR texture carries the soft disk mask.
    _set_first_parm(light, ("lighttype",), "UsdLuxRectLight")

    # Spherical position on dome radius `distance`
    az_rad = math.radians(info.azimuth_deg)
    el_rad = math.radians(info.elevation_deg)
    tx = distance * math.cos(el_rad) * math.sin(az_rad)
    ty = distance * math.sin(el_rad)
    tz = distance * math.cos(el_rad) * math.cos(az_rad)

    _set_first_parm(light, ("tx",), float(tx))
    _set_first_parm(light, ("ty",), float(ty))
    _set_first_parm(light, ("tz",), float(tz))

    # Aim -Z emission axis back toward origin (0, 0, 0)
    _set_first_parm(light, ("rx",), float(-info.elevation_deg))
    _set_first_parm(light, ("ry",), float(info.azimuth_deg))
    _set_first_parm(light, ("rz",), 0.0)

    # Physical rect size matching angular span at `distance`
    w_units = max(0.5, 2.0 * distance * math.tan(math.radians(min(150.0, info.angular_width_deg) * 0.5)))
    h_units = max(0.5, 2.0 * distance * math.tan(math.radians(min(150.0, info.angular_height_deg) * 0.5)))

    _set_parm_control(light, ("xn__inputswidth_control_06a",))
    _set_first_parm(light, ("xn__inputswidth_zta", "inputs:width", "width"), float(w_units))

    _set_parm_control(light, ("xn__inputsheight_control_n8a",))
    _set_first_parm(light, ("xn__inputsheight_mva", "inputs:height", "height"), float(h_units))

    # Wire baked 32-bit float EXR light texture
    _set_parm_control(light, ("xn__inputstexturefile_control_shbh", "xn__inputstexturefile_control_r3a"))
    _set_first_parm(
        light,
        ("xn__inputstexturefile_r3ah", "xn__inputstexturefile_r3a", "inputs:texture:file", "texturefile"),
        light_tex_exr_path,
    )
    _set_parm_control(light, ("xn__redshiftlightuseAlpha_control_dobf",))
    _set_first_parm(light, ("xn__redshiftlightuseAlpha_cbbf",), 1)
    _set_parm_control(light, ("xn__inputskarmalightrenderlightgeo_control_y2bff",))
    _set_first_parm(light, ("xn__inputskarmalightrenderlightgeo_xpbff",), 0)

    # Intensity & Exposure (32-bit EXR carries full HDR energy, so intensity=1.0 default)
    _set_parm_control(light, ("xn__inputsintensity_control_jeb", "xn__inputsintensity_control_i5a"))
    _set_first_parm(light, ("xn__inputsintensity_i0a", "xn__inputsintensity_i5a", "inputs:intensity", "intensity"), float(intensity))

    _set_parm_control(light, ("xn__inputsexposure_control_wcb", "xn__inputsexposure_control_e5a"))
    _set_first_parm(light, ("xn__inputsexposure_vya", "xn__inputsexposure_e5a", "inputs:exposure", "exposure"), float(exposure))

    # Color (1, 1, 1) so baked EXR texture provides exact color & temperature
    _set_parm_control(light, ("xn__inputscolor_control_06a",))
    _set_first_parm_tuple(light, ("xn__inputscolor_zta", "inputs:color", "color"), (1.0, 1.0, 1.0))

    valid_parent = _resolve_valid_parent_lop(stage, parent_dome)
    if valid_parent is not None and valid_parent != light:
        try:
            light.setInput(0, valid_parent)
        except Exception:
            pass

    light.moveToGoodPosition()
    if not _rewire_lookdev_to_tail(stage, light, set_display_flag=set_display_flag):
        if set_display_flag:
            try:
                light.setDisplayFlag(True)
            except Exception:
                pass
    return light


def create_hotspot_light_rig(
    hotspots: List[HotspotLight],
    parent_dome=None,
    rig_prefix: str = "hdri_maker_key",
    intensity_scale: float = 1.0,
):
    """
    Create Solaris distant/key LOP lights matching the extracted plate hotspots.

    Chains the created lights cleanly after `parent_dome` (if provided) on /stage.
    """
    if not hotspots:
        return []

    stage = _ensure_stage_context()
    created = []
    prev_node = _resolve_valid_parent_lop(stage, parent_dome)

    for i, hs in enumerate(hotspots):
        name = f"{rig_prefix}_{i + 1}"
        light = _create_lop(stage, ("distantlight::2.0", "light::2.0", "distantlight"), name)
        _set_first_parm(light, ("primpath",), f"/lights/{name}")

        # Point light from spherical (azimuth, elevation) toward origin
        _set_first_parm(light, ("rx",), float(-hs.elevation_deg))
        _set_first_parm(light, ("ry",), float(hs.azimuth_deg))

        _set_parm_control(light, ("xn__inputsintensity_control_i5a",))
        _set_first_parm(
            light,
            ("xn__inputsintensity_i5a", "inputs:intensity", "intensity"),
            float(hs.mean_intensity * intensity_scale),
        )

        _set_parm_control(light, ("xn__inputscolor_control_i4a",))
        _set_first_parm_tuple(
            light,
            ("xn__inputscolor_i4a", "inputs:color", "color"),
            hs.color_rgb,
        )

        _set_parm_control(light, ("xn__inputsangle_control_e4a",))
        _set_first_parm(
            light,
            ("xn__inputsangle_e4a", "inputs:angle", "angle"),
            float(max(0.5, hs.radius_deg * 0.5)),
        )

        if prev_node is not None:
            light.setInput(0, prev_node)
        light.moveToGoodPosition()
        prev_node = light
        created.append(light)

    if prev_node is not None:
        if not _rewire_lookdev_to_tail(stage, prev_node, set_display_flag=True):
            try:
                prev_node.setDisplayFlag(True)
            except Exception:
                pass

    return created


def create_or_update_lookdev_and_ground(
    ground_tex_exr_path: str,
    ground_opacity_exr_path: str,
    parent_node=None,
    colorspace: str = "ACEScg",
    ground_radius: float = 10.0,
    tripod_height: float = 1.5,
    ground_emission: float = 0.5,
    ground_shader_mode: str = "Emissive + Diffuse",
    node_name: str = "hdri_maker_lookdev",
    set_display_flag: bool = True,
):
    """
    Create or update a Solaris pythonscript LOP on /stage that builds:
      1. Lookdev Tripod (/lookdev/tripod) with Mirror Chrome + 18% Gray ref balls
         and a 24-patch Macbeth ColorChecker chart (ACEScg or Linear Rec.709).
      2. Circular Ground Disc (/lookdev/ground_disc) mapped with the top-down
         planar HDRI ground projection EXR + radial soft-blending transparency
         supporting Karma MaterialX, Arnold Standard Surface, Redshift Standard Material,
         and UsdPreviewSurface.
    """
    stage = _ensure_stage_context()
    lop = _create_lop(stage, ("pythonscript",), node_name)
    valid_parent = _find_tail_light_lop(stage, fallback_parent=parent_node, exclude_node=lop)
    if valid_parent is not None and valid_parent != lop:
        try:
            lop.setInput(0, valid_parent)
        except Exception:
            pass

    ground_tex_norm = ground_tex_exr_path.replace("\\", "/")
    ground_op_norm = ground_opacity_exr_path.replace("\\", "/")
    use_acescg = "aces" in str(colorspace).lower()

    mode_norm = str(ground_shader_mode or "Emissive + Diffuse").strip().lower()
    is_shadow_catcher = ("shadow" in mode_norm) or ("matte" in mode_norm) or ("catcher" in mode_norm)
    use_diffuse = is_shadow_catcher or (("no diffuse" not in mode_norm) and ("emissive only" not in mode_norm))
    use_emissive = (not is_shadow_catcher) and ("no emission" not in mode_norm) and ("diffuse only" not in mode_norm)

    code = f'''import math
import hou
from pxr import UsdGeom, UsdShade, Gf, Sdf

stage = hou.pwd().editableStage()
ground_tex = {ground_tex_norm!r}
ground_op_tex = {ground_op_norm!r}
use_acescg = {use_acescg!r}
ground_radius = {float(ground_radius)}
tripod_height = {float(tripod_height)}
ground_emission = {max(0.0, float(ground_emission))}
use_diffuse = {bool(use_diffuse)}
use_emissive = {bool(use_emissive)}
is_shadow_catcher = {bool(is_shadow_catcher)}

def _ensure_scope(path):
    if not stage.GetPrimAtPath(path).IsValid():
        UsdGeom.Scope.Define(stage, path)

def _srgb_to_lin_rec709(c):
    out = []
    for v in c:
        v = max(0.0, float(v))
        out.append(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4)
    return tuple(out)

def _rec709_to_acescg(c):
    r, g, b = c
    return (
        0.613097 * r + 0.339523 * g + 0.047379 * b,
        0.070194 * r + 0.916354 * g + 0.013452 * b,
        0.020616 * r + 0.109570 * g + 0.869815 * b,
    )

def _make_pbr_mtl(mtl_path, color, metallic=0.0, roughness=0.5):
    mat = UsdShade.Material.Define(stage, mtl_path)
    c3 = Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))

    # 1. UsdPreviewSurface (GL viewport)
    ups = UsdShade.Shader.Define(stage, f"{{mtl_path}}/PreviewSurface")
    ups.CreateIdAttr("UsdPreviewSurface")
    ups.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(c3)
    ups.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    ups.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    mat.CreateSurfaceOutput().ConnectToSource(ups.ConnectableAPI(), "surface")

    # 2. Karma MaterialX
    km = UsdShade.Shader.Define(stage, f"{{mtl_path}}/KarmaMtlX")
    km.CreateIdAttr("ND_standard_surface_surfaceshader")
    km.CreateInput("base", Sdf.ValueTypeNames.Float).Set(1.0)
    km.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).Set(c3)
    km.CreateInput("metalness", Sdf.ValueTypeNames.Float).Set(float(metallic))
    km.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(1.0 if metallic > 0.5 else max(0.0, 1.0 - float(roughness)))
    km.CreateInput("specular_roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    mat.CreateSurfaceOutput("karma").ConnectToSource(km.ConnectableAPI(), "out")

    # 3. Arnold Standard Surface
    ai = UsdShade.Shader.Define(stage, f"{{mtl_path}}/ArnoldStandardSurface")
    ai.CreateIdAttr("arnold:standard_surface")
    ai.CreateInput("base", Sdf.ValueTypeNames.Float).Set(1.0)
    ai.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).Set(c3)
    ai.CreateInput("metalness", Sdf.ValueTypeNames.Float).Set(float(metallic))
    ai.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(1.0 if metallic > 0.5 else max(0.0, 1.0 - float(roughness)))
    ai.CreateInput("specular_roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    mat.CreateSurfaceOutput("arnold").ConnectToSource(ai.ConnectableAPI(), "shader")

    # 4. Redshift Standard Material
    rsm = UsdShade.Shader.Define(stage, f"{{mtl_path}}/RedshiftMaterial")
    rsm.CreateIdAttr("redshift_usd_material")
    rstd = UsdShade.Shader.Define(stage, f"{{mtl_path}}/RedshiftStd")
    rstd.CreateIdAttr("redshift::StandardMaterial")
    rstd.CreateInput("base_color_weight", Sdf.ValueTypeNames.Float).Set(1.0)
    rstd.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).Set(c3)
    rstd.CreateInput("metalness", Sdf.ValueTypeNames.Float).Set(float(metallic))
    rstd.CreateInput("refl_weight", Sdf.ValueTypeNames.Float).Set(1.0 if metallic > 0.5 else max(0.0, 1.0 - float(roughness)))
    rstd.CreateInput("refl_roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    rstd.CreateOutput("outColor", Sdf.ValueTypeNames.Color3f)
    rsm.CreateInput("Surface", Sdf.ValueTypeNames.Token).ConnectToSource(rstd.ConnectableAPI(), "outColor")
    rsm.CreateOutput("Shader", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput("Redshift").ConnectToSource(rsm.ConnectableAPI(), "Shader")
    return mat

def _build_smooth_sphere(path, radius, rings=48, sectors=96):
    mesh = UsdGeom.Mesh.Define(stage, path)
    pts, nrms, uvs, counts, indices = [], [], [], [], []
    for r in range(rings + 1):
        phi = math.pi * float(r) / float(rings)
        y, ny, sp = radius * math.cos(phi), math.cos(phi), math.sin(phi)
        v = 1.0 - (float(r) / float(rings))
        for s in range(sectors + 1):
            th = 2.0 * math.pi * float(s) / float(sectors)
            nx, nz = sp * math.cos(th), sp * math.sin(th)
            pts.append(Gf.Vec3f(radius * nx, y, radius * nz))
            nrms.append(Gf.Vec3f(nx, ny, nz))
            uvs.append(Gf.Vec2f(float(s) / float(sectors), v))
    for r in range(rings):
        for s in range(sectors):
            p0 = r * (sectors + 1) + s
            p1 = (r + 1) * (sectors + 1) + s
            counts.append(4)
            indices.extend([p0, p1, p1 + 1, p0 + 1])
    mesh.GetPointsAttr().Set(pts)
    mesh.GetFaceVertexCountsAttr().Set(counts)
    mesh.GetFaceVertexIndicesAttr().Set(indices)
    mesh.GetNormalsAttr().Set(nrms)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(uvs)
    return mesh

root = "/lookdev"
_ensure_scope(root)
_ensure_scope(f"{{root}}/materials")
_ensure_scope(f"{{root}}/tripod")
_ensure_scope(f"{{root}}/tripod/stand")
_ensure_scope(f"{{root}}/tripod/spheres")
_ensure_scope(f"{{root}}/tripod/macbeth")

# Materials
stand_mtl = _make_pbr_mtl(f"{{root}}/materials/tripod_stand_mtl", (0.08, 0.08, 0.09), metallic=0.85, roughness=0.35)
chrome_mtl = _make_pbr_mtl(f"{{root}}/materials/chrome_mtl", (0.95, 0.95, 0.95), metallic=1.0, roughness=0.001)
grey_mtl = _make_pbr_mtl(f"{{root}}/materials/grey_18_mtl", (0.18, 0.18, 0.18), metallic=0.0, roughness=0.6)

# 1. Tripod Stand (Mast + 3 Legs + Crossbar)
sphere_r = 0.14
spacing = 0.42
crossbar_y = tripod_height
ball_y = crossbar_y + sphere_r

mast = UsdGeom.Cylinder.Define(stage, f"{{root}}/tripod/stand/mast")
mast.GetRadiusAttr().Set(0.018)
mast.GetHeightAttr().Set(crossbar_y)
mast.GetAxisAttr().Set("Y")
xf_m = UsdGeom.Xformable(mast.GetPrim())
xf_m.ClearXformOpOrder()
xf_m.AddTranslateOp().Set(Gf.Vec3d(0.0, crossbar_y * 0.5, 0.0))
UsdShade.MaterialBindingAPI.Apply(mast.GetPrim()).Bind(stand_mtl)

collar_h = min(0.35, crossbar_y * 0.4)
foot_spread = 0.42
for i in range(3):
    ang = i * (2.0 * math.pi / 3.0)
    lx, lz = math.sin(ang) * foot_spread, math.cos(ang) * foot_spread
    leg_len = math.sqrt(lx * lx + lz * lz + collar_h * collar_h)
    leg = UsdGeom.Cylinder.Define(stage, f"{{root}}/tripod/stand/leg_{{i+1}}")
    leg.GetRadiusAttr().Set(0.012)
    leg.GetHeightAttr().Set(leg_len)
    leg.GetAxisAttr().Set("Y")
    xf_l = UsdGeom.Xformable(leg.GetPrim())
    xf_l.ClearXformOpOrder()
    xf_l.AddTranslateOp().Set(Gf.Vec3d(lx * 0.5, collar_h * 0.5, lz * 0.5))
    xf_l.AddRotateYOp().Set(math.degrees(math.atan2(lx, lz)))
    xf_l.AddRotateXOp().Set(math.degrees(math.atan2(math.sqrt(lx * lx + lz * lz), collar_h)))
    UsdShade.MaterialBindingAPI.Apply(leg.GetPrim()).Bind(stand_mtl)

cbar = UsdGeom.Cylinder.Define(stage, f"{{root}}/tripod/stand/crossbar")
cbar.GetRadiusAttr().Set(0.012)
cbar.GetHeightAttr().Set(spacing * 2.0)
cbar.GetAxisAttr().Set("X")
xf_c = UsdGeom.Xformable(cbar.GetPrim())
xf_c.ClearXformOpOrder()
xf_c.AddTranslateOp().Set(Gf.Vec3d(0.0, crossbar_y, 0.0))
UsdShade.MaterialBindingAPI.Apply(cbar.GetPrim()).Bind(stand_mtl)

# 2. Chrome & 18% Gray Reference Spheres
chrome = _build_smooth_sphere(f"{{root}}/tripod/spheres/chrome_sphere", sphere_r)
xf_ch = UsdGeom.Xformable(chrome.GetPrim())
xf_ch.ClearXformOpOrder()
xf_ch.AddTranslateOp().Set(Gf.Vec3d(-spacing * 0.5, ball_y, 0.0))
UsdShade.MaterialBindingAPI.Apply(chrome.GetPrim()).Bind(chrome_mtl)

grey = _build_smooth_sphere(f"{{root}}/tripod/spheres/grey_sphere", sphere_r)
xf_gr = UsdGeom.Xformable(grey.GetPrim())
xf_gr.ClearXformOpOrder()
xf_gr.AddTranslateOp().Set(Gf.Vec3d(spacing * 0.5, ball_y, 0.0))
UsdShade.MaterialBindingAPI.Apply(grey.GetPrim()).Bind(grey_mtl)

# 3. Macbeth ColorChecker Chart (ACEScg or Linear Rec.709)
macbeth_srgb = [
    (0.400, 0.200, 0.161), (0.690, 0.451, 0.380), (0.322, 0.400, 0.537), (0.294, 0.376, 0.180),
    (0.459, 0.388, 0.596), (0.369, 0.647, 0.588), (0.737, 0.420, 0.102), (0.227, 0.271, 0.569),
    (0.671, 0.259, 0.259), (0.286, 0.169, 0.318), (0.537, 0.667, 0.133), (0.788, 0.576, 0.102),
    (0.133, 0.176, 0.475), (0.294, 0.529, 0.192), (0.553, 0.149, 0.137), (0.843, 0.749, 0.102),
    (0.596, 0.259, 0.471), (0.090, 0.400, 0.549), (0.871, 0.871, 0.871), (0.588, 0.588, 0.588),
    (0.365, 0.365, 0.365), (0.200, 0.200, 0.200), (0.090, 0.090, 0.090), (0.031, 0.031, 0.031),
]
chart_xf = UsdGeom.Xform.Define(stage, f"{{root}}/tripod/macbeth/chart")
xf_cd = UsdGeom.Xformable(chart_xf.GetPrim())
xf_cd.ClearXformOpOrder()
xf_cd.AddTranslateOp().Set(Gf.Vec3d(0.0, crossbar_y - 0.16, 0.03))
xf_cd.AddRotateXOp().Set(-8.0)

patch_sz, gap, cols, rows = 0.032, 0.004, 6, 4
tot_w, tot_h = cols * (patch_sz + gap), rows * (patch_sz + gap)
backing = UsdGeom.Mesh.Define(stage, f"{{root}}/tripod/macbeth/chart/backing")
backing.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
hw, hh = tot_w * 0.55, tot_h * 0.55
backing.GetPointsAttr().Set([Gf.Vec3f(-hw, -hh, -0.002), Gf.Vec3f(hw, -hh, -0.002), Gf.Vec3f(hw, hh, -0.002), Gf.Vec3f(-hw, hh, -0.002)])
backing.GetFaceVertexCountsAttr().Set([4])
backing.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
UsdShade.MaterialBindingAPI.Apply(backing.GetPrim()).Bind(stand_mtl)

for idx, cs in enumerate(macbeth_srgb):
    r_i, c_i = idx // cols, idx % cols
    px = c_i * (patch_sz + gap) - tot_w * 0.5
    py = (rows - 1 - r_i) * (patch_sz + gap) - tot_h * 0.5
    pm = UsdGeom.Mesh.Define(stage, f"{{root}}/tripod/macbeth/chart/patch_{{idx:02d}}")
    pm.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    pm.GetPointsAttr().Set([
        Gf.Vec3f(px, py, 0.001), Gf.Vec3f(px + patch_sz, py, 0.001),
        Gf.Vec3f(px + patch_sz, py + patch_sz, 0.001), Gf.Vec3f(px, py + patch_sz, 0.001),
    ])
    pm.GetFaceVertexCountsAttr().Set([4])
    pm.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    lin = _srgb_to_lin_rec709(cs)
    col_out = _rec709_to_acescg(lin) if use_acescg else lin
    mb_mtl = _make_pbr_mtl(f"{{root}}/materials/macbeth_{{idx:02d}}", col_out, metallic=0.0, roughness=0.85)
    UsdShade.MaterialBindingAPI.Apply(pm.GetPrim()).Bind(mb_mtl)

# 4. Circular Ground Disc with Top-Down Planar HDRI Projection + Soft Blending Transparency
disc = UsdGeom.Mesh.Define(stage, f"{{root}}/ground_disc")
disc.GetSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
g_rings, g_sectors = 32, 96
g_pts, g_nrms, g_uvs, g_opacities, g_counts, g_indices = [], [], [], [], [], []
for r in range(g_rings + 1):
    rad = ground_radius * (float(r) / float(g_rings))
    rn = float(r) / float(g_rings)
    t_f = max(0.0, min(1.0, (rn - 0.55) / 0.45))
    alpha = 1.0 - (t_f * t_f * (3.0 - 2.0 * t_f))
    for s in range(g_sectors + 1):
        th = 2.0 * math.pi * float(s) / float(g_sectors)
        x, z = rad * math.cos(th), rad * math.sin(th)
        u = (x + ground_radius) / (2.0 * ground_radius)
        v = 1.0 - ((z + ground_radius) / (2.0 * ground_radius))
        g_pts.append(Gf.Vec3f(x, 0.0, z))
        g_nrms.append(Gf.Vec3f(0.0, 1.0, 0.0))
        g_uvs.append(Gf.Vec2f(u, v))
        g_opacities.append(float(alpha))
for r in range(g_rings):
    for s in range(g_sectors):
        p0 = r * (g_sectors + 1) + s
        p1 = (r + 1) * (g_sectors + 1) + s
        g_counts.append(4)
        g_indices.extend([p0, p0 + 1, p1 + 1, p1])
disc.GetPointsAttr().Set(g_pts)
disc.GetFaceVertexCountsAttr().Set(g_counts)
disc.GetFaceVertexIndicesAttr().Set(g_indices)
disc.GetNormalsAttr().Set(g_nrms)
disc.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
UsdGeom.PrimvarsAPI(disc).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(g_uvs)
UsdGeom.PrimvarsAPI(disc).CreatePrimvar("displayOpacity", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set(g_opacities)

# Multi-renderer Ground Projection Material (UsdPreviewSurface + Karma MtlX + Arnold Standard Surface)
g_mtl_path = f"{{root}}/materials/ground_proj_mtl"
g_mat = UsdShade.Material.Define(stage, g_mtl_path)

# UsdPreviewSurface
ups = UsdShade.Shader.Define(stage, f"{{g_mtl_path}}/PreviewSurface")
ups.CreateIdAttr("UsdPreviewSurface")
ups.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
ups.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
pvr = UsdShade.Shader.Define(stage, f"{{g_mtl_path}}/PreviewUV")
pvr.CreateIdAttr("UsdPrimvarReader_float2")
pvr.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
pvr.CreateOutput("result", Sdf.ValueTypeNames.Float2)
utex = UsdShade.Shader.Define(stage, f"{{g_mtl_path}}/PreviewColorTex")
utex.CreateIdAttr("UsdUVTexture")
utex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(ground_tex))
utex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
utex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
utex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
utex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(pvr.ConnectableAPI(), "result")
utex.CreateOutput("rgb", Sdf.ValueTypeNames.Color3f)
if use_diffuse:
    ups.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(utex.ConnectableAPI(), "rgb")
else:
    ups.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
if use_emissive and ground_emission > 0.0:
    uem = UsdShade.Shader.Define(stage, f"{{g_mtl_path}}/PreviewEmissiveTex")
    uem.CreateIdAttr("UsdUVTexture")
    uem.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(ground_tex))
    uem.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    uem.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    uem.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    uem.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(ground_emission, ground_emission, ground_emission, 1.0))
    uem.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(pvr.ConnectableAPI(), "result")
    uem.CreateOutput("rgb", Sdf.ValueTypeNames.Color3f)
    ups.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(uem.ConnectableAPI(), "rgb")
else:
    ups.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
uop = UsdShade.Shader.Define(stage, f"{{g_mtl_path}}/PreviewOpacityTex")
uop.CreateIdAttr("UsdUVTexture")
uop.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(ground_op_tex))
uop.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
uop.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
uop.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
uop.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(pvr.ConnectableAPI(), "result")
uop.CreateOutput("r", Sdf.ValueTypeNames.Float)
ups.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(uop.ConnectableAPI(), "r")
ups.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.0)
g_mat.CreateSurfaceOutput().ConnectToSource(ups.ConnectableAPI(), "surface")

# Karma MaterialX
km = UsdShade.Shader.Define(stage, f"{{g_mtl_path}}/KarmaMtlX")
km.CreateIdAttr("ND_standard_surface_surfaceshader")
km.CreateInput("base", Sdf.ValueTypeNames.Float).Set(1.0 if use_diffuse else 0.0)
km.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(0.0)
km.CreateInput("specular_roughness", Sdf.ValueTypeNames.Float).Set(1.0)
kuv = UsdShade.Shader.Define(stage, f"{{g_mtl_path}}/KarmaUV")
kuv.CreateIdAttr("ND_texcoord_vector2")
kuv.CreateInput("index", Sdf.ValueTypeNames.Int).Set(0)
kuv.CreateOutput("out", Sdf.ValueTypeNames.Float2)
ktex = UsdShade.Shader.Define(stage, f"{{g_mtl_path}}/KarmaColorTex")
ktex.CreateIdAttr("ND_image_color3")
ktex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(ground_tex))
ktex.CreateInput("uaddressmode", Sdf.ValueTypeNames.String).Set("clamp")
ktex.CreateInput("vaddressmode", Sdf.ValueTypeNames.String).Set("clamp")
ktex.CreateInput("texcoord", Sdf.ValueTypeNames.Float2).ConnectToSource(kuv.ConnectableAPI(), "out")
ktex.CreateOutput("out", Sdf.ValueTypeNames.Color3f)
kop = UsdShade.Shader.Define(stage, f"{{g_mtl_path}}/KarmaOpacityTex")
kop.CreateIdAttr("ND_image_color3")
kop.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(ground_op_tex))
kop.CreateInput("uaddressmode", Sdf.ValueTypeNames.String).Set("clamp")
kop.CreateInput("vaddressmode", Sdf.ValueTypeNames.String).Set("clamp")
kop.CreateInput("texcoord", Sdf.ValueTypeNames.Float2).ConnectToSource(kuv.ConnectableAPI(), "out")
kop.CreateOutput("out", Sdf.ValueTypeNames.Color3f)
km.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(ktex.ConnectableAPI(), "out")
km.CreateInput("emission", Sdf.ValueTypeNames.Float).Set(ground_emission if use_emissive else 0.0)
km.CreateInput("emission_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(ktex.ConnectableAPI(), "out")
km.CreateInput("opacity", Sdf.ValueTypeNames.Color3f).ConnectToSource(kop.ConnectableAPI(), "out")
g_mat.CreateSurfaceOutput("karma").ConnectToSource(km.ConnectableAPI(), "out")
g_mat.CreateSurfaceOutput("mtlx").ConnectToSource(km.ConnectableAPI(), "out")

# Arnold Standard Surface (or Arnold Shadow Matte when Shadow Catcher mode is active)
atex = UsdShade.Shader.Define(stage, f"{{g_mtl_path}}/ArnoldColorTex")
atex.CreateIdAttr("arnold:image")
atex.CreateInput("filename", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(ground_tex))
atex.CreateInput("color_space", Sdf.ValueTypeNames.String).Set("raw")
atex.CreateInput("swrap", Sdf.ValueTypeNames.String).Set("clamp")
atex.CreateInput("twrap", Sdf.ValueTypeNames.String).Set("clamp")
atex.CreateInput("uvset", Sdf.ValueTypeNames.String).Set("st")
atex.CreateOutput("rgb", Sdf.ValueTypeNames.Color3f)
aop = UsdShade.Shader.Define(stage, f"{{g_mtl_path}}/ArnoldOpacityTex")
aop.CreateIdAttr("arnold:image")
aop.CreateInput("filename", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(ground_op_tex))
aop.CreateInput("color_space", Sdf.ValueTypeNames.String).Set("raw")
aop.CreateInput("swrap", Sdf.ValueTypeNames.String).Set("clamp")
aop.CreateInput("twrap", Sdf.ValueTypeNames.String).Set("clamp")
aop.CreateInput("uvset", Sdf.ValueTypeNames.String).Set("st")
aop.CreateOutput("rgb", Sdf.ValueTypeNames.Color3f)
if is_shadow_catcher:
    ai = UsdShade.Shader.Define(stage, f"{{g_mtl_path}}/ArnoldShadowMatte")
    ai.CreateIdAttr("arnold:shadow_matte")
    ai.CreateInput("background_type", Sdf.ValueTypeNames.Int).Set(0)
    ai.CreateInput("background_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    ai.CreateInput("shadow_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    ai.CreateInput("shadow_opacity", Sdf.ValueTypeNames.Float).Set(1.0)
    ai.CreateInput("diffuse_intensity", Sdf.ValueTypeNames.Float).Set(0.0)
    ai.CreateInput("indirect_diffuse_enable", Sdf.ValueTypeNames.Bool).Set(True)
    ai.CreateInput("opacity", Sdf.ValueTypeNames.Color3f).ConnectToSource(aop.ConnectableAPI(), "rgb")
    g_mat.CreateSurfaceOutput("arnold").ConnectToSource(ai.ConnectableAPI(), "shader")
else:
    ai = UsdShade.Shader.Define(stage, f"{{g_mtl_path}}/ArnoldStandardSurface")
    ai.CreateIdAttr("arnold:standard_surface")
    ai.CreateInput("base", Sdf.ValueTypeNames.Float).Set(1.0 if use_diffuse else 0.0)
    ai.CreateInput("specular", Sdf.ValueTypeNames.Float).Set(0.0)
    ai.CreateInput("specular_roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    ai.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(atex.ConnectableAPI(), "rgb")
    ai.CreateInput("emission", Sdf.ValueTypeNames.Float).Set(ground_emission if use_emissive else 0.0)
    ai.CreateInput("emission_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(atex.ConnectableAPI(), "rgb")
    ai.CreateInput("opacity", Sdf.ValueTypeNames.Color3f).ConnectToSource(aop.ConnectableAPI(), "rgb")
    g_mat.CreateSurfaceOutput("arnold").ConnectToSource(ai.ConnectableAPI(), "shader")

# Redshift Standard Material
rsm = UsdShade.Shader.Define(stage, f"{{g_mtl_path}}/RedshiftMaterial")
rsm.CreateIdAttr("redshift_usd_material")
rstd = UsdShade.Shader.Define(stage, f"{{g_mtl_path}}/RedshiftStd")
rstd.CreateIdAttr("redshift::StandardMaterial")
rstd.CreateInput("base_color_weight", Sdf.ValueTypeNames.Float).Set(1.0 if use_diffuse else 0.0)
rstd.CreateInput("refl_weight", Sdf.ValueTypeNames.Float).Set(0.0)
rstd.CreateInput("refl_roughness", Sdf.ValueTypeNames.Float).Set(1.0)
rstd.CreateInput("coat_weight", Sdf.ValueTypeNames.Float).Set(0.0)
rstd.CreateInput("sheen_weight", Sdf.ValueTypeNames.Float).Set(0.0)
rtex = UsdShade.Shader.Define(stage, f"{{g_mtl_path}}/RedshiftColorTex")
rtex.CreateIdAttr("redshift::TextureSampler")
rtex.CreateInput("tex0", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(ground_tex))
rtex.CreateInput("tex0_colorSpace", Sdf.ValueTypeNames.String).Set("Raw")
rtex.CreateInput("tspace_id", Sdf.ValueTypeNames.String).Set("st")
rtex.CreateInput("wrapU", Sdf.ValueTypeNames.Int).Set(0)
rtex.CreateInput("wrapV", Sdf.ValueTypeNames.Int).Set(0)
rtex.CreateOutput("outColor", Sdf.ValueTypeNames.Color3f)
rop = UsdShade.Shader.Define(stage, f"{{g_mtl_path}}/RedshiftOpacityTex")
rop.CreateIdAttr("redshift::TextureSampler")
rop.CreateInput("tex0", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(ground_op_tex))
rop.CreateInput("tex0_colorSpace", Sdf.ValueTypeNames.String).Set("Raw")
rop.CreateInput("tspace_id", Sdf.ValueTypeNames.String).Set("st")
rop.CreateInput("wrapU", Sdf.ValueTypeNames.Int).Set(0)
rop.CreateInput("wrapV", Sdf.ValueTypeNames.Int).Set(0)
rop.CreateOutput("outColor", Sdf.ValueTypeNames.Color3f)
rstd.CreateInput("base_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(rtex.ConnectableAPI(), "outColor")
rstd.CreateInput("emission_weight", Sdf.ValueTypeNames.Float).Set(ground_emission if use_emissive else 0.0)
rstd.CreateInput("emission_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(rtex.ConnectableAPI(), "outColor")
rstd.CreateInput("opacity_color", Sdf.ValueTypeNames.Color3f).ConnectToSource(rop.ConnectableAPI(), "outColor")
rstd.CreateOutput("outColor", Sdf.ValueTypeNames.Color3f)
rsm.CreateInput("Surface", Sdf.ValueTypeNames.Token).ConnectToSource(rstd.ConnectableAPI(), "outColor")
rsm.CreateOutput("Shader", Sdf.ValueTypeNames.Token)
g_mat.CreateSurfaceOutput("Redshift").ConnectToSource(rsm.ConnectableAPI(), "Shader")

UsdShade.MaterialBindingAPI.Apply(disc.GetPrim()).Bind(g_mat)
if is_shadow_catcher:
    disc_pv = UsdGeom.PrimvarsAPI(disc)
    # Karma native Shadow Catcher (Shadows-Only Holdout Mode = 2)
    disc_pv.CreatePrimvar("karma:object:matte", Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.constant).Set(True)
    disc_pv.CreatePrimvar("karma:object:holdoutmode", Sdf.ValueTypeNames.Int, UsdGeom.Tokens.constant).Set(2)
    disc_pv.CreatePrimvar("karma:object:rendervisibility", Sdf.ValueTypeNames.String, UsdGeom.Tokens.constant).Set("primary")
    # Arnold Matte
    disc_pv.CreatePrimvar("arnold:matte", Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.constant).Set(True)
    # Redshift Solaris Matte / Shadow Catcher
    disc_pv.CreatePrimvar("redshift:object:RS_objprop_matte_enable", Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.constant).Set(True)
    disc_pv.CreatePrimvar("redshift:object:RS_objprop_matte_showbackg", Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.constant).Set(True)
    disc_pv.CreatePrimvar("redshift:object:RS_objprop_matte_alpha", Sdf.ValueTypeNames.Float, UsdGeom.Tokens.constant).Set(0.0)
    disc_pv.CreatePrimvar("redshift:object:RS_objprop_matte_shadowenable", Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.constant).Set(True)
    disc_pv.CreatePrimvar("redshift:object:RS_objprop_matte_shadowalpha", Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.constant).Set(True)
    disc_pv.CreatePrimvar("redshift:object:RS_objprop_matte_shadowsFromM", Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.constant).Set(True)
    disc_pv.CreatePrimvar("redshift:object:RS_objprop_matte_diffscale", Sdf.ValueTypeNames.Float, UsdGeom.Tokens.constant).Set(0.0)
    disc_pv.CreatePrimvar("redshift:object:RS_objprop_matte_reflscale", Sdf.ValueTypeNames.Float, UsdGeom.Tokens.constant).Set(0.0)
    disc_pv.CreatePrimvar("redshift:object:RS_objprop_matte_refrscale", Sdf.ValueTypeNames.Float, UsdGeom.Tokens.constant).Set(0.0)
'''

    lop.parm("python").set(code)
    lop.bypass(False)
    lop.moveToGoodPosition()
    lop.cook(force=True)
    try:
        hou.hscript("texcache -c")
        hou.hscript("glcache -c")
    except Exception:
        pass
    if set_display_flag:
        try:
            lop.setDisplayFlag(True)
        except Exception:
            pass
    return lop
