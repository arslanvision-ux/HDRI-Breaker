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
            if parent_node.name() in ("hdri_maker_lookdev", "hdri_maker_materials"):
                lookdev_node = stage_net.node("hdri_maker_lookdev") if parent_node.name() == "hdri_maker_materials" else parent_node
                if lookdev_node is not None:
                    ins = lookdev_node.inputs()
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
    try:
        mat_lib = stage_net.node("hdri_maker_materials")
    except Exception:
        mat_lib = None
    if mat_lib is not None:
        try:
            mat_lib.setInput(0, lookdev)
            mat_lib.moveToGoodPosition()
        except Exception:
            pass
    display_target = mat_lib if mat_lib is not None else lookdev
    if set_display_flag:
        try:
            display_target.setDisplayFlag(True)
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


def _vop_connect(
    dst,
    input_name,
    src,
    output_name=None,
    fallback_in: Optional[int] = None,
) -> None:
    """Connect `src` output to `dst` named input (supports candidate name tuples)."""
    in_names = (input_name,) if isinstance(input_name, str) else tuple(input_name)
    idx = -1
    for nm in in_names:
        idx = dst.inputIndex(nm)
        if idx >= 0:
            break
    if idx < 0:
        if fallback_in is None:
            return
        idx = int(fallback_in)

    out_idx = 0
    if output_name:
        out_names = (output_name,) if isinstance(output_name, str) else tuple(output_name)
        for onm in out_names:
            oi = src.outputIndex(onm)
            if oi >= 0:
                out_idx = oi
                break
    dst.setInput(idx, src, out_idx)


def _set_vop_parm(node, name: str, value) -> None:
    p = node.parm(name)
    if p is not None:
        p.set(value)


def _set_vop_parm_tuple(node, name: str, value) -> None:
    pt = node.parmTuple(name)
    if pt is not None:
        pt.set(value)


def _normalize_render_engine(render_engine: Optional[str]) -> str:
    eng = str(render_engine or "Karma").strip().lower()
    if "redshift" in eng or eng == "rs":
        return "redshift"
    if "arnold" in eng or eng == "htoa":
        return "arnold"
    return "karma"


def _create_pbr_shader_for_engine(
    ml,
    name: str,
    color_rgb: Tuple[float, float, float],
    metallic: float,
    roughness: float,
    render_engine: str,
    vop_types,
    use_display_color: bool = False,
) -> str:
    """Create a dedicated single-renderer PBR shader inside a Solaris Material Library."""
    eng = _normalize_render_engine(render_engine)
    spec_w = 1.0 if metallic > 0.5 else max(0.0, 1.0 - float(roughness))

    if eng == "redshift" and "redshift_usd_material" in vop_types and "redshift::StandardMaterial" in vop_types:
        rsm = ml.createNode("redshift_usd_material", f"{name}_rs_mat")
        rstd = ml.createNode("redshift::StandardMaterial", f"{name}_rs_std")
        _set_vop_parm(rstd, "base_color_weight", 1.0)
        _set_vop_parm(rstd, "metalness", float(metallic))
        _set_vop_parm(rstd, "refl_weight", spec_w)
        _set_vop_parm(rstd, "refl_roughness", float(roughness))
        if use_display_color and "redshift::RSUserDataColor" in vop_types:
            rud = ml.createNode("redshift::RSUserDataColor", f"{name}_rs_dc")
            _set_vop_parm(rud, "attribute", "displayColor")
            _vop_connect(rstd, "base_color", rud, "out")
        else:
            _set_vop_parm_tuple(rstd, "base_color", color_rgb)
        _vop_connect(rsm, "Surface", rstd, "outColor")
        return rsm.name()

    if eng == "arnold" and "arnold::standard_surface" in vop_types:
        astd = ml.createNode("arnold::standard_surface", f"{name}_ai_std")
        _set_vop_parm(astd, "base", 1.0)
        _set_vop_parm(astd, "metalness", float(metallic))
        _set_vop_parm(astd, "specular", spec_w)
        _set_vop_parm(astd, "specular_roughness", float(roughness))
        ai_dc_type = next(
            (t for t in ("arnold::user_data_rgb", "arnold::user_data_color", "arnold::user_data_rgba") if t in vop_types),
            None,
        )
        if use_display_color and ai_dc_type is not None:
            aud = ml.createNode(ai_dc_type, f"{name}_ai_dc")
            _set_vop_parm(aud, "attribute", "displayColor")
            _vop_connect(astd, "base_color", aud, ("out", "rgb", "rgba"))
        else:
            _set_vop_parm_tuple(astd, "base_color", color_rgb)
        if "arnold_material" in vop_types:
            amat = ml.createNode("arnold_material", f"{name}_ai_mat")
            _vop_connect(amat, ("surface", "Surface"), astd, ("shader", "out", "rgb"), fallback_in=0)
            return amat.name()
        return astd.name()

    if "mtlxstandard_surface" in vop_types:
        ms = ml.createNode("mtlxstandard_surface", f"{name}_mtlx")
        _set_vop_parm(ms, "base", 1.0)
        _set_vop_parm(ms, "metalness", float(metallic))
        _set_vop_parm(ms, "specular", spec_w)
        _set_vop_parm(ms, "specular_roughness", float(roughness))
        if use_display_color and "mtlxgeompropvalue" in vop_types:
            mg = ml.createNode("mtlxgeompropvalue", f"{name}_mtlx_dc")
            _set_vop_parm(mg, "signature", "color3")
            _set_vop_parm(mg, "geomprop", "displayColor")
            _vop_connect(ms, "base_color", mg, "out")
        else:
            _set_vop_parm_tuple(ms, "base_color", color_rgb)
        return ms.name()

    # Fallback UsdPreviewSurface
    ups = ml.createNode("usdpreviewsurface", f"{name}_preview")
    _set_vop_parm(ups, "metallic", float(metallic))
    _set_vop_parm(ups, "roughness", float(roughness))
    if use_display_color and "usdprimvarreader" in vop_types:
        pvr = ml.createNode("usdprimvarreader", f"{name}_preview_dc")
        _set_vop_parm(pvr, "signature", "float3")
        _set_vop_parm(pvr, "varname", "displayColor")
        _vop_connect(ups, "diffuseColor", pvr, "result")
    else:
        _set_vop_parm_tuple(ups, "diffuseColor", color_rgb)
    return ups.name()


def _create_ground_shader_for_engine(
    ml,
    ground_tex: str,
    ground_op_tex: str,
    use_diffuse: bool,
    use_emissive: bool,
    ground_emission: float,
    is_shadow_catcher: bool,
    render_engine: str,
    vop_types,
) -> str:
    """Create the dedicated single-renderer ground_proj_mtl shader inside a Solaris Material Library."""
    eng = _normalize_render_engine(render_engine)

    if eng == "redshift" and "redshift_usd_material" in vop_types and "redshift::StandardMaterial" in vop_types:
        rsm = ml.createNode("redshift_usd_material", "ground_rs_mat")
        rstd = ml.createNode("redshift::StandardMaterial", "ground_rs_std")
        _set_vop_parm(rstd, "base_color_weight", 1.0 if use_diffuse else 0.0)
        _set_vop_parm(rstd, "refl_weight", 0.0)
        _set_vop_parm(rstd, "refl_roughness", 1.0)
        _set_vop_parm(rstd, "coat_weight", 0.0)
        _set_vop_parm(rstd, "sheen_weight", 0.0)
        _set_vop_parm(rstd, "emission_weight", ground_emission if use_emissive else 0.0)
        rtex = ml.createNode("redshift::TextureSampler", "ground_rs_col")
        _set_vop_parm(rtex, "tex0", ground_tex)
        _set_vop_parm(rtex, "tex0_colorSpace", "Raw")
        _set_vop_parm(rtex, "tspace_id", "st")
        _set_vop_parm(rtex, "wrapU", 0)
        _set_vop_parm(rtex, "wrapV", 0)
        rop = ml.createNode("redshift::TextureSampler", "ground_rs_op")
        _set_vop_parm(rop, "tex0", ground_op_tex)
        _set_vop_parm(rop, "tex0_colorSpace", "Raw")
        _set_vop_parm(rop, "tspace_id", "st")
        _set_vop_parm(rop, "wrapU", 0)
        _set_vop_parm(rop, "wrapV", 0)
        _vop_connect(rstd, "base_color", rtex, "outColor")
        _vop_connect(rstd, "emission_color", rtex, "outColor")
        _vop_connect(rstd, "opacity_color", rop, "outColor")
        _vop_connect(rsm, "Surface", rstd, "outColor")
        return rsm.name()

    if eng == "arnold" and "arnold::standard_surface" in vop_types:
        atex = ml.createNode("arnold::image", "ground_ai_col")
        _set_vop_parm(atex, "filename", ground_tex)
        _set_vop_parm(atex, "color_space", "raw")
        _set_vop_parm(atex, "swrap", "clamp")
        _set_vop_parm(atex, "twrap", "clamp")
        _set_vop_parm(atex, "uvset", "st")
        aop = ml.createNode("arnold::image", "ground_ai_op")
        _set_vop_parm(aop, "filename", ground_op_tex)
        _set_vop_parm(aop, "color_space", "raw")
        _set_vop_parm(aop, "swrap", "clamp")
        _set_vop_parm(aop, "twrap", "clamp")
        _set_vop_parm(aop, "uvset", "st")
        if is_shadow_catcher and "arnold::shadow_matte" in vop_types:
            ai = ml.createNode("arnold::shadow_matte", "ground_ai_matte")
            _set_vop_parm(ai, "background_type", 0)
            _set_vop_parm_tuple(ai, "background_color", (0.0, 0.0, 0.0))
            _set_vop_parm_tuple(ai, "shadow_color", (0.0, 0.0, 0.0))
            _set_vop_parm(ai, "shadow_opacity", 1.0)
            _set_vop_parm(ai, "diffuse_intensity", 0.0)
            _set_vop_parm(ai, "indirect_diffuse_enable", True)
            _vop_connect(ai, "opacity", aop, ("rgb", "out", "rgba"))
        else:
            ai = ml.createNode("arnold::standard_surface", "ground_ai_std")
            _set_vop_parm(ai, "base", 1.0 if use_diffuse else 0.0)
            _set_vop_parm(ai, "specular", 0.0)
            _set_vop_parm(ai, "specular_roughness", 1.0)
            _set_vop_parm(ai, "emission", ground_emission if use_emissive else 0.0)
            _vop_connect(ai, "base_color", atex, ("rgb", "out", "rgba"))
            _vop_connect(ai, "emission_color", atex, ("rgb", "out", "rgba"))
            _vop_connect(ai, "opacity", aop, ("rgb", "out", "rgba"))
        if "arnold_material" in vop_types:
            amat = ml.createNode("arnold_material", "ground_ai_mat")
            _vop_connect(amat, ("surface", "Surface"), ai, ("shader", "out", "rgb"), fallback_in=0)
            return amat.name()
        return ai.name()

    if "mtlxstandard_surface" in vop_types and "mtlximage" in vop_types:
        ms = ml.createNode("mtlxstandard_surface", "ground_mtlx")
        _set_vop_parm(ms, "base", 1.0 if use_diffuse else 0.0)
        _set_vop_parm(ms, "specular", 0.0)
        _set_vop_parm(ms, "specular_roughness", 1.0)
        _set_vop_parm(ms, "emission", ground_emission if use_emissive else 0.0)
        kuv = ml.createNode("mtlxtexcoord", "ground_mtlx_uv")
        _set_vop_parm(kuv, "index", 0)
        ktex = ml.createNode("mtlximage", "ground_mtlx_col")
        _set_vop_parm(ktex, "signature", "color3")
        _set_vop_parm(ktex, "file", ground_tex)
        _set_vop_parm(ktex, "uaddressmode", "clamp")
        _set_vop_parm(ktex, "vaddressmode", "clamp")
        _vop_connect(ktex, "texcoord", kuv, "out")
        kop = ml.createNode("mtlximage", "ground_mtlx_op")
        _set_vop_parm(kop, "signature", "color3")
        _set_vop_parm(kop, "file", ground_op_tex)
        _set_vop_parm(kop, "uaddressmode", "clamp")
        _set_vop_parm(kop, "vaddressmode", "clamp")
        _vop_connect(kop, "texcoord", kuv, "out")
        _vop_connect(ms, "base_color", ktex, "out")
        _vop_connect(ms, "emission_color", ktex, "out")
        _vop_connect(ms, "opacity", kop, "out")
        return ms.name()

    # Fallback UsdPreviewSurface
    ups = ml.createNode("usdpreviewsurface", "ground_preview")
    _set_vop_parm(ups, "roughness", 1.0)
    _set_vop_parm(ups, "metallic", 0.0)
    _set_vop_parm(ups, "opacityThreshold", 0.0)
    pvr = ml.createNode("usdprimvarreader", "ground_preview_uv")
    _set_vop_parm(pvr, "signature", "float2")
    _set_vop_parm(pvr, "varname", "st")
    utex = ml.createNode("usduvtexture", "ground_preview_col")
    _set_vop_parm(utex, "file", ground_tex)
    _set_vop_parm(utex, "sourceColorSpace", "raw")
    _set_vop_parm(utex, "wrapS", "clamp")
    _set_vop_parm(utex, "wrapT", "clamp")
    _vop_connect(utex, "st", pvr, "result")
    if use_diffuse:
        _vop_connect(ups, "diffuseColor", utex, "rgb")
    else:
        _set_vop_parm_tuple(ups, "diffuseColor", (0.0, 0.0, 0.0))
    if use_emissive and ground_emission > 0.0:
        uem = ml.createNode("usduvtexture", "ground_preview_em")
        _set_vop_parm(uem, "file", ground_tex)
        _set_vop_parm(uem, "sourceColorSpace", "raw")
        _set_vop_parm(uem, "wrapS", "clamp")
        _set_vop_parm(uem, "wrapT", "clamp")
        _set_vop_parm_tuple(uem, "scale", (ground_emission, ground_emission, ground_emission, 1.0))
        _vop_connect(uem, "st", pvr, "result")
        _vop_connect(ups, "emissiveColor", uem, "rgb")
    else:
        _set_vop_parm_tuple(ups, "emissiveColor", (0.0, 0.0, 0.0))
    uop = ml.createNode("usduvtexture", "ground_preview_op")
    _set_vop_parm(uop, "file", ground_op_tex)
    _set_vop_parm(uop, "sourceColorSpace", "raw")
    _set_vop_parm(uop, "wrapS", "clamp")
    _set_vop_parm(uop, "wrapT", "clamp")
    _vop_connect(uop, "st", pvr, "result")
    _vop_connect(ups, "opacity", uop, "r")
    return ups.name()


def _build_lookdev_material_library(
    stage_net,
    lookdev_lop,
    mat_lib_name: str,
    ground_tex: str,
    ground_op_tex: str,
    use_diffuse: bool,
    use_emissive: bool,
    ground_emission: float,
    is_shadow_catcher: bool,
    render_engine: str = "Karma (MaterialX)",
):
    """Create or rebuild the Solaris Material Library LOP with 5 unique materials for the chosen render engine."""
    ml = _create_lop(stage_net, ("materiallibrary",), mat_lib_name)
    for child in list(ml.children()):
        try:
            child.destroy()
        except Exception:
            pass

    vop_types = hou.vopNodeTypeCategory().nodeTypes()
    _set_vop_parm(ml, "matpathprefix", "/lookdev/materials/")

    entries = [
        (
            _create_pbr_shader_for_engine(ml, "tripod_stand_mtl", (0.08, 0.08, 0.09), 0.85, 0.35, render_engine, vop_types),
            "/lookdev/materials/tripod_stand_mtl",
            "/lookdev/tripod/stand/* /lookdev/tripod/macbeth/chart/backing",
        ),
        (
            _create_pbr_shader_for_engine(ml, "chrome_mtl", (0.95, 0.95, 0.95), 1.0, 0.001, render_engine, vop_types),
            "/lookdev/materials/chrome_mtl",
            "/lookdev/tripod/spheres/chrome_sphere",
        ),
        (
            _create_pbr_shader_for_engine(ml, "grey_18_mtl", (0.18, 0.18, 0.18), 0.0, 0.6, render_engine, vop_types),
            "/lookdev/materials/grey_18_mtl",
            "/lookdev/tripod/spheres/grey_sphere",
        ),
        (
            _create_pbr_shader_for_engine(
                ml, "macbeth_mtl", (0.18, 0.18, 0.18), 0.0, 0.85, render_engine, vop_types, use_display_color=True
            ),
            "/lookdev/materials/macbeth_mtl",
            "/lookdev/tripod/macbeth/chart/patch_*",
        ),
        (
            _create_ground_shader_for_engine(
                ml,
                ground_tex=ground_tex,
                ground_op_tex=ground_op_tex,
                use_diffuse=use_diffuse,
                use_emissive=use_emissive,
                ground_emission=ground_emission,
                is_shadow_catcher=is_shadow_catcher,
                render_engine=render_engine,
                vop_types=vop_types,
            ),
            "/lookdev/materials/ground_proj_mtl",
            "/lookdev/ground_disc",
        ),
    ]

    _set_vop_parm(ml, "materials", len(entries))
    for i, (vop_name, mat_path, geo_path) in enumerate(entries, start=1):
        _set_vop_parm(ml, f"matnode{i}", vop_name)
        _set_vop_parm(ml, f"matpath{i}", mat_path)
        _set_vop_parm(ml, f"assign{i}", 1)
        _set_vop_parm(ml, f"geopath{i}", geo_path)

    ml.layoutChildren()
    ml.setInput(0, lookdev_lop)
    ml.bypass(False)
    ml.moveToGoodPosition()
    ml.cook(force=True)
    return ml


def create_or_update_lookdev_and_ground(
    ground_tex_exr_path: str,
    ground_opacity_exr_path: str,
    parent_node=None,
    colorspace: str = "ACEScg",
    ground_radius: float = 10.0,
    tripod_height: float = 1.5,
    ground_emission: float = 0.5,
    ground_shader_mode: str = "Emissive + Diffuse",
    render_engine: str = "Karma (MaterialX)",
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
            # Counter-clockwise outward right-handed winding matching vertex normals
            indices.extend([p0, p0 + 1, p1 + 1, p1])
    mesh.GetPointsAttr().Set(pts)
    mesh.GetFaceVertexCountsAttr().Set(counts)
    mesh.GetFaceVertexIndicesAttr().Set(indices)
    mesh.GetNormalsAttr().Set(nrms)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(uvs)
    return mesh

root = "/lookdev"
_ensure_scope(root)
_ensure_scope(f"{{root}}/tripod")
_ensure_scope(f"{{root}}/tripod/stand")
_ensure_scope(f"{{root}}/tripod/spheres")
_ensure_scope(f"{{root}}/tripod/macbeth")

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

cbar = UsdGeom.Cylinder.Define(stage, f"{{root}}/tripod/stand/crossbar")
cbar.GetRadiusAttr().Set(0.012)
cbar.GetHeightAttr().Set(spacing * 2.0)
cbar.GetAxisAttr().Set("X")
xf_c = UsdGeom.Xformable(cbar.GetPrim())
xf_c.ClearXformOpOrder()
xf_c.AddTranslateOp().Set(Gf.Vec3d(0.0, crossbar_y, 0.0))

# 2. Chrome & 18% Gray Reference Spheres
chrome = _build_smooth_sphere(f"{{root}}/tripod/spheres/chrome_sphere", sphere_r)
xf_ch = UsdGeom.Xformable(chrome.GetPrim())
xf_ch.ClearXformOpOrder()
xf_ch.AddTranslateOp().Set(Gf.Vec3d(-spacing * 0.5, ball_y, 0.0))

grey = _build_smooth_sphere(f"{{root}}/tripod/spheres/grey_sphere", sphere_r)
xf_gr = UsdGeom.Xformable(grey.GetPrim())
xf_gr.ClearXformOpOrder()
xf_gr.AddTranslateOp().Set(Gf.Vec3d(spacing * 0.5, ball_y, 0.0))

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
    pm.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set(
        [Gf.Vec3f(float(col_out[0]), float(col_out[1]), float(col_out[2]))]
    )

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
disc_pv = UsdGeom.PrimvarsAPI(disc)
disc_pv.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(g_uvs)
disc_pv.CreatePrimvar("displayOpacity", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex).Set(g_opacities)
# Arnold requires opaque=False on polymesh prims to evaluate feathered shader opacity
disc_pv.CreatePrimvar("arnold:opaque", Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.constant).Set(False)

if is_shadow_catcher:
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

    mat_lib_name = "hdri_maker_materials" if node_name == "hdri_maker_lookdev" else f"{node_name}_materials"
    mat_lib = _build_lookdev_material_library(
        stage_net=stage,
        lookdev_lop=lop,
        mat_lib_name=mat_lib_name,
        ground_tex=ground_tex_norm,
        ground_op_tex=ground_op_norm,
        use_diffuse=bool(use_diffuse),
        use_emissive=bool(use_emissive),
        ground_emission=max(0.0, float(ground_emission)),
        is_shadow_catcher=bool(is_shadow_catcher),
        render_engine=render_engine,
    )

    try:
        hou.hscript("texcache -c")
        hou.hscript("glcache -c")
    except Exception:
        pass
    if set_display_flag:
        try:
            mat_lib.setDisplayFlag(True)
        except Exception:
            pass
    return mat_lib
