"""
End-to-end verification tests for Solaris HDRI Maker.

Run via Houdini's hython (tests full OIIO, Qt, and Solaris /stage integration):
    hython.exe tests/test_hdri_maker.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

# Ensure local python package is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "python"))

import OpenImageIO as oiio
from PIL import Image

from hdri_maker.hdri_generator import (
    HDRISynthesisParams,
    bake_planar_ground_texture,
    extract_and_inpaint_hotspot,
    extract_and_inpaint_multiple_hotspots,
    generate_latlong_hdri,
    render_preview_srgb,
)
from hdri_maker.lighting_analyzer import (
    PlateViewInput,
    analyze_multiview_plates,
    analyze_plate,
    cct_to_rgb,
    merge_lighting_profiles,
    rgb_to_cct_and_tint,
)
from hdri_maker.plate_io import (
    read_plate,
    resolve_plate_sequence,
    write_exr_32f,
)
from hdri_maker.solaris_stage import (
    HAS_HOU,
    create_or_update_area_light_from_hotspot,
    create_or_update_domelight,
    create_or_update_lookdev_and_ground,
)


def _make_synthetic_plate(width=256, height=144, hotspot_x=0.75, hotspot_y=0.25):
    """Create a synthetic linear plate with sky/ground gradient and a bright warm sun hotspot."""
    img = np.zeros((height, width, 3), dtype=np.float32)
    y_norm = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]

    # Cool blue sky in upper half, warm earthy ground in lower half
    sky_mask = (y_norm < 0.5).astype(np.float32)
    img[:, :, 0] = sky_mask * 0.14 + (1.0 - sky_mask) * 0.12
    img[:, :, 1] = sky_mask * 0.22 + (1.0 - sky_mask) * 0.10
    img[:, :, 2] = sky_mask * 0.45 + (1.0 - sky_mask) * 0.06

    # Add a bright warm sun hotspot
    yy, xx = np.mgrid[0:height, 0:width]
    dx = (xx / float(width)) - hotspot_x
    dy = (yy / float(height)) - hotspot_y
    r2 = dx * dx + dy * dy
    sun = 18.0 * np.exp(-r2 / (2.0 * 0.025 * 0.025)).astype(np.float32)
    img[:, :, 0] += sun * 1.0
    img[:, :, 1] += sun * 0.85
    img[:, :, 2] += sun * 0.55
    return img


class TestHDRIMaker(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_cct_roundtrip(self):
        for target_k in (3200.0, 5600.0, 6500.0, 9500.0):
            rgb = cct_to_rgb(target_k)
            est_k, _ = rgb_to_cct_and_tint(rgb)
            self.assertAlmostEqual(est_k, target_k, delta=350.0)

    def test_plate_analysis_and_32f_exr_export(self):
        plate_arr = _make_synthetic_plate()
        plate_exr = os.path.join(self.tmpdir.name, "plate_0001.exr")
        write_exr_32f(plate_exr, plate_arr)

        loaded = read_plate(plate_exr)
        self.assertTrue(loaded.is_hdr_source if hasattr(loaded, "is_hdr_source") else loaded.is_hdr)

        profile = analyze_plate(loaded.pixels, is_hdr=loaded.is_hdr, fov_h_deg=90.0)
        self.assertGreater(profile.dynamic_range_stops, 3.0)
        self.assertGreater(len(profile.hotspots), 0)

        # Hotspot was placed at u=0.75 (right of center => positive azimuth ~ +22.5 deg)
        key = profile.hotspots[0]
        self.assertGreater(key.azimuth_deg, 10.0)
        self.assertGreater(key.elevation_deg, 5.0)
        self.assertGreater(key.peak_intensity, 5.0)

        # Generate 2:1 Lat/Long HDRI (with hotspot_boost=1.0 to test synthetic hotspot energy)
        hdri = generate_latlong_hdri(
            profile, HDRISynthesisParams(width=512, height=256, hotspot_boost=1.0)
        )
        self.assertEqual(hdri.shape, (256, 512, 3))
        self.assertEqual(hdri.dtype, np.float32)
        self.assertGreater(float(np.max(hdri)), 5.0)

        # Write and verify true 32-bit float EXR via OpenImageIO
        out_exr = os.path.join(self.tmpdir.name, "generated_latlong.exr")
        write_exr_32f(
            out_exr,
            hdri,
            metadata={"hdri_maker:cct_kelvin": f"{profile.overall_cct_kelvin:.1f}"},
        )
        inp = oiio.ImageInput.open(out_exr)
        self.assertIsNotNone(inp)
        try:
            spec = inp.spec()
            self.assertEqual(spec.width, 512)
            self.assertEqual(spec.height, 256)
            self.assertEqual(spec.nchannels, 3)
            self.assertEqual(spec.format, oiio.FLOAT)
            self.assertEqual(spec.get_string_attribute("hdri_maker:envmap"), "latlong")
        finally:
            inp.close()

        # Verify preview tone-mapping
        preview = render_preview_srgb(hdri, preview_ev=0.0, tonemap="aces")
        self.assertEqual(preview.dtype, np.uint8)
        self.assertEqual(preview.shape, (256, 512, 3))

    def test_jpg_sequence_resolution_and_merge(self):
        for f in range(1, 4):
            jpg_path = os.path.join(self.tmpdir.name, f"shot_{f:04d}.jpg")
            arr = np.clip(_make_synthetic_plate(hotspot_x=0.25 + 0.05 * f), 0.0, 1.0)
            img = Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")
            img.save(jpg_path)

        pattern = os.path.join(self.tmpdir.name, "shot_$F4.jpg")
        frames = resolve_plate_sequence(pattern)
        self.assertEqual([f for f, _ in frames], [1, 2, 3])

        profiles = []
        for fnum, fpath in frames:
            p = read_plate(fpath, frame=fnum)
            profiles.append(analyze_plate(p.pixels, is_hdr=p.is_hdr, frame=fnum))

        merged = merge_lighting_profiles(profiles)
        self.assertGreater(len(merged.hotspots), 0)
        self.assertTrue(merged.hotspots[0].is_clipped_ldr)
        self.assertGreater(merged.hotspots[0].peak_intensity, 2.0)

    def test_multiview_different_positions(self):
        # 4 plates shot at 0, +90, 180, -90 degrees azimuth with 100-deg FOV
        views = []
        for idx, az in enumerate((0.0, 90.0, 180.0, -90.0)):
            # Put a bright hotspot only in the +90 deg plate
            arr = _make_synthetic_plate(hotspot_x=0.5, hotspot_y=0.25)
            if az != 90.0:
                arr = np.clip(arr, 0.0, 0.4)
            views.append(
                PlateViewInput(
                    pixels=arr,
                    is_hdr=True,
                    source_path=f"view_{idx}.exr",
                    fov_h_deg=100.0,
                    cam_azimuth_deg=az,
                    cam_elevation_deg=0.0,
                )
            )

        prof = analyze_multiview_plates(views, max_hotspots=5)
        self.assertIsNotNone(prof.spherical_plate_map)
        self.assertIsNotNone(prof.spherical_coverage_mask)
        # Equator should have high coverage across all 360 degrees
        mid_row = prof.spherical_coverage_mask[prof.spherical_coverage_mask.shape[0] // 2]
        self.assertGreater(float(np.mean(mid_row)), 0.7)
        # Brightest hotspot should be near +90 deg azimuth
        self.assertGreater(len(prof.hotspots), 0)
        self.assertAlmostEqual(prof.hotspots[0].azimuth_deg, 90.0, delta=15.0)

        hdri = generate_latlong_hdri(prof, HDRISynthesisParams(width=512, height=256))
        self.assertEqual(hdri.shape, (256, 512, 3))
        self.assertEqual(hdri.dtype, np.float32)
        self.assertGreater(float(np.max(hdri)), 5.0)

        hdri_blurred = generate_latlong_hdri(
            prof, HDRISynthesisParams(width=512, height=256, hdri_blur=0.35)
        )
        self.assertEqual(hdri_blurred.shape, (256, 512, 3))
        self.assertEqual(hdri_blurred.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(hdri_blurred)))
        self.assertLess(float(np.std(hdri_blurred)), float(np.std(hdri)))

    def test_solaris_stage_integration(self):
        if not HAS_HOU:
            self.skipTest("Not running inside Houdini hython.")
        import hou

        hou.hipFile.clear(suppress_save_prompt=True)
        plate_arr = _make_synthetic_plate()
        profile = analyze_plate(plate_arr, is_hdr=True)
        hdri = generate_latlong_hdri(profile, HDRISynthesisParams(width=256, height=128))
        out_exr = os.path.join(self.tmpdir.name, "solaris_dome.exr")
        write_exr_32f(out_exr, hdri)

        cleaned_multi, multi_results = extract_and_inpaint_multiple_hotspots(
            hdri,
            [
                {"rect": (0.50, 0.25, 0.65, 0.45), "shape": "rect"},
                {"rect": (0.20, 0.30, 0.32, 0.45), "shape": "circle"},
            ],
            inpaint_hdri=True,
        )
        self.assertEqual(len(multi_results), 2)
        self.assertEqual(multi_results[0][1].shape, "rect")
        self.assertEqual(multi_results[1][1].shape, "circle")
        light_tex, light_info = multi_results[0]
        self.assertEqual(light_tex.dtype, np.float32)
        self.assertEqual(light_tex.shape[2], 4)
        light_exr = os.path.join(self.tmpdir.name, "solaris_light.exr")
        write_exr_32f(light_exr, light_tex)

        dome = create_or_update_domelight(out_exr, profile=profile)
        self.assertIsNotNone(dome)
        self.assertTrue(dome.path().startswith("/stage/"))
        self.assertAlmostEqual(dome.parm("ry").eval(), 0.0)
        self.assertEqual(dome.parm("xn__inputstexturefile_r3ah").eval(), out_exr)
        self.assertEqual(dome.parm("xn__inputstextureformat_06ah").evalAsString(), "latlong")

        area_light = create_or_update_area_light_from_hotspot(
            light_info, light_tex_exr_path=light_exr, parent_dome=dome
        )
        self.assertIsNotNone(area_light)
        self.assertEqual(area_light.parm("lighttype").evalAsString(), "UsdLuxRectLight")
        self.assertEqual(area_light.parm("xn__inputstexturefile_r3ah").eval(), light_exr)

        disk_tex, disk_info = multi_results[1]
        # Verify circle/disk texture masks corners to 0.0 in both RGB and Alpha
        self.assertAlmostEqual(float(np.max(disk_tex[0, 0])), 0.0, places=5)
        self.assertGreater(float(np.max(disk_tex[disk_tex.shape[0] // 2, disk_tex.shape[1] // 2, :3])), 0.0)
        disk_exr = os.path.join(self.tmpdir.name, "solaris_disk.exr")
        write_exr_32f(disk_exr, disk_tex)
        disk_light = create_or_update_area_light_from_hotspot(
            disk_info,
            light_tex_exr_path=disk_exr,
            parent_dome=area_light,
            node_name="hdri_maker_disk_test",
            prim_path="/lights/hdri_maker_disk_test",
        )
        self.assertEqual(disk_light.parm("lighttype").evalAsString(), "UsdLuxRectLight")
        self.assertEqual(disk_light.parm("xn__inputstexturefile_r3ah").eval(), disk_exr)

        g_rgb, g_op = bake_planar_ground_texture(hdri, ground_radius=10.0, tripod_height=1.5, resolution=128)
        self.assertEqual(g_rgb.shape, (128, 128, 3))
        self.assertEqual(g_op.shape, (128, 128, 3))
        g_tex_exr = os.path.join(self.tmpdir.name, "ground_proj.exr")
        g_op_exr = os.path.join(self.tmpdir.name, "ground_op.exr")
        write_exr_32f(g_tex_exr, g_rgb, output_colorspace="ACEScg")
        write_exr_32f(g_op_exr, g_op, output_colorspace="Linear Rec.709")

        mat_lib = create_or_update_lookdev_and_ground(
            g_tex_exr, g_op_exr, parent_node=dome, colorspace="ACEScg"
        )
        self.assertEqual(mat_lib.type().name(), "materiallibrary")
        lookdev_lop = mat_lib.inputs()[0]
        # Even when parent_node=dome is passed, lookdev_lop must wire after the last light (disk_light)
        self.assertEqual(lookdev_lop.inputs()[0].path(), disk_light.path())
        self.assertTrue(mat_lib.isDisplayFlagSet())

        # Creating/updating a new area light after lookdev_lop already exists must keep lookdev_lop + mat_lib at the end
        extra_light = create_or_update_area_light_from_hotspot(
            light_info,
            light_tex_exr_path=light_exr,
            parent_dome=disk_light,
            node_name="hdri_maker_area_1",
            prim_path="/lights/hdri_maker_area_1",
            set_display_flag=True,
        )
        self.assertEqual(lookdev_lop.inputs()[0].path(), extra_light.path())
        self.assertEqual(mat_lib.inputs()[0].path(), lookdev_lop.path())
        self.assertTrue(mat_lib.isDisplayFlagSet())

        stage = mat_lib.stage()
        self.assertTrue(stage.GetPrimAtPath("/lookdev/tripod/spheres/chrome_sphere").IsValid())
        self.assertTrue(stage.GetPrimAtPath("/lookdev/tripod/spheres/grey_sphere").IsValid())
        self.assertTrue(stage.GetPrimAtPath("/lookdev/tripod/macbeth/chart/patch_00").IsValid())
        self.assertTrue(stage.GetPrimAtPath("/lookdev/ground_disc").IsValid())
        g_mtl = stage.GetPrimAtPath("/lookdev/materials/ground_proj_mtl")
        self.assertTrue(g_mtl.IsValid())
        self.assertTrue(g_mtl.GetAttribute("outputs:mtlx:surface").IsValid())
        # Single unique macbeth_mtl shader instead of 24 per-patch materials
        mb_mtl = stage.GetPrimAtPath("/lookdev/materials/macbeth_mtl")
        self.assertTrue(mb_mtl.IsValid())
        self.assertTrue(mb_mtl.GetAttribute("outputs:mtlx:surface").IsValid())
        self.assertFalse(stage.GetPrimAtPath("/lookdev/materials/macbeth_00").IsValid())

        # Verify lookdev spheres have outward right-handed winding aligned with vertex normals
        from pxr import Gf
        chrome_prim = stage.GetPrimAtPath("/lookdev/tripod/spheres/chrome_sphere")
        self.assertTrue(chrome_prim.IsValid())
        c_pts = chrome_prim.GetAttribute("points").Get()
        c_nrms = chrome_prim.GetAttribute("normals").Get()
        c_counts = chrome_prim.GetAttribute("faceVertexCounts").Get()
        c_idx = chrome_prim.GetAttribute("faceVertexIndices").Get()
        mid_face = len(c_counts) // 2
        f_off = sum(c_counts[:mid_face])
        i0, i1, i2 = c_idx[f_off], c_idx[f_off + 1], c_idx[f_off + 2]
        face_nrm = Gf.Cross(c_pts[i1] - c_pts[i0], c_pts[i2] - c_pts[i0])
        self.assertGreater(Gf.Dot(face_nrm, c_nrms[i0]), 0.0)
        # Verify ground_disc sets arnold:opaque=False so Arnold evaluates feathered opacity
        disc_prim = stage.GetPrimAtPath("/lookdev/ground_disc")
        self.assertTrue(disc_prim.IsValid())
        self.assertIs(disc_prim.GetAttribute("primvars:arnold:opaque").Get(), False)

        # Switch render_engine to Redshift and verify native Redshift Material Library outputs
        rs_lib = create_or_update_lookdev_and_ground(
            g_tex_exr, g_op_exr, parent_node=extra_light, colorspace="ACEScg", render_engine="Redshift"
        )
        rs_stage = rs_lib.stage()
        rs_g_mtl = rs_stage.GetPrimAtPath("/lookdev/materials/ground_proj_mtl")
        self.assertTrue(rs_g_mtl.GetAttribute("outputs:Redshift:surface").IsValid())
        rs_std = rs_stage.GetPrimAtPath("/lookdev/materials/ground_proj_mtl/ground_rs_std")
        self.assertTrue(rs_std.IsValid())
        self.assertAlmostEqual(float(rs_std.GetAttribute("inputs:refl_weight").Get()), 0.0, places=5)
        self.assertAlmostEqual(float(rs_std.GetAttribute("inputs:refl_roughness").Get()), 1.0, places=5)
        rs_mb_mtl = rs_stage.GetPrimAtPath("/lookdev/materials/macbeth_mtl")
        self.assertTrue(rs_mb_mtl.GetAttribute("outputs:Redshift:surface").IsValid())

        # Verify Shadow Catcher (Matte) sets Karma holdoutmode=2 and Redshift matte primvars
        sc_lop = create_or_update_lookdev_and_ground(
            g_tex_exr,
            g_op_exr,
            parent_node=extra_light,
            colorspace="ACEScg",
            ground_shader_mode="Shadow Catcher (Matte)",
            node_name="hdri_maker_lookdev_sc_test",
        )
        sc_stage = sc_lop.stage()
        sc_disc = sc_stage.GetPrimAtPath("/lookdev/ground_disc")
        self.assertEqual(sc_disc.GetAttribute("primvars:karma:object:holdoutmode").Get(), 2)
        self.assertTrue(sc_disc.GetAttribute("primvars:redshift:object:RS_objprop_matte_enable").Get())
        self.assertTrue(sc_disc.GetAttribute("primvars:redshift:object:RS_objprop_matte_showbackg").Get())
        self.assertTrue(sc_disc.GetAttribute("primvars:redshift:object:RS_objprop_matte_shadowenable").Get())
        self.assertTrue(sc_disc.GetAttribute("primvars:redshift:object:RS_objprop_matte_shadowalpha").Get())

        # Verify UI widget instantiates cleanly under current Qt binding (PySide2 or PySide6)
        from hdri_maker.qt_compat import QtWidgets
        from hdri_maker.ui_panel import HDRIMakerWidget

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        widget = HDRIMakerWidget(restore_session=False)
        self.assertIsNotNone(widget)
        res_items = [widget.res_combo.itemText(i) for i in range(widget.res_combo.count())]
        self.assertIn("8192 x 4096", res_items)
        widget.res_combo.setCurrentText("8192 x 4096")
        self.assertEqual(widget._current_synthesis_params().width, 8192)
        self.assertEqual(widget._current_synthesis_params().height, 4096)
        widget.light_res_combo.setCurrentText("1024 x 1024")
        self.assertEqual(widget._current_light_tex_size(), (1024, 1024))
        widget.preview_label.set_zoom(2.5)
        self.assertAlmostEqual(widget.preview_label.zoom_factor(), 2.5)
        self.assertEqual(widget.zoom_reset_btn.text(), "2.5x")
        widget.preview_label.reset_zoom()
        self.assertAlmostEqual(widget.preview_label.zoom_factor(), 1.0)
        widget.yaw_spin.setValue(45.0)
        widget.vshift_spin.setValue(-10.0)
        widget.render_engine_combo.setCurrentText("Arnold")
        state = widget.get_state_dict()
        widget2 = HDRIMakerWidget(restore_session=False)
        widget2.apply_state_dict(state, auto_analyze=False)
        self.assertAlmostEqual(widget2.yaw_spin.value(), 45.0)
        self.assertAlmostEqual(widget2.vshift_spin.value(), -10.0)
        self.assertEqual(widget2.render_engine_combo.currentText(), "Arnold")

        # Verify scene-level userData persistence across close & reopen
        import hou
        widget.plate_path_edit.setText("")
        widget._auto_save_session()
        self.assertTrue(bool(hou.node("/").userData("hdri_maker_state")))
        widget3 = HDRIMakerWidget(restore_session=True)
        self.assertAlmostEqual(widget3.yaw_spin.value(), 45.0)
        self.assertEqual(widget3.render_engine_combo.currentText(), "Arnold")
        widget3.close()

        widget.yaw_spin.reset()
        self.assertAlmostEqual(widget.yaw_spin.value(), 0.0)
        widget.close()
        widget2.close()

    def test_hdri_breaker_new_features_and_linux_packaging(self):
        from hdri_maker.lighting_analyzer import estimate_horizon_shift_deg
        from hdri_maker.plate_io import linear_to_srgb, merge_ldr_brackets_to_hdr

        # 1. Bracketed LDR exposure merge (-2 EV, 0 EV, +2 EV) -> true 32-bit HDR
        base_hdr = _make_synthetic_plate(width=128, height=64)
        bracket_paths = []
        for idx, ev in enumerate((-2.0, 0.0, 2.0)):
            exposed = np.clip(linear_to_srgb(base_hdr * (2.0 ** ev)), 0.0, 1.0)
            b_path = os.path.join(self.tmpdir.name, f"bracket_{idx}.jpg")
            Image.fromarray((exposed * 255.0 + 0.5).astype(np.uint8), mode="RGB").save(b_path, quality=95)
            bracket_paths.append(b_path)

        merged_hdr = merge_ldr_brackets_to_hdr(bracket_paths, ev_offsets=[-2.0, 0.0, 2.0], colorspace="sRGB")
        self.assertEqual(merged_hdr.shape, (64, 128, 3))
        self.assertEqual(merged_hdr.dtype, np.float32)
        self.assertGreater(float(np.max(merged_hdr)), 1.5)

        # 2. Automatic Horizon Detection (sky above v=0.40, dark ground below v=0.40 -> ~ -18 deg shift)
        synth_map = np.zeros((100, 200, 3), dtype=np.float32)
        synth_map[:40, :] = np.array([1.2, 1.6, 2.4], dtype=np.float32)
        synth_map[40:, :] = np.array([0.12, 0.10, 0.06], dtype=np.float32)
        est_shift = estimate_horizon_shift_deg(synth_map)
        self.assertAlmostEqual(est_shift, -18.0, delta=4.0)

        # 3. False Color (EV) preview tonemapping & anti-aliased downsample_image
        from hdri_maker.plate_io import downsample_image

        big_arr = np.ones((300, 600, 3), dtype=np.float32) * 2.5
        ds_arr = downsample_image(big_arr, max_dim=200)
        self.assertEqual(ds_arr.shape, (100, 200, 3))
        self.assertEqual(ds_arr.dtype, np.float32)
        self.assertAlmostEqual(float(np.mean(ds_arr)), 2.5, places=4)

        fc_rgb = render_preview_srgb(synth_map, preview_ev=0.0, tonemap="False Color (EV)")
        self.assertEqual(fc_rgb.shape, (100, 200, 3))
        self.assertEqual(fc_rgb.dtype, np.uint8)
        self.assertFalse(np.allclose(fc_rgb[10, 100], fc_rgb[80, 100]))

        # 4. Linux Rocky 8 / 9 shell script & installer invariants
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for sh_name in ("install.sh", "uninstall.sh", "run_tests.sh"):
            sh_path = os.path.join(repo_root, sh_name)
            self.assertTrue(os.path.isfile(sh_path), f"Missing {sh_name}")
            with open(sh_path, "rb") as f:
                raw = f.read()
            self.assertNotIn(b"\r\n", raw, f"{sh_name} must use strict LF line endings for Linux")
            self.assertNotIn(b"&>", raw, f"{sh_name} must avoid non-POSIX &> redirection")

        sys.path.insert(0, os.path.join(repo_root, "scripts"))
        import install as installer_mod

        mock_pref = os.path.join(self.tmpdir.name, "houdini20.5")
        rc = installer_mod.install_package(repo_root, target_dir=mock_pref)
        self.assertEqual(rc, 0)
        pkg_json = os.path.join(mock_pref, "packages", "hdri_maker.json")
        self.assertTrue(os.path.isfile(pkg_json))

        # 5. Macbeth White Balance gains & MacbethWBDialog sampling
        from hdri_maker.hdri_generator import compute_neutral_wb_gains

        sample_rgb = (0.36, 0.24, 0.12)
        gains, cct, _duv, lum = compute_neutral_wb_gains(sample_rgb)
        balanced = (sample_rgb[0] * gains[0], sample_rgb[1] * gains[1], sample_rgb[2] * gains[2])
        self.assertAlmostEqual(balanced[0], balanced[1], places=4)
        self.assertAlmostEqual(balanced[1], balanced[2], places=4)
        self.assertAlmostEqual(
            0.2126 * balanced[0] + 0.7152 * balanced[1] + 0.0722 * balanced[2],
            lum,
            places=4,
        )
        self.assertLess(cct, 5000.0)

        prof = analyze_plate(base_hdr, is_hdr=True)
        hdri_wb = generate_latlong_hdri(
            prof, HDRISynthesisParams(width=128, height=64, wb_gains=gains)
        )
        self.assertEqual(hdri_wb.shape, (64, 128, 3))
        self.assertTrue(np.all(np.isfinite(hdri_wb)))

        if HAS_HOU:
            from hdri_maker.qt_compat import QtWidgets
            from hdri_maker.ui_panel import MacbethWBDialog

            _app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            macbeth_img = np.full((64, 96, 3), np.array(sample_rgb, dtype=np.float32), dtype=np.float32)
            macbeth_exr = os.path.join(self.tmpdir.name, "macbeth_ref.exr")
            write_exr_32f(macbeth_exr, macbeth_img, output_colorspace="Linear Rec.709")
            dlg = MacbethWBDialog(initial_path=macbeth_exr, initial_colorspace="Linear Rec.709")
            self.assertIsNotNone(dlg.canvas.pixmap())
            self.assertFalse(dlg.canvas.pixmap().isNull())
            sampled = dlg.sample_patch_rect((0.25, 0.25, 0.75, 0.75))
            self.assertIsNotNone(sampled)
            self.assertAlmostEqual(sampled[0], gains[0], places=3)
            self.assertAlmostEqual(sampled[1], gains[1], places=3)
            self.assertAlmostEqual(sampled[2], gains[2], places=3)

    def test_plate_transform_roll_scale_warp_and_hit_test(self):
        from hdri_maker.lighting_analyzer import (
            compute_plate_outline_uv,
            hit_test_plate_uv,
            project_multiview_plates_to_latlong,
        )

        img = np.zeros((64, 96, 3), dtype=np.float32)
        img[:32, :, 0] = 1.0  # top half red
        img[32:, :, 2] = 1.0  # bottom half blue

        v0 = PlateViewInput(pixels=img, is_hdr=True, fov_h_deg=60.0, cam_azimuth_deg=0.0, cam_elevation_deg=0.0)
        v1 = PlateViewInput(
            pixels=img,
            is_hdr=True,
            fov_h_deg=60.0,
            cam_azimuth_deg=90.0,
            cam_elevation_deg=15.0,
            cam_roll_deg=25.0,
            scale_x=1.2,
            scale_y=0.9,
            warp_k1=0.2,
        )

        # Hit test center of v0 (u=0.5, v=0.5) and v1 (az=90 -> u=0.75, el=15 -> v=0.5 - 15/180)
        self.assertEqual(hit_test_plate_uv([v0, v1], 0.5, 0.5), 0)
        self.assertEqual(hit_test_plate_uv([v0, v1], 0.75, 0.5 - 15.0 / 180.0), 1)
        self.assertIsNone(hit_test_plate_uv([v0, v1], 0.1, 0.1))

        # Outline computation
        outline = compute_plate_outline_uv(v1)
        self.assertGreaterEqual(len(outline["polylines"]), 1)
        self.assertEqual(len(outline["corners"]), 4)

        # Projection with roll, scale, warp produces finite RGB and coverage
        rgb, cov = project_multiview_plates_to_latlong([v0, v1], out_width=128, out_height=64)
        self.assertEqual(rgb.shape, (64, 128, 3))
        self.assertEqual(cov.shape, (64, 128))
        self.assertTrue(np.all(np.isfinite(rgb)))
        self.assertGreater(float(np.max(cov)), 0.8)

    def test_advanced_ptgui_and_sequence_features(self):
        from hdri_maker.hdri_generator import export_hdri_sequence
        from hdri_maker.lighting_analyzer import (
            auto_equalize_plate_exposures,
            compute_overlap_difference_map,
            equirect_rect_to_plate_rect,
            project_multiview_plates_to_latlong,
        )
        from hdri_maker.plate_io import format_frame_path, read_plate_metadata

        # 1. Fisheye, vignette, exposure_ev, exclude_rects
        img_a = np.full((64, 64, 3), 0.4, dtype=np.float32)
        img_b = np.full((64, 64, 3), 0.2, dtype=np.float32)  # 1 stop darker
        va = PlateViewInput(
            pixels=img_a, is_hdr=True, fov_h_deg=120.0, cam_azimuth_deg=0.0,
            lens_type="fisheye_equidistant", vignette_v1=0.3,
        )
        vb = PlateViewInput(
            pixels=img_b, is_hdr=True, fov_h_deg=120.0, cam_azimuth_deg=30.0,
            lens_type="fisheye_equisolid", exclude_rects=[(0.0, 0.0, 0.2, 0.2)],
        )
        rgb, _ = project_multiview_plates_to_latlong([va, vb], 128, 64)
        self.assertEqual(rgb.shape, (64, 128, 3))
        self.assertTrue(np.all(np.isfinite(rgb)))

        # 2. Auto-equalize EV & Overlap difference map
        eq = auto_equalize_plate_exposures([va, vb])
        self.assertGreater(eq[1].exposure_ev, 0.5)
        diff = compute_overlap_difference_map(eq, 128, 64)
        self.assertEqual(diff.shape, (64, 128, 3))

        # 3. Equirect mask mapping
        rect_p = equirect_rect_to_plate_rect(
            PlateViewInput(pixels=img_a, is_hdr=True, fov_h_deg=90.0),
            (0.45, 0.45, 0.55, 0.55),
        )
        self.assertIsNotNone(rect_p)

        # 4. Physical sky fill
        prof = analyze_plate(img_a, is_hdr=True, fov_h_deg=60.0)
        hdri_sky = generate_latlong_hdri(
            prof, HDRISynthesisParams(width=128, height=64, physical_sky_fill=0.8), views=[va]
        )
        self.assertEqual(hdri_sky.shape, (64, 128, 3))
        self.assertTrue(np.all(np.isfinite(hdri_sky)))

        # 5. Sequence export & metadata fallback
        self.assertTrue(format_frame_path("out.$F4.exr", 12).endswith("out.0012.exr"))
        pat, written = export_hdri_sequence(
            [analyze_plate(img_a, is_hdr=True, frame=1), analyze_plate(img_b, is_hdr=True, frame=2)],
            [va, vb],
            HDRISynthesisParams(width=64, height=32),
            os.path.join(self.tmpdir.name, "seq.$F4.exr"),
        )
        self.assertEqual(len(written), 2)
        for wf in written:
            self.assertTrue(os.path.isfile(wf))
        meta = read_plate_metadata(written[0])
        self.assertIn("roll_deg", meta)

    def test_environment_mapping_enhancements(self):
        from hdri_maker.hdri_generator import (
            HDRISynthesisParams,
            export_hdri_sequence,
            generate_diffuse_irradiance_hdri,
            generate_latlong_hdri,
            recover_clipped_highlights,
            render_preview_srgb,
        )
        from hdri_maker.lighting_analyzer import (
            PlateViewInput,
            analyze_plate,
            estimate_horizon_roll_deg,
            estimate_radial_vignette,
            project_multiview_plates_to_latlong,
        )

        # 1. Chrome ball unwrap
        ball = np.full((64, 64, 3), 0.35, dtype=np.float32)
        ball[24:40, 24:40] = [1.0, 0.8, 0.6]
        cb_view = PlateViewInput(pixels=ball, is_hdr=False, lens_type="chrome_ball")
        cb_rgb, cb_cov = project_multiview_plates_to_latlong([cb_view], out_width=128, out_height=64)
        self.assertEqual(cb_rgb.shape, (64, 128, 3))
        self.assertGreater(float(np.mean(cb_cov)), 0.5)

        # 2. Roll & Vignette estimation
        img_vig = np.full((64, 64, 3), 0.8, dtype=np.float32)
        img_vig[:10, :10] = 0.2
        img_vig[:10, -10:] = 0.2
        img_vig[-10:, :10] = 0.2
        img_vig[-10:, -10:] = 0.2
        self.assertGreater(estimate_radial_vignette(img_vig), 0.2)
        self.assertTrue(-25.0 <= estimate_horizon_roll_deg(img_vig) <= 25.0)

        # 3. Highlight recovery, nadir cleanup, neutralize, roughness preview, diffuse split
        clipped = np.full((64, 128, 3), 0.2, dtype=np.float32)
        clipped[:, :, 0] = 0.45  # warm cast
        clipped[24:40, 56:72] = 1.0
        rec = recover_clipped_highlights(clipped, strength=3.0)
        self.assertGreater(float(np.max(rec)), 1.5)

        prof = analyze_plate(img_vig, is_hdr=False, fov_h_deg=90.0)
        synth = generate_latlong_hdri(
            prof,
            HDRISynthesisParams(
                width=128,
                height=64,
                highlight_recovery=2.0,
                nadir_cleanup=0.75,
                neutralize_environment=True,
            ),
            views=[PlateViewInput(pixels=img_vig, is_hdr=False, fov_h_deg=90.0)],
        )
        self.assertEqual(synth.shape, (64, 128, 3))
        self.assertTrue(np.all(np.isfinite(synth)))

        diff_exr = generate_diffuse_irradiance_hdri(rec)
        self.assertEqual(diff_exr.shape, rec.shape)
        self.assertLessEqual(float(np.max(diff_exr)), float(np.max(rec)))

        rough_srgb = render_preview_srgb(synth, roughness_preview=0.6)
        self.assertEqual(rough_srgb.dtype, np.uint8)

        # 4. Temporal stabilize + diffuse split sequence export
        p1 = analyze_plate(np.full((32, 32, 3), 0.2, dtype=np.float32), is_hdr=True, frame=1)
        p2 = analyze_plate(np.full((32, 32, 3), 0.8, dtype=np.float32), is_hdr=True, frame=2)
        _, seq_paths = export_hdri_sequence(
            [p1, p2],
            [],
            HDRISynthesisParams(width=64, height=32, temporal_stabilize=0.8),
            os.path.join(self.tmpdir.name, "stab.$F4.exr"),
            export_diffuse_split=True,
        )
        self.assertEqual(len(seq_paths), 2)
        for sp in seq_paths:
            stem, ext = os.path.splitext(sp)
            self.assertTrue(os.path.isfile(f"{stem}_diffuse{ext}"))

        # 4b. 32-bit Float Canvas Brush Tools & Fast Baked Post-Adjustments
        from hdri_maker.hdri_generator import apply_brush_dab, apply_brush_strokes, apply_post_adjustments_to_hdri

        canvas = np.full((64, 128, 3), 0.25, dtype=np.float32)
        canvas[:, 10:25, :] = [4.0, 0.2, 0.2]  # bright red region to clone/heal/burn
        # Clone from (0.12, 0.5) onto wrap seam (0.99, 0.5)
        cloned = apply_brush_dab(
            canvas, u=0.99, v=0.5, mode="clone", radius_u=0.05, strength=1.0, src_offset_uv=(0.12 - 0.99, 0.0)
        )
        self.assertGreater(float(cloned[32, 126, 0]), 1.5)
        # Burn dims hot region, Dodge boosts it
        burned = apply_brush_dab(canvas, u=0.14, v=0.5, mode="burn", radius_u=0.05, strength=1.0)
        self.assertLess(float(burned[32, 18, 0]), 3.5)
        dodged = apply_brush_dab(canvas, u=0.14, v=0.5, mode="dodge", radius_u=0.05, strength=1.0)
        self.assertGreater(float(dodged[32, 18, 0]), 4.5)
        # Replay resolution-independent strokes (heal, blur, desat)
        replayed = apply_brush_strokes(
            canvas,
            [
                {"mode": "heal", "radius_u": 0.04, "softness": 0.6, "strength": 0.9, "points": [[0.5, 0.5]]},
                {"mode": "blur", "radius_u": 0.04, "softness": 0.6, "strength": 0.8, "points": [[0.5, 0.5]]},
                {"mode": "desat", "radius_u": 0.04, "softness": 0.6, "strength": 1.0, "points": [[0.14, 0.5]]},
            ],
        )
        self.assertEqual(replayed.shape, canvas.shape)
        self.assertTrue(np.all(np.isfinite(replayed)))

        baked_adj = apply_post_adjustments_to_hdri(
            replayed,
            HDRISynthesisParams(width=128, height=64, exposure_ev=1.0, saturation=0.8, azimuth_offset_deg=45.0),
            prof,
        )
        self.assertEqual(baked_adj.shape, (64, 128, 3))
        self.assertTrue(np.all(np.isfinite(baked_adj)))

        # 4c. Disabled PlateViewInput is skipped by projection, hit-test, and multiview analysis
        from hdri_maker.lighting_analyzer import (
            PlateViewInput,
            analyze_multiview_plates,
            hit_test_plate_uv,
            project_multiview_plates_to_latlong,
        )

        v_on = PlateViewInput(pixels=np.full((32, 32, 3), 0.5, dtype=np.float32), fov_h_deg=60.0, cam_azimuth_deg=0.0, enabled=True)
        v_off = PlateViewInput(pixels=np.full((32, 32, 3), 2.0, dtype=np.float32), fov_h_deg=60.0, cam_azimuth_deg=180.0, enabled=False)
        proj_map, proj_cov = project_multiview_plates_to_latlong([v_on, v_off], out_width=128, out_height=64)
        self.assertEqual(float(proj_cov[32, 0]), 0.0)
        self.assertGreater(float(proj_cov[32, 64]), 0.0)
        self.assertIsNone(hit_test_plate_uv([v_on, v_off], 0.0, 0.5))
        self.assertEqual(hit_test_plate_uv([v_on, v_off], 0.5, 0.5), 0)
        mv_prof = analyze_multiview_plates([v_on, v_off])
        self.assertIsNotNone(mv_prof)

        # 5. Solaris distant sun & USD rig export (if hou available)
        try:
            import hou  # noqa: F401
            from hdri_maker.solaris_stage import (
                create_or_update_distant_sun_from_hotspot,
                export_solaris_rig_usd,
                list_scene_cameras,
            )

            _ = list_scene_cameras()
            if prof.hotspots:
                sun = create_or_update_distant_sun_from_hotspot(prof.hotspots[0])
                self.assertIsNotNone(sun)
                usd_out = os.path.join(self.tmpdir.name, "test_rig.usda")
                self.assertTrue(os.path.isfile(export_solaris_rig_usd(usd_out)))
        except ImportError:
            pass

    def test_batch_brackets_exif_ev_sun_and_lookdev_spheres(self) -> None:
        from hdri_maker.hdri_generator import (
            HDRISynthesisParams,
            reconstruct_physical_sun_disk,
            render_lookdev_sphere_preview_srgb,
        )
        from hdri_maker.plate_io import (
            batch_merge_bracket_groups,
            group_bracket_sequences,
            write_exr_32f,
        )
        from hdri_maker.solaris_stage import sync_solaris_dome_live

        # 1. Batch bracket grouping & merging (2 angles x 2 brackets)
        bracket_paths = []
        for ang_idx in range(2):
            for ev_idx, val in enumerate((0.15, 0.75)):
                arr = np.full((32, 32, 3), val, dtype=np.float32)
                p = os.path.join(self.tmpdir.name, f"ang{ang_idx}_b{ev_idx}.exr")
                write_exr_32f(p, arr, output_colorspace="Linear Rec.709")
                bracket_paths.append(p)
        groups = group_bracket_sequences(bracket_paths, frames_per_angle=2)
        self.assertEqual(len(groups), 2)
        merged_exrs = batch_merge_bracket_groups(groups, out_dir=self.tmpdir.name)
        self.assertEqual(len(merged_exrs), 2)
        for mp in merged_exrs:
            self.assertTrue(os.path.isfile(mp))

        # 2. Physical sun disk reconstruction
        sky = np.full((64, 128, 3), 0.25, dtype=np.float32)
        sky[16, 64] = [1.0, 0.95, 0.85]
        reconstructed = reconstruct_physical_sun_disk(sky, strength=1.0)
        self.assertGreater(float(np.max(reconstructed)), 50.0)
        self.assertEqual(HDRISynthesisParams(sun_reconstruct=1.0).sun_reconstruct, 1.0)

        # 3. Lookdev Sphere Preview (Chrome + 18% Gray ball)
        spheres = render_lookdev_sphere_preview_srgb(reconstructed, width=256, height=128)
        self.assertEqual(spheres.shape, (128, 256, 3))
        self.assertEqual(spheres.dtype, np.uint8)
        self.assertGreater(int(np.max(spheres)), 20)

        # 4. Solaris Live Sync helper (graceful False outside Houdini or when no dome exists)
        res = sync_solaris_dome_live(None, yaw_deg=45.0, exposure_ev=0.5)
        self.assertIsInstance(res, bool)

    def test_equirectangular_360_brackets_passthrough(self) -> None:
        from hdri_maker.hdri_generator import HDRISynthesisParams, generate_latlong_hdri
        from hdri_maker.lighting_analyzer import (
            PlateViewInput,
            analyze_plate,
            project_multiview_plates_to_latlong,
        )
        from hdri_maker.plate_io import (
            is_equirectangular_image,
            linear_to_srgb,
            merge_ldr_brackets_to_hdr,
        )

        # 1. Aspect-ratio & OIIO metadata detection for 2:1 360° Lat/Long vs non-2:1 plate on disk
        self.assertTrue(is_equirectangular_image("", 2048, 1024))
        self.assertFalse(is_equirectangular_image("", 1920, 1080))
        from PIL import Image as _PILImage

        non_eq_path = os.path.join(self.tmpdir.name, "non_eq_plate.jpg")
        _PILImage.fromarray(np.full((48, 64, 3), 128, dtype=np.uint8), mode="RGB").save(non_eq_path)
        self.assertFalse(is_equirectangular_image(non_eq_path, 64, 48))

        # 2. Build a synthetic night/HDR 2:1 360° scene with distinct left/right/top/bottom colors
        h, w = 64, 128
        true_hdr = np.full((h, w, 3), 0.02, dtype=np.float32)
        true_hdr[:16, :, :] = [0.05, 0.08, 0.20]          # zenith: deep blue
        true_hdr[48:, :, :] = [0.12, 0.09, 0.06]          # nadir: warm ground
        true_hdr[24:40, 16:32, :] = [0.10, 0.40, 1.80]    # left quadrant: bright blue sign
        true_hdr[24:40, 88:104, :] = [3.50, 1.20, 0.20]   # right quadrant: bright warm lamp

        # Simulate 3 LDR JPG brackets (-2 EV, 0 EV, +2 EV) without EXIF
        from PIL import Image

        bracket_paths = []
        for idx, ev in enumerate((-2.0, 0.0, 2.0)):
            exposed = np.clip(true_hdr * (2.0 ** ev), 0.0, 1.0)
            srgb_u8 = np.clip(np.round(linear_to_srgb(exposed) * 255.0), 0, 255).astype(np.uint8)
            p = os.path.join(self.tmpdir.name, f"pano_360_ev{idx}.jpg")
            Image.fromarray(srgb_u8, mode="RGB").save(p, quality=98)
            bracket_paths.append(p)

        merged = merge_ldr_brackets_to_hdr(bracket_paths, colorspace="sRGB")
        self.assertEqual(merged.shape, (h, w, 3))
        self.assertGreater(float(np.max(merged)), 1.5)

        # 3. Equirectangular pass-through projection: full coverage, no 3x mirroring
        view = PlateViewInput(
            pixels=merged,
            is_hdr=True,
            source_path=bracket_paths[1],
            lens_type="equirectangular",
        )
        proj, cov = project_multiview_plates_to_latlong([view], out_width=w, out_height=h)
        self.assertAlmostEqual(float(np.min(cov)), 1.0, places=5)
        self.assertLess(float(np.max(np.abs(proj - merged))), 0.05)

        # 4. generate_latlong_hdri preserves left/right asymmetry without mirroring/tiling
        prof = analyze_plate(merged, is_hdr=True)
        out_hdri = generate_latlong_hdri(
            prof,
            HDRISynthesisParams(width=w, height=h, plate_projection_blend=1.0),
            views=[view],
        )
        left_patch = out_hdri[28:36, 20:28]
        right_patch = out_hdri[28:36, 92:100]
        # Left patch must remain blue-dominant, right patch red/warm-dominant (not mirrored!)
        self.assertGreater(float(np.mean(left_patch[..., 2])), float(np.mean(left_patch[..., 0])) * 2.0)
        self.assertGreater(float(np.mean(right_patch[..., 0])), float(np.mean(right_patch[..., 2])) * 2.0)

        # 5. Peak White Target lifts clipped highlight cores to exact target (e.g. 5000) without affecting midtones
        from hdri_maker.hdri_generator import lift_peak_whites

        night = np.full((32, 64, 3), 0.05, dtype=np.float32)
        night[10:14, 10:14] = [0.6, 0.8, 1.0]        # mid-bright blue wall (untouched)
        night[16:18, 30:32] = [21.0, 46.5, 41.9]     # clipped lamp core (~46.5 peak)
        lifted = lift_peak_whites(night, target_peak=5000.0, knee=0.5)
        self.assertAlmostEqual(float(np.max(lifted)), 5000.0, places=2)
        # Midtones & walls below 46.5 * 0.5 = 23.25 remain 100% untouched
        self.assertAlmostEqual(float(lifted[0, 0, 0]), 0.05, places=6)
        self.assertAlmostEqual(float(lifted[12, 12, 2]), 1.0, places=6)
        # Chromaticity ratio on peak pixel is preserved
        self.assertAlmostEqual(float(lifted[16, 30, 0] / lifted[16, 30, 1]), 21.0 / 46.5, places=4)

    def test_pro_hdri_and_image_upgrades(self) -> None:
        from PIL import Image
        from hdri_maker.hdri_generator import (
            apply_brush_dab,
            apply_pole_rectilinear_patch,
            compute_gray_ball_normalize_ev,
            compute_hdri_stats,
            extract_pole_rectilinear_patch,
            render_preview_srgb,
        )
        from hdri_maker.plate_io import (
            convert_colorspace_to_linear_rec709,
            inspect_bracket_paths,
            linear_to_srgb,
            list_available_input_colorspaces,
            merge_ldr_brackets_to_hdr,
        )

        # 1. Camera Log & OCIO input colorspaces
        spaces = list_available_input_colorspaces()
        for req in ("Auto", "sRGB", "ARRI LogC3", "ARRI LogC4", "Sony S-Log3", "RED Log3G10", "Canon Log 2", "ACEScct"):
            self.assertIn(req, spaces)
        mid_log = np.full((8, 8, 3), 0.391, dtype=np.float32)
        for cs in ("ARRI LogC3", "ARRI LogC4", "Sony S-Log3", "RED Log3G10", "Canon Log 2", "ACEScct"):
            lin, name = convert_colorspace_to_linear_rec709(mid_log, cs)
            self.assertEqual(name, cs)
            self.assertEqual(lin.shape, (8, 8, 3))
            self.assertTrue(np.all(np.isfinite(lin)))

        # 2. Pro Bracket Merge: inspect_bracket_paths + align + deghost + crf_calibrate
        h, w = 64, 128
        base = np.full((h, w, 3), 0.18, dtype=np.float32)
        base[20:44, 50:78, :] = [0.6, 0.4, 0.2]
        paths = []
        for idx, ev in enumerate((-2.0, 0.0, 2.0)):
            arr = base.copy()
            if idx == 2:
                # Inject a moving bright ghost object only in the +2 EV bracket
                arr[8:16, 8:16, :] = [0.95, 0.95, 0.95]
            exposed = np.clip(arr * (2.0 ** ev), 0.0, 1.0)
            srgb_u8 = np.clip(np.round(linear_to_srgb(exposed) * 255.0), 0, 255).astype(np.uint8)
            p = os.path.join(self.tmpdir.name, f"pro_b{idx}.jpg")
            Image.fromarray(srgb_u8, mode="RGB").save(p, quality=98)
            paths.append(p)

        info_rows = inspect_bracket_paths(paths, colorspace="sRGB")
        self.assertEqual(len(info_rows), 3)
        merged_dg = merge_ldr_brackets_to_hdr(
            paths,
            ev_offsets=[-2.0, 0.0, 2.0],
            colorspace="sRGB",
            align=True,
            deghost=True,
            crf_calibrate=True,
        )
        self.assertEqual(merged_dg.shape, (h, w, 3))
        # Deghosting should keep the ghosted patch close to base (0.18), not blown out by the moving object
        self.assertLess(float(np.mean(merged_dg[10:14, 10:14])), 0.35)

        # 3. Nuke-style Telemetry + 18% Gray Ball Normalize
        dome = np.full((64, 128, 3), 0.25, dtype=np.float32)
        dome[16, 64] = [5000.0, 5000.0, 5000.0]
        stats = compute_hdri_stats(dome)
        self.assertAlmostEqual(stats["max_lum"], 5000.0, places=1)
        self.assertGreater(stats["dynamic_range_stops"], 10.0)
        norm_ev = compute_gray_ball_normalize_ev(dome, target_gray=0.18)
        self.assertAlmostEqual(norm_ev, 2.0, places=2)
        # Strictly idempotent on repeated calls even with 5000.0 specular spikes
        norm_ev_2 = compute_gray_ball_normalize_ev(dome * float(2.0 ** norm_ev), target_gray=0.18)
        self.assertAlmostEqual(norm_ev_2, 0.0, places=3)

        # 4. Energy Heatmap (5k) preview: 5000+ renders crisp white
        hm = render_preview_srgb(dome, tonemap="Energy Heatmap (5k)")
        self.assertEqual(tuple(int(c) for c in hm[16, 64]), (255, 255, 255))

        # 5. Selective Boost Light Core brush leaves dark surroundings untouched
        canvas = np.full((64, 128, 3), 0.02, dtype=np.float32)
        canvas[30:34, 62:66] = [5.0, 5.0, 5.0]
        boosted = apply_brush_dab(canvas, 0.5, 0.5, mode="boost_light", radius_u=0.08, strength=1.0)
        self.assertGreater(float(boosted[32, 64, 0]), 12.0)
        self.assertAlmostEqual(float(boosted[32, 58, 0]), 0.02, places=3)

        # 6. Nadir / Zenith rectilinear patch extraction & re-projection
        nadir_hdri = np.full((64, 128, 3), 0.1, dtype=np.float32)
        nadir_hdri[56:, :, :] = [0.8, 0.1, 0.1]  # red tripod at nadir
        patch = extract_pole_rectilinear_patch(nadir_hdri, pole="nadir", patch_size=128, fov_deg=110.0)
        self.assertEqual(patch.shape, (128, 128, 3))
        self.assertGreater(float(patch[64, 64, 0]), 0.7)
        cleaned_patch = np.full_like(patch, 0.25)
        reproj = apply_pole_rectilinear_patch(nadir_hdri, cleaned_patch, pole="nadir", fov_deg=110.0)
        self.assertAlmostEqual(float(reproj[63, 64, 0]), 0.25, places=2)
        self.assertAlmostEqual(float(reproj[0, 64, 0]), 0.1, places=4)

    def test_audit_fixes_solaris_color_and_spherical_math(self) -> None:
        from hdri_maker.hdri_generator import (
            extract_and_inpaint_hotspot,
            generate_diffuse_irradiance_hdri,
        )
        from hdri_maker.lighting_analyzer import HotspotLight
        from hdri_maker.plate_io import HAS_OIIO, convert_colorspace_to_linear_rec709, read_plate

        # 1. Sony S-Log3 / RED Log3G10 / Canon Log 2 apply wide-gamut -> Rec.709 matrices
        # Pure wide-gamut red (0.7, 0.2, 0.2) should produce negative/clamped or suppressed G/B relative to R
        red_log = np.full((4, 4, 3), [0.65, 0.35, 0.35], dtype=np.float32)
        for cs_name in ("Sony S-Log3", "RED Log3G10", "Canon Log 2"):
            lin_cs, _ = convert_colorspace_to_linear_rec709(red_log, cs_name)
            self.assertGreater(float(lin_cs[0, 0, 0]), float(lin_cs[0, 0, 1]) * 3.0)

        # 2. Linear-space downsampling in read_plate: checkerboard of 0 and 255 sRGB -> ~0.50 linear (not 0.214!)
        if HAS_OIIO:
            import OpenImageIO as oiio

            srgb_chk = np.zeros((64, 64, 3), dtype=np.uint8)
            srgb_chk[:, 0::2] = 255
            chk_path = os.path.join(self.tmpdir.name, "checker_srgb.png")
            out = oiio.ImageOutput.create(chk_path)
            spec = oiio.ImageSpec(64, 64, 3, oiio.UINT8)
            out.open(chk_path, spec)
            out.write_image(srgb_chk)
            out.close()
            pf = read_plate(chk_path, colorspace="sRGB", max_dim=16)
            self.assertGreater(float(np.mean(pf.pixels)), 0.45)

        # 3. RectLight alpha preserves >90% energy & region_blur wraps across u=0/1 seam
        hdri = np.full((64, 128, 3), 0.2, dtype=np.float32)
        hdri[24:40, 0:8] = 5.0
        hdri[24:40, 120:128] = 5.0
        cleaned, light_tex, _ = extract_and_inpaint_hotspot(
            hdri,
            (0.0, 0.375, 0.1, 0.625),
            inpaint_hdri=True,
            shape="rect",
            region_blur=0.5,
        )
        self.assertGreater(float(np.mean(light_tex[..., 3])), 0.90)
        self.assertTrue(np.all(np.isfinite(cleaned)))

        # 4. Solid-angle weighted diffuse irradiance does not over-weight zenith pole
        polar = np.full((64, 128, 3), 0.1, dtype=np.float32)
        polar[0:2, :, :] = 100.0  # tiny solid-angle zenith cap
        diff = generate_diffuse_irradiance_hdri(polar, clamp_percentile=99.9)
        self.assertLess(float(np.mean(diff[32, :])), 1.0)

        # 5. Solaris USD checks: Sun orientation/angle, Key light rig, Dome color temp reset, Macbeth linear Reflectance
        if HAS_HOU:
            import hou
            from hdri_maker.solaris_stage import (
                create_hotspot_light_rig,
                create_or_update_distant_sun_from_hotspot,
                create_or_update_domelight,
                create_or_update_lookdev_and_ground,
            )

            hou.hipFile.clear(suppress_save_prompt=True)
            dome_exr = os.path.join(self.tmpdir.name, "audit_dome.exr")
            write_exr_32f(dome_exr, np.full((32, 64, 3), 0.2, dtype=np.float32))
            prof = analyze_plate(_make_synthetic_plate(), is_hdr=True)
            dome = create_or_update_domelight(dome_exr, profile=prof, enable_color_temp=True)
            self.assertTrue(bool(dome.stage().GetPrimAtPath("/lights/hdri_maker_dome").GetAttribute("inputs:enableColorTemperature").Get()))
            dome = create_or_update_domelight(dome_exr, profile=prof, enable_color_temp=False)
            self.assertFalse(bool(dome.stage().GetPrimAtPath("/lights/hdri_maker_dome").GetAttribute("inputs:enableColorTemperature").Get()))

            hs = HotspotLight(
                index=0, plate_u=0.25, plate_v=0.25,
                azimuth_deg=30.0, elevation_deg=45.0, radius_deg=4.0,
                peak_intensity=25.0, mean_intensity=12.0,
                color_rgb=(1.0, 0.8, 0.6), cct_kelvin=4500.0,
            )
            sun = create_or_update_distant_sun_from_hotspot(hs, parent_node=dome, azimuth_offset_deg=15.0)
            self.assertAlmostEqual(sun.parm("ry").eval(), -45.0, places=3)
            sun_prim = sun.stage().GetPrimAtPath("/lights/hdri_maker_sun")
            self.assertAlmostEqual(float(sun_prim.GetAttribute("inputs:angle").Get()), 8.0, places=3)

            keys = create_hotspot_light_rig([hs], parent_dome=sun)
            key_prim = keys[0].stage().GetPrimAtPath("/lights/hdri_maker_key_1")
            self.assertAlmostEqual(float(key_prim.GetAttribute("inputs:intensity").Get()), 12.0, places=3)
            self.assertAlmostEqual(float(key_prim.GetAttribute("inputs:angle").Get()), 8.0, places=3)

            g_exr = os.path.join(self.tmpdir.name, "audit_g.exr")
            go_exr = os.path.join(self.tmpdir.name, "audit_go.exr")
            write_exr_32f(g_exr, np.full((32, 32, 3), 0.2, dtype=np.float32))
            write_exr_32f(go_exr, np.full((32, 32, 3), 1.0, dtype=np.float32))
            ld = create_or_update_lookdev_and_ground(
                g_exr, go_exr, parent_node=keys[0], colorspace="Linear Rec.709", feather_start=0.70
            )
            st = ld.stage()
            neutral5 = st.GetPrimAtPath("/lookdev/tripod/macbeth/chart/patch_21").GetAttribute("primvars:displayColor").Get()[0]
            self.assertAlmostEqual(float(neutral5[0]), 0.200, places=3)
            op_vals = st.GetPrimAtPath("/lookdev/ground_disc").GetAttribute("primvars:displayOpacity").Get()
            # At r_norm = 0.625 (< 0.70 feather_start), opacity must still be 1.0
            self.assertAlmostEqual(float(op_vals[20 * 97]), 1.0, places=4)

    def test_json_save_load_restores_lights_brush_and_baked_edits(self):
        import json
        from hdri_maker.qt_compat import QtWidgets
        from hdri_maker.ui_panel import HDRIMakerWidget
        from hdri_maker.plate_io import write_exr_32f

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        plate_path = os.path.join(self.tmpdir.name, "json_roundtrip_plate.exr")
        write_exr_32f(plate_path, _make_synthetic_plate(128, 256))

        w1 = HDRIMakerWidget(restore_session=False)
        w1.plate_path_edit.setText(plate_path)
        w1.input_projection_combo.setCurrentText("360° Equirectangular (Lat/Long)")
        w1.yaw_spin.setValue(35.0)
        w1.vshift_spin.setValue(-4.5)
        w1.ev_spin.setValue(0.75)
        w1.light_shape_combo.setCurrentText("Circle")
        w1.light_res_combo.setCurrentText("1024 x 1024")
        w1.light_blur_slider.setValue(0.65)
        w1.remove_hotspot_chk.setChecked(False)
        w1.brush_mode_combo.setCurrentText("Dodge (Boost HDR)")
        w1.brush_size_slider.setValue(42)
        w1.brush_strength_slider.setValue(70)
        w1.brush_soft_slider.setValue(50)
        w1._on_analyze_clicked()

        # Add light #1 and brush stroke #1, then Bake Edits
        w1._sky_rect = (0.1, 0.05, 0.3, 0.2)
        w1._light_regions = [{"rect": (0.20, 0.20, 0.35, 0.35), "shape": "rect"}]
        w1._brush_strokes = [
            {
                "mode": "dodge",
                "radius_u": 0.03,
                "softness": 0.5,
                "strength": 0.7,
                "src_offset": None,
                "points": [[0.5, 0.5], [0.52, 0.5]],
            }
        ]
        w1._on_bake_edits_to_canvas()
        self.assertEqual(len(w1._baked_extracted_lights), 1)
        self.assertEqual(len(w1._baked_ops), 1)

        # Add live unbaked light #2 and brush stroke #2 after baking
        w1._ground_rect = (0.4, 0.7, 0.6, 0.9)
        w1._light_regions = [{"rect": (0.60, 0.25, 0.72, 0.38), "shape": "circle"}]
        w1._brush_strokes = [
            {
                "mode": "heal",
                "radius_u": 0.02,
                "softness": 0.65,
                "strength": 0.85,
                "src_offset": None,
                "points": [[0.25, 0.6]],
            }
        ]
        w1._regenerate_preview()
        self.assertEqual(len(w1._extracted_lights), 2)

        json_path = os.path.join(self.tmpdir.name, "saved_state.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(w1.get_state_dict(), f, indent=2)
        w1.close()

        with open(json_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        w2 = HDRIMakerWidget(restore_session=False)
        w2.apply_state_dict(loaded, auto_analyze=True)

        self.assertEqual(w2.input_projection_combo.currentText(), "360° Equirectangular (Lat/Long)")
        self.assertAlmostEqual(w2.yaw_spin.value(), 35.0)
        self.assertAlmostEqual(w2.vshift_spin.value(), -4.5)
        self.assertAlmostEqual(w2.ev_spin.value(), 0.75)
        self.assertEqual(w2.light_shape_combo.currentText(), "Circle")
        self.assertEqual(w2.light_res_combo.currentText(), "1024 x 1024")
        self.assertAlmostEqual(w2.light_blur_slider.value(), 0.65)
        self.assertFalse(w2.remove_hotspot_chk.isChecked())
        self.assertEqual(w2.brush_mode_combo.currentText(), "Dodge (Boost HDR)")
        self.assertEqual(w2.brush_size_slider.value(), 42)
        self.assertEqual(w2.brush_strength_slider.value(), 70)
        self.assertEqual(w2.brush_soft_slider.value(), 50)
        self.assertEqual(w2._ground_rect, (0.4, 0.7, 0.6, 0.9))
        self.assertEqual(len(w2._light_regions), 1)
        self.assertEqual(len(w2._brush_strokes), 1)
        self.assertIsNotNone(w2._baked_canvas_hdri)
        self.assertEqual(len(w2._baked_extracted_lights), 1)
        self.assertEqual(len(w2._extracted_lights), 2)
        self.assertEqual(len(w2.preview_label._light_regions), 1)
        self.assertEqual(w2.preview_label._ground_rect, (0.4, 0.7, 0.6, 0.9))
        self.assertIsNotNone(w2._hdri_pixels)
        w2.close()

    def test_polygon_mask_grade_blur_feather_and_json(self):
        import json
        from hdri_maker.hdri_generator import apply_polygon_masks, rasterize_polygon_mask
        from hdri_maker.plate_io import write_exr_32f
        from hdri_maker.qt_compat import QtWidgets
        from hdri_maker.ui_panel import HDRIMakerWidget

        # 1. Pure-numpy polygon mask rasterizer + desaturate + feather + blur + exposure
        img = np.zeros((64, 128, 3), dtype=np.float32)
        img[:, :, 0] = 2.0  # saturated HDR red everywhere
        img[:, :, 1] = 0.2
        img[:, :, 2] = 0.2
        pts = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]

        m_sharp = rasterize_polygon_mask(64, 128, pts, feather=0.0, invert=False)
        self.assertAlmostEqual(float(m_sharp[32, 64, 0]), 1.0, places=4)
        self.assertAlmostEqual(float(m_sharp[4, 4, 0]), 0.0, places=4)

        m_feath = rasterize_polygon_mask(64, 128, pts, feather=0.5, invert=False)
        self.assertGreater(float(m_feath[16, 64, 0]), 0.1)
        self.assertLess(float(m_feath[16, 64, 0]), 0.95)

        desat = apply_polygon_masks(
            img,
            [{"points": pts, "saturation": 0.0, "feather": 0.0, "blur": 0.0, "exposure_ev": 0.0}],
        )
        # Inside polygon: R == G == B (desaturated)
        self.assertAlmostEqual(float(desat[32, 64, 0]), float(desat[32, 64, 1]), places=4)
        # Outside polygon: untouched
        self.assertAlmostEqual(float(desat[4, 4, 0]), 2.0, places=4)

        # Blur test on an impulse inside the polygon
        imp = np.full((64, 128, 3), 0.1, dtype=np.float32)
        imp[32, 64, :] = 10.0
        blurred = apply_polygon_masks(
            imp,
            [{"points": pts, "blur": 0.4, "feather": 0.0}],
        )
        self.assertLess(float(blurred[32, 64, 0]), 5.0)
        self.assertGreater(float(blurred[32, 66, 0]), 0.12)

        # 2. UI dynamic toolbar + JSON roundtrip
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication([])

        plate_path = os.path.join(self.tmpdir.name, "poly_plate.exr")
        write_exr_32f(plate_path, _make_synthetic_plate(128, 256))

        w1 = HDRIMakerWidget(restore_session=False)
        w1.plate_path_edit.setText(plate_path)
        w1._on_analyze_clicked()
        w1.poly_mask_btn.setChecked(True)
        self.assertFalse(w1.poly_toolbar_widget.isHidden())
        w1._on_poly_mask_finished(pts)
        w1._on_desaturate_poly_mask()
        w1.poly_ev_spin.setValue(-1.0)
        w1.poly_feather_spin.setValue(0.25)
        w1.poly_blur_spin.setValue(0.30)
        w1._on_poly_param_changed()

        state = json.loads(json.dumps(w1.get_state_dict()))
        w1.close()

        w2 = HDRIMakerWidget(restore_session=False)
        w2.apply_state_dict(state, auto_analyze=True)
        self.assertEqual(len(w2._poly_masks), 1)
        self.assertAlmostEqual(float(w2._poly_masks[0]["saturation"]), 0.0, places=3)
        self.assertAlmostEqual(float(w2._poly_masks[0]["exposure_ev"]), -1.0, places=3)
        self.assertAlmostEqual(float(w2._poly_masks[0]["feather"]), 0.25, places=3)
        self.assertAlmostEqual(float(w2._poly_masks[0]["blur"]), 0.30, places=3)
        self.assertEqual(len(w2.preview_label._poly_masks), 1)

        # 3. Brush Paint clipped to active Polygon Mask
        w2._on_reset_poly_grade()
        w2.brush_btn.setChecked(True)
        self.assertTrue(w2.brush_clip_poly_chk.isChecked())
        self.assertFalse(w2.poly_toolbar_widget.isHidden())
        self.assertFalse(w2.brush_toolbar_widget.isHidden())
        w2.brush_mode_combo.setCurrentText("Dodge (Boost HDR)")
        w2.brush_size_slider.setValue(120)
        w2.brush_strength_slider.setValue(100)
        pre_outside = w2._hdri_pixels[10, 10].copy()
        pre_inside = w2._hdri_pixels[
            int(0.5 * (w2._hdri_pixels.shape[0] - 1)),
            int(0.5 * (w2._hdri_pixels.shape[1] - 1)),
        ].copy()
        # Paint across the boundary of the polygon mask (0.4,0.4)-(0.6,0.6)
        w2._on_brush_stroke_started(0.5, 0.5)
        w2._on_brush_stroke_moved(0.62, 0.62)
        w2._on_brush_stroke_finished()
        post_inside = w2._hdri_pixels[
            int(0.5 * (w2._hdri_pixels.shape[0] - 1)),
            int(0.5 * (w2._hdri_pixels.shape[1] - 1)),
        ]
        post_outside = w2._hdri_pixels[10, 10]
        outside_near_edge = w2._hdri_pixels[
            int(0.63 * (w2._hdri_pixels.shape[0] - 1)),
            int(0.63 * (w2._hdri_pixels.shape[1] - 1)),
        ]
        self.assertGreater(float(np.mean(post_inside)), float(np.mean(pre_inside)) * 1.2)
        self.assertTrue(np.allclose(post_outside, pre_outside, atol=1e-6))
        self.assertIsNotNone(w2._brush_strokes[-1].get("poly_mask"))
        # Verify outside-near-edge (which is inside brush radius around (0.62,0.62) but outside poly mask) wasn't boosted like inside
        self.assertLess(float(np.mean(outside_near_edge)), float(np.mean(post_inside)))

        # 4. Move polygon vertex, deselect mask, and start a new mask
        w2.poly_mask_btn.setChecked(True)
        moved_pts = [[0.35, 0.35], [0.6, 0.4], [0.6, 0.6], [0.4, 0.6]]
        w2._on_poly_mask_points_changed(0, moved_pts)
        self.assertAlmostEqual(float(w2._poly_masks[0]["points"][0][0]), 0.35, places=4)
        w2._on_poly_mask_selected(-1)
        self.assertEqual(w2._selected_poly_idx, -1)
        self.assertEqual(w2.poly_sel_combo.currentIndex(), -1)
        w2._on_new_poly_mask()
        self.assertTrue(w2.preview_label._poly_force_new)
        self.assertEqual(w2.preview_label._selected_poly_idx, -1)
        w2.close()

    def test_custom_file_dialog_toolbar_and_bookmarks(self) -> None:
        import tempfile
        from hdri_maker.file_dialog import HDRIFileDialog, load_bookmarks_data, parse_filter_string
        from hdri_maker.qt_compat import QtWidgets

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        self.assertIsNotNone(app)

        parsed = parse_filter_string("HDRI Maker State (*.json);;All Files (*)")
        self.assertEqual(parsed[0][1], ["*.json"])
        self.assertEqual(parsed[1][1], [])

        with tempfile.TemporaryDirectory() as tmpdir:
            sub_a = os.path.join(tmpdir, "shot_010")
            sub_b = os.path.join(tmpdir, "shot_020")
            os.makedirs(sub_a, exist_ok=True)
            os.makedirs(sub_b, exist_ok=True)
            bm_json = os.path.join(tmpdir, "bookmarks.json")

            dlg = HDRIFileDialog(
                caption="Save HDRI Maker State",
                initial_path=os.path.join(sub_a, "hdri_maker_state.json"),
                filter_str="HDRI Maker State (*.json);;All Files (*)",
                mode=HDRIFileDialog.MODE_SAVE,
                bookmarks_path=bm_json,
            )
            # Clear labeled top toolbar buttons
            self.assertIn("Back", dlg.btn_back.text())
            self.assertIn("Up", dlg.btn_up.text())
            self.assertIn("New Folder", dlg.btn_new_folder.text())
            self.assertIn("Bookmark", dlg.btn_bookmark.text())
            self.assertEqual(dlg.filename_edit.text(), "hdri_maker_state.json")

            # Bookmark sub_a, navigate to sub_b, then back and verify persistence
            dlg.add_current_bookmark()
            self.assertEqual(dlg.bookmarks_list.count(), 1)
            self.assertTrue(dlg.navigate_to(sub_b))
            self.assertEqual(dlg.current_dir(), os.path.normpath(sub_b))
            dlg.go_back()
            self.assertEqual(dlg.current_dir(), os.path.normpath(sub_a))

            saved = load_bookmarks_data(bm_json)
            self.assertEqual(len(saved["bookmarks"]), 1)
            self.assertEqual(saved["bookmarks"][0]["path"], os.path.normpath(sub_a))

            # Re-open dialog and verify bookmark persists across instances
            dlg2 = HDRIFileDialog(
                caption="Load HDRI Maker State",
                initial_path="",
                filter_str="HDRI Maker State (*.json)",
                mode=HDRIFileDialog.MODE_OPEN_ONE,
                bookmarks_path=bm_json,
            )
            self.assertEqual(dlg2.bookmarks_list.count(), 1)
            dlg2.bookmarks_list.setCurrentRow(0)
            dlg2.remove_selected_bookmark()
            self.assertEqual(dlg2.bookmarks_list.count(), 0)
            dlg.close()
            dlg2.close()

    def test_baked_light_textures_preserved_and_source_plate_protected(self) -> None:
        import json
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from hdri_maker.qt_compat import QtWidgets
        from hdri_maker.ui_panel import HDRIMakerWidget
        from hdri_maker.plate_io import read_plate, write_exr_32f

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as tmpdir:
            src_exr = os.path.join(tmpdir, "source_plate.exr")
            img = np.full((128, 256, 3), 0.2, dtype=np.float32)
            img[50:78, 102:154, :] = 50.0
            write_exr_32f(src_exr, img, output_colorspace="Linear Rec.709")

            w1 = HDRIMakerWidget(restore_session=False)
            w1.plate_path_edit.setText(src_exr)
            w1._on_analyze_clicked()
            w1._light_regions = [{"rect": (0.4, 0.4, 0.6, 0.6), "shape": "rect"}]

            # Guard: exporting to the same path as the loaded input plate must redirect to *_hdri.exr
            w1.out_path_edit.setText(src_exr)
            written_path = w1._on_export_exr()
            self.assertIsNotNone(written_path)
            self.assertNotEqual(os.path.normcase(os.path.abspath(written_path)), os.path.normcase(os.path.abspath(src_exr)))
            self.assertTrue(written_path.endswith("_hdri.exr"))
            # Source plate on disk must still retain the original un-inpainted hotspot
            src_check = read_plate(src_exr, colorspace="Linear Rec.709", max_dim=None)
            self.assertGreater(float(np.max(src_check.pixels)), 40.0)

            # Bake, save JSON, reload into a fresh widget, bake again, and re-export
            w1._on_bake_edits_to_canvas()
            state_json = json.loads(json.dumps(w1.get_state_dict()))
            w1.close()

            w2 = HDRIMakerWidget(restore_session=False)
            w2.apply_state_dict(state_json, auto_analyze=True)
            self.assertEqual(len(w2._baked_extracted_lights), 1)
            w2._on_bake_edits_to_canvas()
            out2 = w2._on_export_exr()
            self.assertIsNotNone(out2)
            self.assertEqual(len(w2._last_light_exr_paths), 1)
            light_check = read_plate(w2._last_light_exr_paths[0], colorspace="Linear Rec.709", max_dim=None)
            self.assertGreater(float(np.max(light_check.pixels)), 40.0)
            self.assertGreater(float(np.mean(light_check.pixels)), 30.0)
            w2.close()
        app.processEvents()

    def test_pole_editor_and_bake_preserve_exposure(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from hdri_maker.qt_compat import QtWidgets
        from hdri_maker.ui_panel import HDRIMakerWidget
        from hdri_maker.plate_io import write_exr_32f

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as tmpdir:
            src_exr = os.path.join(tmpdir, "plate_ev.exr")
            img = np.full((128, 256, 3), 0.25, dtype=np.float32)
            write_exr_32f(src_exr, img, output_colorspace="Linear Rec.709")

            w = HDRIMakerWidget(restore_session=False)
            w.plate_path_edit.setText(src_exr)
            w._on_analyze_clicked()
            w.ev_spin.setValue(1.5)
            w._regenerate_preview()
            mean_before = float(np.mean(w._hdri_pixels))

            # Pole source must remain un-graded so storing it into _baked_canvas_hdri never doubles exposure_ev
            pole_src = w._build_ungraded_pole_source()
            w._baked_canvas_hdri = np.ascontiguousarray(pole_src, dtype=np.float32)
            w._baked_azimuth_offset_deg = float(w.yaw_spin.value())
            w._brush_strokes.append(
                {
                    "mode": "clone",
                    "radius_u": 0.04,
                    "softness": 0.5,
                    "strength": 1.0,
                    "src_offset": [0.2, 0.0],
                    "points": [[0.3, 0.5]],
                }
            )
            w._regenerate_preview()
            mean_after_pole = float(np.mean(w._hdri_pixels))
            self.assertAlmostEqual(mean_after_pole, mean_before, places=3)

            w._on_bake_edits_to_canvas()
            mean_after_bake = float(np.mean(w._hdri_pixels))
            self.assertAlmostEqual(mean_after_bake, mean_before, places=3)
            w.close()
        app.processEvents()

    def test_exposure_math_invariants_and_8k_ui_optimizations(self) -> None:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from hdri_maker.qt_compat import QtWidgets
        from hdri_maker.ui_panel import HDRIMakerWidget
        from hdri_maker.plate_io import read_plate, write_exr_32f

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. A1, A2, A3 invariance tests on graded plate + light + polygon mask + sky/ground gain
            src_exr = os.path.join(tmpdir, "grade_plate.exr")
            img = np.full((128, 256, 3), 0.30, dtype=np.float32)
            img[52:76, 108:148, :] = 50.0
            write_exr_32f(src_exr, img, output_colorspace="Linear Rec.709")

            w = HDRIMakerWidget(restore_session=False)
            w.plate_path_edit.setText(src_exr)
            w._on_analyze_clicked()
            w.ev_spin.setValue(-2.0)
            w.sky_gain_spin.setValue(2.0)
            w.gnd_gain_spin.setValue(1.6)
            w._light_regions = [{"rect": (0.4, 0.38, 0.6, 0.62), "shape": "rect"}]
            w._poly_masks = [
                {
                    "points": [[0.05, 0.10], [0.25, 0.10], [0.25, 0.40], [0.05, 0.40]],
                    "enabled": True,
                    "invert": True,
                    "feather": 0.0,
                    "blur": 0.0,
                    "exposure_ev": 0.0,
                    "saturation": 1.0,
                    "gamma": 0.5,
                    "temp_shift": 0.0,
                    "gain_r": 1.0,
                    "gain_g": 1.0,
                    "gain_b": 1.0,
                }
            ]
            w._regenerate_preview()

            pre_light_mean = float(np.mean(w._extracted_lights[0][0]))
            pre_bake_mean = float(np.mean(w._hdri_pixels))
            self.assertGreater(pre_light_mean, 0.1)

            # Exercise B2/B3 fast polygon cache & dirty-row brush preview before Bake
            w._poly_masks[0]["gamma"] = 0.5
            w._refresh_from_poly_cache()
            w._on_brush_stroke_started(0.75, 0.5)
            w._on_brush_stroke_moved(0.80, 0.5)
            self.assertIsNotNone(w._cached_preview_srgb)
            cached_copy = w._cached_preview_srgb.copy()
            w._on_brush_stroke_finished()
            self.assertTrue(np.array_equal(cached_copy, w._cached_preview_srgb))
            pre_bake_after_brush = float(np.mean(w._hdri_pixels))

            w._on_bake_edits_to_canvas()
            post_light_mean = float(np.mean(w._baked_extracted_lights[0][0]))
            post_bake_mean = float(np.mean(w._hdri_pixels))
            self.assertAlmostEqual(post_light_mean / pre_light_mean, 1.0, delta=0.02)
            self.assertAlmostEqual(post_bake_mean / pre_bake_after_brush, 1.0, delta=0.01)
            w._regenerate_preview(fast=True)
            self.assertGreater(len(w._baked_mip_cache), 0)
            w.close()

            # 2. A5 & A6: 8K equirectangular sun flux conservation + native 8K detail preservation post-bake
            exr_8k = os.path.join(tmpdir, "env_8k.exr")
            h8, w8 = 4096, 8192
            env = np.full((h8, w8, 3), 0.20, dtype=np.float32)
            # High-frequency alternating 1-pixel columns in an equatorial patch + bright 4x4 sun core
            yy, xx = np.meshgrid(np.arange(1900, 2100), np.arange(3800, 4000), indexing="ij")
            env[1900:2100, 3800:4000, :] = np.where((xx + yy) % 2 == 0, 0.80, 0.10)[..., None]
            env[1020:1024, 2044:2048, :] = 400.0
            write_exr_32f(exr_8k, env, output_colorspace="Linear Rec.709", compression="zips")

            w8k = HDRIMakerWidget(restore_session=False)
            w8k.plate_path_edit.setText(exr_8k)
            idx_8k = w8k.res_combo.findText("8192 x 4096")
            if idx_8k >= 0:
                w8k.res_combo.setCurrentIndex(idx_8k)
            w8k._on_analyze_clicked()

            src_mean_lum = float(np.mean(0.2126 * env[..., 0] + 0.7152 * env[..., 1] + 0.0722 * env[..., 2]))
            prev_mean_lum = float(
                np.mean(0.2126 * w8k._hdri_pixels[..., 0] + 0.7152 * w8k._hdri_pixels[..., 1] + 0.0722 * w8k._hdri_pixels[..., 2])
            )
            self.assertAlmostEqual(prev_mean_lum / src_mean_lum, 1.0, delta=0.01)

            w8k._brush_strokes.append(
                {
                    "mode": "clone",
                    "radius_u": 0.02,
                    "softness": 0.5,
                    "strength": 1.0,
                    "src_offset": [0.2, 0.0],
                    "points": [[0.1, 0.5]],
                }
            )
            w8k._on_bake_edits_to_canvas()
            out_8k = os.path.join(tmpdir, "out_8k.exr")
            w8k.out_path_edit.setText(out_8k)
            written_8k = w8k._on_export_exr()
            self.assertIsNotNone(written_8k)
            rep = read_plate(written_8k, colorspace="Linear Rec.709", max_dim=None)
            self.assertEqual(rep.pixels.shape[:2], (4096, 8192))
            # Native 8K replay must preserve the crisp 1-pixel checkerboard (upsampled 4K would have ~0 std)
            patch_std = float(np.std(rep.pixels[1900:2100, 3800:4000, 0]))
            self.assertGreater(patch_std, 0.20)
            w8k.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main(verbosity=2)
