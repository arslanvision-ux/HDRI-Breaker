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
        self.assertTrue(g_mtl.GetAttribute("outputs:surface").IsValid())
        self.assertTrue(g_mtl.GetAttribute("outputs:mtlx:surface").IsValid())
        self.assertTrue(g_mtl.GetAttribute("outputs:Redshift:surface").IsValid())
        rs_std = stage.GetPrimAtPath("/lookdev/materials/ground_proj_mtl/ground_rs_std")
        self.assertTrue(rs_std.IsValid())
        self.assertAlmostEqual(float(rs_std.GetAttribute("inputs:refl_weight").Get()), 0.0, places=5)
        self.assertAlmostEqual(float(rs_std.GetAttribute("inputs:refl_roughness").Get()), 1.0, places=5)
        # Single unique macbeth_mtl shader instead of 24 per-patch materials
        mb_mtl = stage.GetPrimAtPath("/lookdev/materials/macbeth_mtl")
        self.assertTrue(mb_mtl.IsValid())
        self.assertTrue(mb_mtl.GetAttribute("outputs:mtlx:surface").IsValid())
        self.assertTrue(mb_mtl.GetAttribute("outputs:Redshift:surface").IsValid())
        self.assertFalse(stage.GetPrimAtPath("/lookdev/materials/macbeth_00").IsValid())

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
        state = widget.get_state_dict()
        widget2 = HDRIMakerWidget(restore_session=False)
        widget2.apply_state_dict(state, auto_analyze=False)
        self.assertAlmostEqual(widget2.yaw_spin.value(), 45.0)
        self.assertAlmostEqual(widget2.vshift_spin.value(), -10.0)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
