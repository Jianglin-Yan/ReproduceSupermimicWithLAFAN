"""Render recorded rollouts (scripts/media/record_rollout.py) with MuJoCo offscreen: solid robot = policy in
simulation, translucent blue ghost = reference motion. Writes one mp4 per clip and optionally a concatenated
highlight; GIF conversion is left to ffmpeg (see README).

  MUJOCO_GL=glfw uv run python scripts/media/render_rollout.py media/rollouts.npz media/out --fps 25 --seconds 6
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "glfw")
import mujoco
import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

GHOST_RGBA = "0.25 0.50 1.0 0.22"


def two_robot_xml(mjcf: Path, root_body: str) -> str:
    xml = mjcf.read_text()
    xml = re.sub(r'meshdir="([^"]*)"', lambda m: f'meshdir="{(mjcf.parent / m.group(1)).resolve()}"', xml, count=1)
    # strip anything that is not needed for drawing (actuators / sensors / keyframes keep working but are unused)
    body = xml[xml.index(f'<body name="{root_body}"') : xml.rindex("</worldbody>")]
    ghost = re.sub(r' name="([^"]+)"', r' name="ref_\1"', body)
    # keep only the visual mesh geoms of the ghost, visual-only and translucent
    ghost = re.sub(r"<geom\b[^>]*/>", lambda m: "" if "mesh=" not in m.group(0) else m.group(0), ghost)
    ghost = re.sub(r' (contype|conaffinity|group|material|rgba|class)="[^"]*"', "", ghost)
    ghost = ghost.replace("<geom ", f'<geom type="mesh" contype="0" conaffinity="0" group="1" rgba="{GHOST_RGBA}" ')
    ghost = re.sub(r"<site[^>]*/>", "", ghost)
    ghost = re.sub(r"<freejoint[^>]*/>", '<freejoint name="ref_root"/>', ghost, count=1)
    scene = """
    <light name="top" pos="0 0 5" dir="0 0 -1" directional="true" diffuse="0.45 0.45 0.45" specular="0.05 0.05 0.05" castshadow="true"/>

    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="0" conaffinity="0"/>
    """
    xml = xml.replace("</worldbody>", ghost + scene + "</worldbody>")
    assets = """
    <texture name="grid" type="2d" builtin="checker" rgb1="0.90 0.91 0.93" rgb2="0.76 0.78 0.82" width="512" height="512" mark="edge" markrgb="0.68 0.70 0.74"/>
    <material name="grid" texture="grid" texrepeat="2 2" texuniform="true" reflectance="0.0" specular="0.05" shininess="0.05"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.97 0.97 0.98" rgb2="0.80 0.84 0.90" width="256" height="256"/>
    """
    xml = xml.replace("<asset>", "<asset>" + assets, 1)
    xml = xml.replace("<mujoco", '<mujoco', 1)
    visual = '<visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096" offsamples="8"/><map znear="0.01"/><headlight ambient="0.25 0.25 0.25" diffuse="0.35 0.35 0.35" specular="0.15 0.15 0.15"/></visual>\n'
    xml = re.sub(r"<visual>.*?</visual>", "", xml, flags=re.S)
    xml = xml.replace("<asset>", visual + "  <asset>", 1)
    # actuators/sensors/keyframes reference names that still exist; keep the file valid by removing them anyway
    xml = re.sub(r"<actuator>.*?</actuator>", "", xml, flags=re.S)
    xml = re.sub(r"<sensor>.*?</sensor>", "", xml, flags=re.S)
    xml = re.sub(r"<keyframe>.*?</keyframe>", "", xml, flags=re.S)
    return xml


def draw_label(img: np.ndarray, text: str, sub: str, font, font_small) -> np.ndarray:
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im, "RGBA")
    pad = 14
    w = max(dr.textlength(text, font=font), dr.textlength(sub, font=font_small)) + 2 * pad
    dr.rounded_rectangle([16, 16, 16 + w, 16 + 2 * pad + 52], radius=10, fill=(20, 24, 32, 150))
    dr.text((16 + pad, 16 + pad - 2), text, font=font, fill=(255, 255, 255, 255))
    dr.text((16 + pad, 16 + pad + 28), sub, font=font_small, fill=(210, 220, 235, 255))
    return np.asarray(im)


def pretty(name: str) -> str:
    base = name.split("__")[0]
    base = re.sub(r"_(R|L|M)$", "", base)
    base = re.sub(r"_\d{3}$", "", base)
    return base.replace("_", " ")


def main(traj: Path, out_dir: Path, fps: int = 25, seconds: float | None = None, width: int = 1280, height: int = 720,
         azimuth: float = 140.0, elevation: float = -14.0, distance: float = 2.7, ghost: bool = True, label: bool = True, clips: list[str] | None = None,
         concat: Path | None = None, segments: list[str] | None = None, titles: list[str] | None = None) -> None:
    """``segments``: per-clip time windows as ``name:start:end`` (seconds); ``titles``: ``name=Display title`` overrides."""

    model = mujoco.MjModel.from_xml_string(two_robot_xml(Path(G1_XML), "pelvis"))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[1] = 1  # ghost
    opt.geomgroup[2] = 1  # visual meshes
    opt.geomgroup[0] = 1  # floor
    opt.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = 1
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except OSError:
        font = font_small = ImageFont.load_default()

    d = dict(np.load(traj))
    names = [str(n) for n in d["names"]]
    nq_robot = d["robot_qpos"].shape[-1]
    assert model.nq == 2 * nq_robot, (model.nq, nq_robot)
    out_dir.mkdir(parents=True, exist_ok=True)
    stride = max(1, int(round(50 / fps)))
    outputs = []
    seg = {x.split(":")[0]: (float(x.split(":")[1]), float(x.split(":")[2])) for x in (segments or [])}
    ttl = {x.split("=", 1)[0]: x.split("=", 1)[1] for x in (titles or [])}
    order = [n for n in (clips or names) if n in names]
    for name in order:
        i = names.index(name)
        T = int(d["lengths"][i])
        t0, t1 = seg.get(name, (0.0, seconds if seconds is not None else T / 50))
        a, b = int(t0 * 50), min(T, int(t1 * 50) + 1)
        qr, qf = d["robot_qpos"][i, a:b], d["ref_qpos"][i, a:b]
        T = b - a
        out = out_dir / f"{name}.mp4"
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
                               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
        # smooth camera: follow a low-passed root position
        target = qr[0, :3].copy()
        for t in range(0, T, stride):
            data.qpos[:nq_robot] = qr[t]
            data.qpos[nq_robot:] = qf[t]
            if not ghost:
                data.qpos[nq_robot:nq_robot + 3] = [0, 0, -5]
            mujoco.mj_forward(model, data)
            target = 0.92 * target + 0.08 * qr[t, :3]
            cam.lookat[:] = [target[0], target[1], 0.65]
            cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
            renderer.update_scene(data, cam, opt)
            img = renderer.render()
            if label:
                img = draw_label(img, ttl.get(name, pretty(name)), f"normalized return {float(d['norm_return'][i]):.2f}   t = {(a + t)/50:4.1f} s", font, font_small)
            ff.stdin.write(np.ascontiguousarray(img).tobytes())
        ff.stdin.close(); ff.wait()
        outputs.append(out)
        print("wrote", out, f"{T/50:.1f} s")
    if concat and outputs:
        lst = out_dir / "concat.txt"
        lst.write_text("".join(f"file '{p.resolve()}'\n" for p in outputs))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(concat)], check=True)
        print("wrote", concat)


if __name__ == "__main__":
    tyro.cli(main)

# Maintainer: Jianglin Yan | 2026-10-28
