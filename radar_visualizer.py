"""
Real-Time Dual Radar CSV Visualizer — PPI playback for Furuno-format and simulator CSVs.

Optimized for large CSVs (868+ bins, 1820+ spokes per rotation):
  - Numpy-vectorized spoke rendering (no per-pixel set_at)
  - Cached max_ticks (computed once on load, not per spoke)
  - Pre-allocated fade surfaces (no allocation churn)
  - Loading indicator (prevents "Not Responding" on large files)
  - Only renders the active view mode PPIs
  - Spoke-per-frame cap to prevent frame stalls at high speed
"""

import os
import sys
import math
import pygame
import numpy as np
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# CSV Loader
# ---------------------------------------------------------------------------

class RadarCSVData:
    """Loads a radar CSV file and splits it into rotations of spokes."""

    def __init__(self):
        self.rotations: List[List[dict]] = []
        self.filename: str = ""
        self.gain: int = 0
        self.range_index: int = 0
        self.num_bins: int = 0
        self.max_ticks: int = 8192  # cached — computed once on load

    @staticmethod
    def load(filepath: str) -> "RadarCSVData":
        data = RadarCSVData()
        data.filename = os.path.basename(filepath)

        spokes: List[dict] = []
        peak_tick = 0
        header_skipped = False

        with open(filepath, "r") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if not header_skipped:
                    if line[0].isalpha():
                        header_skipped = True
                        continue
                    header_skipped = True

                parts = line.split(",")
                if len(parts) < 6:
                    continue

                try:
                    range_idx = int(float(parts[2]))
                    gain = int(float(parts[3]))
                    angle_tick = int(float(parts[4]))
                except (ValueError, IndexError):
                    continue

                # Bulk-parse echo values with numpy
                raw_echoes = parts[5:]
                try:
                    float_vals = np.array(
                        [float(v) for v in raw_echoes if v.strip()],
                        dtype=np.float32)
                except ValueError:
                    continue

                if len(float_vals) == 0:
                    continue

                # Detect float 0.0-1.0 format vs integer 0-252 format
                if float_vals.max() <= 1.0 and '.' in raw_echoes[0]:
                    echoes = (float_vals * 252).astype(np.int16)
                else:
                    echoes = float_vals.astype(np.int16)

                data.gain = gain
                data.range_index = range_idx
                data.num_bins = max(data.num_bins, len(echoes))
                if angle_tick > peak_tick:
                    peak_tick = angle_tick

                spokes.append({"angle_tick": angle_tick, "echoes": echoes})

        if not spokes:
            return data

        # Cache max_ticks once
        data.max_ticks = 8192 if peak_tick > 1000 else max(peak_tick + 10, 360)

        # Split spokes into rotations at angle wraparound points
        rotations: List[List[dict]] = []
        current_rot: List[dict] = [spokes[0]]
        for i in range(1, len(spokes)):
            if spokes[i]["angle_tick"] < spokes[i - 1]["angle_tick"] - 50:
                rotations.append(current_rot)
                current_rot = []
            current_rot.append(spokes[i])
        if current_rot:
            rotations.append(current_rot)

        data.rotations = rotations
        return data


# ---------------------------------------------------------------------------
# PPI Renderer (numpy-vectorized)
# ---------------------------------------------------------------------------

class PPIRenderer:
    """Renders a single PPI display (echo painting, grid, sweep line)."""

    def __init__(self, center_x: int, center_y: int, radius: int, color_scheme: str):
        self.cx = center_x
        self.cy = center_y
        self.radius = radius
        self.color_scheme = color_scheme  # 'green', 'amber'

        size = radius * 2 + 2
        self._size = size
        self.echo_surface = pygame.Surface((size, size), pygame.SRCALPHA)
        self.grid_surface = pygame.Surface((size, size), pygame.SRCALPHA)
        self.sweep_surface = pygame.Surface((size, size), pygame.SRCALPHA)
        self.last_bearing = 0.0

        # Pre-allocate fade surface (reused every frame — no allocation churn)
        self._fade_surface = pygame.Surface((size, size), pygame.SRCALPHA)
        self._fade_surface.fill((0, 0, 0, 4))

        self._draw_grid()

    # -- grid ---------------------------------------------------------------

    def _draw_grid(self):
        self.grid_surface.fill((0, 0, 0, 0))
        r = self.radius
        cx = cy = r + 1

        ring_color = (0, 80, 0, 160) if self.color_scheme == "green" else (80, 60, 0, 160)
        label_color = (0, 160, 0) if self.color_scheme == "green" else (160, 120, 0)

        for i in range(1, 5):
            ring_r = int(r * i / 4)
            pygame.draw.circle(self.grid_surface, ring_color, (cx, cy), ring_r, 1)

        for deg in range(0, 360, 45):
            rad = math.radians(deg - 90)
            ex = cx + int(r * math.cos(rad))
            ey = cy + int(r * math.sin(rad))
            pygame.draw.line(self.grid_surface, ring_color, (cx, cy), (ex, ey), 1)

        font = pygame.font.Font(None, 22)
        for label, deg in [("N", 0), ("E", 90), ("S", 180), ("W", 270)]:
            rad = math.radians(deg - 90)
            lx = cx + int((r + 14) * math.cos(rad))
            ly = cy + int((r + 14) * math.sin(rad))
            surf = font.render(label, True, label_color)
            rect = surf.get_rect(center=(lx, ly))
            self.grid_surface.blit(surf, rect)

    # -- echo painting (vectorized) -----------------------------------------

    def reset(self):
        self.echo_surface.fill((0, 0, 0, 0))

    def paint_spoke(self, angle_tick: int, echoes: np.ndarray, max_ticks: int = 8192):
        """Paint one spoke of echo data using vectorized numpy operations."""
        bearing_deg = angle_tick * 360.0 / max_ticks
        bearing_rad = math.radians(bearing_deg - 90)
        self.last_bearing = bearing_deg

        r = self.radius
        cx = cy = r + 1

        # Find non-zero bins
        nonzero = np.nonzero(echoes > 0)[0]
        if len(nonzero) == 0:
            return

        vals = np.minimum(echoes[nonzero], 252).astype(np.uint8)
        fracs = (nonzero + 1).astype(np.float64) / len(echoes)

        cos_b = math.cos(bearing_rad)
        sin_b = math.sin(bearing_rad)
        pxs = np.clip((cx + r * fracs * cos_b).astype(np.intp), 0, self._size - 1)
        pys = np.clip((cy + r * fracs * sin_b).astype(np.intp), 0, self._size - 1)

        # Batch-write to surface via surfarray
        pix = pygame.surfarray.pixels3d(self.echo_surface)
        if self.color_scheme == "green":
            pix[pxs, pys, 0] = 0
            pix[pxs, pys, 1] = np.maximum(pix[pxs, pys, 1], vals)
            pix[pxs, pys, 2] = 0
        else:  # amber
            pix[pxs, pys, 0] = np.maximum(pix[pxs, pys, 0], vals)
            pix[pxs, pys, 1] = np.maximum(pix[pxs, pys, 1], (vals * 0.78).astype(np.uint8))
            pix[pxs, pys, 2] = 0
        del pix  # release surface lock

        alpha = pygame.surfarray.pixels_alpha(self.echo_surface)
        alpha[pxs, pys] = 220
        del alpha

    def paint_spoke_overlay(self, angle_tick: int, echoes: np.ndarray,
                            is_file_a: bool, max_ticks: int = 8192):
        """Paint a spoke in overlay mode (additive green/amber)."""
        bearing_deg = angle_tick * 360.0 / max_ticks
        bearing_rad = math.radians(bearing_deg - 90)
        self.last_bearing = bearing_deg

        r = self.radius
        cx = cy = r + 1

        nonzero = np.nonzero(echoes > 0)[0]
        if len(nonzero) == 0:
            return

        vals = np.minimum(echoes[nonzero], 252).astype(np.int16)
        fracs = (nonzero + 1).astype(np.float64) / len(echoes)

        cos_b = math.cos(bearing_rad)
        sin_b = math.sin(bearing_rad)
        pxs = np.clip((cx + r * fracs * cos_b).astype(np.intp), 0, self._size - 1)
        pys = np.clip((cy + r * fracs * sin_b).astype(np.intp), 0, self._size - 1)

        pix = pygame.surfarray.pixels3d(self.echo_surface)
        if is_file_a:
            pix[pxs, pys, 1] = np.minimum(
                pix[pxs, pys, 1].astype(np.int16) + vals, 255).astype(np.uint8)
        else:
            pix[pxs, pys, 0] = np.minimum(
                pix[pxs, pys, 0].astype(np.int16) + vals, 255).astype(np.uint8)
            pix[pxs, pys, 1] = np.minimum(
                pix[pxs, pys, 1].astype(np.int16) + (vals * 0.78).astype(np.int16),
                255).astype(np.uint8)
        del pix

        alpha = pygame.surfarray.pixels_alpha(self.echo_surface)
        alpha[pxs, pys] = 220
        del alpha

    # -- fade / sweep -------------------------------------------------------

    def fade(self, amount: int = 4):
        """Apply persistence fade using pre-allocated surface."""
        self.echo_surface.blit(self._fade_surface, (0, 0),
                               special_flags=pygame.BLEND_RGBA_SUB)

    def _draw_sweep(self):
        self.sweep_surface.fill((0, 0, 0, 0))
        r = self.radius
        cx = cy = r + 1
        rad = math.radians(self.last_bearing - 90)
        ex = cx + int(r * math.cos(rad))
        ey = cy + int(r * math.sin(rad))

        sweep_color = (0, 255, 0) if self.color_scheme == "green" else (255, 200, 0)
        pygame.draw.line(self.sweep_surface, (*sweep_color, 180), (cx, cy), (ex, ey), 2)

    # -- composite ----------------------------------------------------------

    def draw(self, target_surface: pygame.Surface):
        """Blit grid + echoes + sweep onto the target at the configured center."""
        ox = self.cx - self.radius - 1
        oy = self.cy - self.radius - 1

        target_surface.blit(self.grid_surface, (ox, oy))
        target_surface.blit(self.echo_surface, (ox, oy))
        self._draw_sweep()
        target_surface.blit(self.sweep_surface, (ox, oy))


# ---------------------------------------------------------------------------
# Playback Engine
# ---------------------------------------------------------------------------

MAX_SPOKES_PER_FRAME = 60  # cap to prevent frame stalls at high speed


class PlaybackEngine:
    """Drives spoke-by-spoke playback of one or two RadarCSVData files."""

    def __init__(self):
        self.data_a: Optional[RadarCSVData] = None
        self.data_b: Optional[RadarCSVData] = None
        self.speed: float = 1.0
        self.playing: bool = False
        self.current_rotation: int = 0
        self.current_spoke: int = 0
        self.time_accumulator: float = 0.0

    def _active_data(self) -> Optional[RadarCSVData]:
        return self.data_a if self.data_a and self.data_a.rotations else (
            self.data_b if self.data_b and self.data_b.rotations else None)

    def total_rotations(self) -> int:
        d = self._active_data()
        return len(d.rotations) if d else 0

    def spokes_in_rotation(self) -> int:
        d = self._active_data()
        if not d or self.current_rotation >= len(d.rotations):
            return 0
        return len(d.rotations[self.current_rotation])

    def update(self, dt: float):
        """Advance playback by dt seconds.  Returns (spokes_a, spokes_b)."""
        spokes_a: List[dict] = []
        spokes_b: List[dict] = []

        primary = self._active_data()
        if not primary:
            return spokes_a, spokes_b

        rot_count = len(primary.rotations)
        if rot_count == 0:
            return spokes_a, spokes_b

        if self.current_rotation >= rot_count:
            self.current_rotation = 0
            self.current_spoke = 0

        cur_rot = primary.rotations[self.current_rotation]
        num_spokes = len(cur_rot)
        if num_spokes == 0:
            return spokes_a, spokes_b

        spoke_interval = 2.5 / num_spokes
        self.time_accumulator += dt * self.speed

        emitted = 0
        while self.time_accumulator >= spoke_interval and emitted < MAX_SPOKES_PER_FRAME:
            self.time_accumulator -= spoke_interval
            emitted += 1

            # Emit spoke from A (use cached max_ticks)
            if self.data_a and self.data_a.rotations:
                if self.current_rotation < len(self.data_a.rotations):
                    rot_a = self.data_a.rotations[self.current_rotation]
                    if self.current_spoke < len(rot_a):
                        sp = rot_a[self.current_spoke]
                        spokes_a.append(sp)

            # Emit spoke from B
            if self.data_b and self.data_b.rotations:
                if self.current_rotation < len(self.data_b.rotations):
                    rot_b = self.data_b.rotations[self.current_rotation]
                    if self.current_spoke < len(rot_b):
                        sp = rot_b[self.current_spoke]
                        spokes_b.append(sp)

            self.current_spoke += 1
            if self.current_spoke >= num_spokes:
                self.current_spoke = 0
                self.current_rotation += 1
                if self.current_rotation >= rot_count:
                    self.current_rotation = 0
                cur_rot = primary.rotations[self.current_rotation]
                num_spokes = len(cur_rot)
                if num_spokes == 0:
                    break
                spoke_interval = 2.5 / num_spokes

        # Drain excess accumulator to prevent catch-up spiral
        if self.time_accumulator > spoke_interval * MAX_SPOKES_PER_FRAME:
            self.time_accumulator = 0.0

        return spokes_a, spokes_b

    def seek_rotation(self, rot_idx: int):
        d = self._active_data()
        if not d:
            return
        self.current_rotation = max(0, min(rot_idx, len(d.rotations) - 1))
        self.current_spoke = 0
        self.time_accumulator = 0.0


# ---------------------------------------------------------------------------
# File dialog helper (tkinter — stdlib)
# ---------------------------------------------------------------------------

def open_file_dialog() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        filepath = filedialog.askopenfilename(
            title="Select Radar CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()
        return filepath if filepath else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

CTRL_BAR_H = 80
WIN_W, WIN_H = 1500, 820
PPI_AREA_H = WIN_H - CTRL_BAR_H


def _btn_rect(x, y, w, h):
    return pygame.Rect(x, y, w, h)


class Button:
    def __init__(self, rect: pygame.Rect, label: str):
        self.rect = rect
        self.label = label
        self.hovered = False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font):
        color = (0, 150, 0) if self.hovered else (0, 100, 0)
        pygame.draw.rect(surface, color, self.rect, border_radius=4)
        pygame.draw.rect(surface, (0, 180, 0), self.rect, 1, border_radius=4)
        txt = font.render(self.label, True, (0, 255, 0))
        tr = txt.get_rect(center=self.rect.center)
        surface.blit(txt, tr)

    def handle(self, pos):
        self.hovered = self.rect.collidepoint(pos)

    def clicked(self, pos):
        return self.rect.collidepoint(pos)


def draw_controls(screen, font, font_sm, playback, view_mode, buttons,
                  file_a_name, file_b_name):
    bar_y = WIN_H - CTRL_BAR_H
    pygame.draw.rect(screen, (0, 20, 0), (0, bar_y, WIN_W, CTRL_BAR_H))
    pygame.draw.line(screen, (0, 80, 0), (0, bar_y), (WIN_W, bar_y), 1)

    col = (0, 220, 0)

    fa = file_a_name if file_a_name else "(none)"
    fb = file_b_name if file_b_name else "(none)"
    screen.blit(font_sm.render(f"A: {fa}", True, col), (10, bar_y + 5))
    screen.blit(font_sm.render(f"B: {fb}", True, (220, 170, 0)), (10, bar_y + 22))

    for b in buttons:
        b.draw(screen, font_sm)

    info_x = 590
    state = "PLAY" if playback.playing else "PAUSE"
    screen.blit(font_sm.render(f"{state}  Speed: {playback.speed:.2f}x", True, col),
                (info_x, bar_y + 5))

    total = playback.total_rotations()
    rot = playback.current_rotation + 1 if total else 0
    spoke = playback.current_spoke
    spokes_total = playback.spokes_in_rotation()
    screen.blit(font_sm.render(f"Rot {rot}/{total}   Spoke {spoke}/{spokes_total}", True, col),
                (info_x, bar_y + 22))

    # Angle display — uses cached max_ticks (O(1), not O(n))
    d = playback._active_data()
    if d and d.rotations and playback.current_rotation < len(d.rotations):
        cur_rot = d.rotations[playback.current_rotation]
        if playback.current_spoke < len(cur_rot):
            tick = cur_rot[playback.current_spoke]["angle_tick"]
            angle_deg = tick * 360.0 / d.max_ticks
            screen.blit(font_sm.render(f"Angle: {angle_deg:.1f} deg", True, col),
                        (info_x, bar_y + 39))

    screen.blit(font_sm.render(f"View: {view_mode.upper()}", True, col), (info_x, bar_y + 56))

    hints = "SPACE=Play/Pause  A/B=Load  1/2/3=View  Arrows=Speed/Rot  ESC=Quit"
    screen.blit(font_sm.render(hints, True, (0, 120, 0)), (820, bar_y + 56))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Radar CSV Visualizer")
    clock = pygame.time.Clock()

    font = pygame.font.Font(None, 26)
    font_sm = pygame.font.Font(None, 20)

    # PPIs
    single_radius = 340
    dual_radius = 260
    ppi_a_single = PPIRenderer(WIN_W // 2, PPI_AREA_H // 2, single_radius, "green")
    ppi_a_dual = PPIRenderer(375, PPI_AREA_H // 2, dual_radius, "green")
    ppi_b_dual = PPIRenderer(1125, PPI_AREA_H // 2, dual_radius, "amber")
    ppi_overlay = PPIRenderer(WIN_W // 2, PPI_AREA_H // 2, single_radius, "green")

    playback = PlaybackEngine()
    view_mode = "single"  # single / dual / overlay

    # Buttons
    bar_y = WIN_H - CTRL_BAR_H
    btn_load_a = Button(_btn_rect(200, bar_y + 8, 70, 24), "Load A")
    btn_load_b = Button(_btn_rect(200, bar_y + 38, 70, 24), "Load B")
    btn_play = Button(_btn_rect(290, bar_y + 8, 70, 24), "Play")
    btn_pause = Button(_btn_rect(290, bar_y + 38, 70, 24), "Pause")
    btn_single = Button(_btn_rect(380, bar_y + 8, 60, 24), "Single")
    btn_dual = Button(_btn_rect(450, bar_y + 8, 50, 24), "Dual")
    btn_overlay = Button(_btn_rect(510, bar_y + 8, 65, 24), "Overlay")
    buttons = [btn_load_a, btn_load_b, btn_play, btn_pause,
               btn_single, btn_dual, btn_overlay]

    file_a_name = ""
    file_b_name = ""
    prompt_font = pygame.font.Font(None, 32)
    running = True

    def show_loading(filename):
        """Show loading indicator so the window doesn't appear frozen."""
        screen.fill((0, 0, 0))
        txt = prompt_font.render(f"Loading {filename}...", True, (0, 200, 0))
        screen.blit(txt, txt.get_rect(center=(WIN_W // 2, WIN_H // 2)))
        pygame.display.flip()
        pygame.event.pump()  # keep Windows from marking "Not Responding"

    def load_file(slot: str):
        nonlocal file_a_name, file_b_name, view_mode
        path = open_file_dialog()
        if not path:
            return

        show_loading(os.path.basename(path))

        data = RadarCSVData.load(path)
        if not data.rotations:
            return
        if slot == "a":
            playback.data_a = data
            file_a_name = data.filename
            ppi_a_single.reset()
            ppi_a_dual.reset()
            ppi_overlay.reset()
            playback.seek_rotation(0)
            playback.playing = True
        else:
            playback.data_b = data
            file_b_name = data.filename
            ppi_b_dual.reset()
            ppi_overlay.reset()
            if view_mode == "single":
                view_mode = "dual"

    while running:
        dt = clock.tick(60) / 1000.0
        mouse_pos = pygame.mouse.get_pos()

        for b in buttons:
            b.handle(mouse_pos)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    playback.playing = not playback.playing
                elif event.key == pygame.K_a:
                    load_file("a")
                elif event.key == pygame.K_b:
                    load_file("b")
                elif event.key == pygame.K_1:
                    view_mode = "single"
                elif event.key == pygame.K_2:
                    view_mode = "dual"
                elif event.key in (pygame.K_3, pygame.K_o):
                    view_mode = "overlay"
                elif event.key == pygame.K_LEFT:
                    playback.speed = max(0.25, playback.speed / 1.5)
                elif event.key == pygame.K_RIGHT:
                    playback.speed = min(10.0, playback.speed * 1.5)
                elif event.key == pygame.K_UP:
                    playback.seek_rotation(playback.current_rotation + 1)
                    ppi_a_single.reset(); ppi_a_dual.reset()
                    ppi_b_dual.reset(); ppi_overlay.reset()
                elif event.key == pygame.K_DOWN:
                    playback.seek_rotation(playback.current_rotation - 1)
                    ppi_a_single.reset(); ppi_a_dual.reset()
                    ppi_b_dual.reset(); ppi_overlay.reset()
                elif event.key == pygame.K_HOME:
                    playback.seek_rotation(0)
                    ppi_a_single.reset(); ppi_a_dual.reset()
                    ppi_b_dual.reset(); ppi_overlay.reset()
                elif event.key == pygame.K_r:
                    playback.data_a = None
                    playback.data_b = None
                    playback.playing = False
                    file_a_name = ""
                    file_b_name = ""
                    ppi_a_single.reset(); ppi_a_dual.reset()
                    ppi_b_dual.reset(); ppi_overlay.reset()
                    view_mode = "single"
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_load_a.clicked(mouse_pos):
                    load_file("a")
                elif btn_load_b.clicked(mouse_pos):
                    load_file("b")
                elif btn_play.clicked(mouse_pos):
                    playback.playing = True
                elif btn_pause.clicked(mouse_pos):
                    playback.playing = False
                elif btn_single.clicked(mouse_pos):
                    view_mode = "single"
                elif btn_dual.clicked(mouse_pos):
                    view_mode = "dual"
                elif btn_overlay.clicked(mouse_pos):
                    view_mode = "overlay"

        # ---- Advance playback ----
        if playback.playing:
            spokes_a, spokes_b = playback.update(dt)

            # Only fade + paint the PPIs used by the current view mode
            if view_mode == "single":
                ppi_a_single.fade(2)
                mt_a = playback.data_a.max_ticks if playback.data_a else 8192
                for sp in spokes_a:
                    ppi_a_single.paint_spoke(sp["angle_tick"], sp["echoes"], mt_a)

            elif view_mode == "dual":
                ppi_a_dual.fade(2)
                ppi_b_dual.fade(2)
                mt_a = playback.data_a.max_ticks if playback.data_a else 8192
                mt_b = playback.data_b.max_ticks if playback.data_b else 8192
                for sp in spokes_a:
                    ppi_a_dual.paint_spoke(sp["angle_tick"], sp["echoes"], mt_a)
                for sp in spokes_b:
                    ppi_b_dual.paint_spoke(sp["angle_tick"], sp["echoes"], mt_b)

            elif view_mode == "overlay":
                ppi_overlay.fade(2)
                mt_a = playback.data_a.max_ticks if playback.data_a else 8192
                mt_b = playback.data_b.max_ticks if playback.data_b else 8192
                for sp in spokes_a:
                    ppi_overlay.paint_spoke_overlay(
                        sp["angle_tick"], sp["echoes"], True, mt_a)
                for sp in spokes_b:
                    ppi_overlay.paint_spoke_overlay(
                        sp["angle_tick"], sp["echoes"], False, mt_b)

        # ---- Render ----
        screen.fill((0, 0, 0))

        if view_mode == "single":
            ppi_a_single.draw(screen)
            if not playback.data_a:
                txt = prompt_font.render("Press A to load a radar CSV", True, (0, 120, 0))
                tr = txt.get_rect(center=(WIN_W // 2, PPI_AREA_H // 2))
                screen.blit(txt, tr)
        elif view_mode == "dual":
            ppi_a_dual.draw(screen)
            ppi_b_dual.draw(screen)
            if not playback.data_a:
                txt = prompt_font.render("Press A to load", True, (0, 100, 0))
                screen.blit(txt, txt.get_rect(center=(375, PPI_AREA_H // 2)))
            if not playback.data_b:
                txt = prompt_font.render("Press B to load", True, (160, 120, 0))
                screen.blit(txt, txt.get_rect(center=(1125, PPI_AREA_H // 2)))
        elif view_mode == "overlay":
            ppi_overlay.draw(screen)
            if not playback.data_a and not playback.data_b:
                txt = prompt_font.render("Load files with A / B keys", True, (0, 120, 0))
                screen.blit(txt, txt.get_rect(center=(WIN_W // 2, PPI_AREA_H // 2)))

        draw_controls(screen, font, font_sm, playback, view_mode, buttons,
                      file_a_name, file_b_name)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
