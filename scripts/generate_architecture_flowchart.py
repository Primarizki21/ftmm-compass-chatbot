#!/usr/bin/env python3
"""
FTMM COMPASS: Pipeline Architecture Flowchart Generator
Generates high-resolution publication-grade flowcharts using pure matplotlib.
Follows strict geometric constraints:
- Process: Rectangle (Snap points at center of sides)
- Decision: Diamond / Polygon (Snap points at vertices)
- Terminal: Capsule / Rounded (Snap points at top/bottom center)
- 1-to-N Branching: Single source point -> Vertical stem -> Horizontal bus -> Orthogonal drop lines to targets
- N-to-1 Merging: Vertical drops from blocks -> Horizontal collector bus -> Single drop line to target
"""

import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class FlowchartBuilder:
    def __init__(self, width: float = 14.5, height: float = 24.5, dpi: int = 300):
        self.width = width
        self.height = height
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        self.ax.set_xlim(0, width)
        self.ax.set_ylim(0, height)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.nodes = {}

        # Academic color palette
        self.colors = {
            "terminal_bg": "#1E293B",      # Slate 800
            "terminal_fg": "#F8FAFC",      # Slate 50
            "process_bg": "#F8FAFC",       # Slate 50
            "process_border": "#334155",   # Slate 700
            "process_title": "#0F172A",    # Slate 900
            "process_sub": "#475569",      # Slate 600
            "decision_bg": "#EFF6FF",      # Blue 50
            "decision_border": "#2563EB",  # Blue 600
            "decision_title": "#1E3A8A",   # Blue 900
            "refusal_bg": "#FFF1F2",       # Rose 50
            "refusal_border": "#E11D48",   # Rose 600
            "refusal_title": "#9F1239",    # Rose 900
            "embed_bg": "#F0FDFA",         # Teal 50
            "embed_border": "#0D9488",     # Teal 600
            "embed_title": "#134E4A",      # Teal 900
            "route_bg": "#F0FDF4",         # Green 50
            "route_border": "#059669",     # Green 600
            "route_title": "#064E3B",      # Green 900
            "arrow": "#334155",            # Slate 700
            "arrow_label": "#1E293B",      # Slate 800
            "bus_line": "#475569",         # Slate 600
        }

    def add_terminal(self, node_id: str, cx: float, cy: float, w: float, h: float,
                     title: str, subtitle: str = ""):
        """Draws a rounded capsule for Start or End."""
        rect = patches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle=patches.BoxStyle("Round", pad=0.1, rounding_size=h / 2),
            linewidth=2.0,
            edgecolor=self.colors["terminal_bg"],
            facecolor=self.colors["terminal_bg"],
            zorder=3
        )
        self.ax.add_patch(rect)

        if subtitle:
            self.ax.text(cx, cy + 0.12, title, ha="center", va="center",
                         fontsize=11, fontweight="bold", color=self.colors["terminal_fg"], zorder=4)
            self.ax.text(cx, cy - 0.16, subtitle, ha="center", va="center",
                         fontsize=8.5, color="#CBD5E1", zorder=4)
        else:
            self.ax.text(cx, cy, title, ha="center", va="center",
                         fontsize=11, fontweight="bold", color=self.colors["terminal_fg"], zorder=4)

        self.nodes[node_id] = {
            "type": "terminal",
            "cx": cx, "cy": cy, "w": w, "h": h,
            "top": (cx, cy + h / 2),
            "bottom": (cx, cy - h / 2),
            "left": (cx - w / 2, cy),
            "right": (cx + w / 2, cy),
        }

    def add_process(self, node_id: str, cx: float, cy: float, w: float, h: float,
                    title: str, subtitle: str = "", theme: str = "default",
                    badge: str = ""):
        """Draws a standard rectangular process box with center-side snap points."""
        bg = self.colors["process_bg"]
        border = self.colors["process_border"]
        title_color = self.colors["process_title"]

        if theme == "embed":
            bg = self.colors["embed_bg"]
            border = self.colors["embed_border"]
            title_color = self.colors["embed_title"]
        elif theme == "route":
            bg = self.colors["route_bg"]
            border = self.colors["route_border"]
            title_color = self.colors["route_title"]
        elif theme == "refusal":
            bg = self.colors["refusal_bg"]
            border = self.colors["refusal_border"]
            title_color = self.colors["refusal_title"]

        # Drop shadow
        shadow = patches.FancyBboxPatch(
            (cx - w / 2 + 0.05, cy - h / 2 - 0.05), w, h,
            boxstyle=patches.BoxStyle("Square", pad=0.04),
            linewidth=0, facecolor="#000000", alpha=0.07, zorder=2
        )
        self.ax.add_patch(shadow)

        # Main box
        rect = patches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle=patches.BoxStyle("Square", pad=0.04),
            linewidth=1.8, edgecolor=border, facecolor=bg, zorder=3
        )
        self.ax.add_patch(rect)

        # Clean vertical stacking: Badge at top-center -> Title -> Subtitle
        if badge:
            badge_w = min(2.5, w - 0.35)
            badge_h = 0.24
            badge_x = cx - badge_w / 2
            badge_y = cy + h / 2 - badge_h - 0.08
            badge_patch = patches.FancyBboxPatch(
                (badge_x, badge_y), badge_w, badge_h,
                boxstyle=patches.BoxStyle("Round", pad=0.02, rounding_size=0.08),
                linewidth=0, facecolor=border, zorder=4
            )
            self.ax.add_patch(badge_patch)
            self.ax.text(cx, badge_y + badge_h / 2, badge,
                         ha="center", va="center", fontsize=7.2, fontweight="bold",
                         color="#FFFFFF", zorder=5)

            # Title placed comfortably below badge
            title_y = badge_y - 0.22
            self.ax.text(cx, title_y, title, ha="center", va="center",
                         fontsize=9.8, fontweight="bold", color=title_color, zorder=4)

            # Subtitle placed below title
            if subtitle:
                self.ax.text(cx, cy - h / 2 + (0.45 if h > 1.5 else 0.32), subtitle, ha="center", va="center",
                             fontsize=7.3, color=self.colors["process_sub"], zorder=4,
                             linespacing=1.22)
        else:
            # No badge: vertically centered title and subtitle
            if subtitle:
                self.ax.text(cx, cy + 0.15, title, ha="center", va="center",
                             fontsize=10.0, fontweight="bold", color=title_color, zorder=4)
                self.ax.text(cx, cy - 0.18, subtitle, ha="center", va="center",
                             fontsize=7.6, color=self.colors["process_sub"], zorder=4,
                             linespacing=1.25)
            else:
                self.ax.text(cx, cy, title, ha="center", va="center",
                             fontsize=10.0, fontweight="bold", color=title_color, zorder=4)

        self.nodes[node_id] = {
            "type": "process",
            "cx": cx, "cy": cy, "w": w, "h": h,
            "top": (cx, cy + h / 2),
            "bottom": (cx, cy - h / 2),
            "left": (cx - w / 2, cy),
            "right": (cx + w / 2, cy),
        }

    def add_decision(self, node_id: str, cx: float, cy: float, w: float, h: float,
                     title: str, subtitle: str = ""):
        """Draws a diamond decision polygon with vertex snap points."""
        top_pt = (cx, cy + h / 2)
        right_pt = (cx + w / 2, cy)
        bottom_pt = (cx, cy - h / 2)
        left_pt = (cx - w / 2, cy)
        diamond_pts = [top_pt, right_pt, bottom_pt, left_pt]

        # Shadow
        shadow_pts = [(x + 0.06, y - 0.06) for x, y in diamond_pts]
        shadow = patches.Polygon(shadow_pts, closed=True, linewidth=0,
                                 facecolor="#000000", alpha=0.07, zorder=2)
        self.ax.add_patch(shadow)

        # Main diamond
        poly = patches.Polygon(diamond_pts, closed=True, linewidth=2.0,
                               edgecolor=self.colors["decision_border"],
                               facecolor=self.colors["decision_bg"], zorder=3)
        self.ax.add_patch(poly)

        # Text centered comfortably within diamond bounds
        if subtitle:
            self.ax.text(cx, cy + 0.14, title, ha="center", va="center",
                         fontsize=9.5, fontweight="bold", color=self.colors["decision_title"], zorder=4)
            self.ax.text(cx, cy - 0.20, subtitle, ha="center", va="center",
                         fontsize=7.3, color="#1E40AF", zorder=4, linespacing=1.20)
        else:
            self.ax.text(cx, cy, title, ha="center", va="center",
                         fontsize=9.8, fontweight="bold", color=self.colors["decision_title"], zorder=4)
        self.nodes[node_id] = {
            "type": "decision",
            "cx": cx, "cy": cy, "w": w, "h": h,
            "top": top_pt,
            "bottom": bottom_pt,
            "left": left_pt,
            "right": right_pt,
        }

    def draw_direct_arrow(self, from_id: str, to_id: str, from_snap: str = "bottom",
                          to_snap: str = "top", label: str = "", label_color: str = "#0F172A",
                          label_side: str = "right"):
        """Draws a direct orthogonal arrow from one snap point to another."""
        p1 = self.nodes[from_id][from_snap]
        p2 = self.nodes[to_id][to_snap]

        arrow = patches.FancyArrowPatch(
            p1, p2,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color=self.colors["arrow"],
            zorder=3
        )
        self.ax.add_patch(arrow)

        if label:
            mx = (p1[0] + p2[0]) / 2
            my = (p1[1] + p2[1]) / 2
            lx = mx + (0.35 if label_side == "right" else -0.35)
            self.ax.text(lx, my, label, ha="left" if label_side == "right" else "right",
                         va="center", fontsize=8.2, fontweight="bold", color=label_color,
                         bbox=dict(boxstyle="square,pad=0.15", fc="#FFFFFF", ec="none", alpha=0.9),
                         zorder=5)

    def draw_horizontal_branch(self, from_id: str, to_id: str, from_snap: str = "left",
                               to_snap: str = "right", label: str = "", label_color: str = "#E11D48"):
        """Draws a horizontal arrow between side snap points."""
        p1 = self.nodes[from_id][from_snap]
        p2 = self.nodes[to_id][to_snap]

        arrow = patches.FancyArrowPatch(
            p1, p2,
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color=label_color,
            zorder=3
        )
        self.ax.add_patch(arrow)

        if label:
            mx = (p1[0] + p2[0]) / 2
            my = p1[1] + 0.15
            self.ax.text(mx, my, label, ha="center", va="bottom",
                         fontsize=7.0, fontweight="bold", color=label_color,
                         bbox=dict(boxstyle="square,pad=0.10", fc="#FFFFFF", ec="none", alpha=0.95),
                         zorder=5)

    def draw_fork_1_to_n(self, source_id: str, target_ids: list[str], bus_y: float,
                         labels: list[str] = None):
        """
        Draws standard 1-to-N Orthogonal Fork Bus:
        1. Single source point from bottom vertex of decision node.
        2. Drops down vertically to horizontal bus_y.
        3. Extends horizontal bus line from min(target_x) to max(target_x).
        4. Drops orthogonal vertical arrows from bus_y down to top-center of each target.
        """
        src_pt = self.nodes[source_id]["bottom"]
        src_x, src_y = src_pt

        # 1. Stem line from source to bus_y
        self.ax.plot([src_x, src_x], [src_y, bus_y], color=self.colors["bus_line"],
                     linewidth=2.0, zorder=3)

        # Compute target x coordinates and top snap points
        target_pts = [self.nodes[tid]["top"] for tid in target_ids]
        target_xs = [pt[0] for pt in target_pts]
        min_x = min(target_xs)
        max_x = max(target_xs)

        # 2. Horizontal bus line
        self.ax.plot([min_x, max_x], [bus_y, bus_y], color=self.colors["bus_line"],
                     linewidth=2.0, zorder=3)

        # 3. Orthogonal vertical arrows to each target
        for i, (tid, pt) in enumerate(zip(target_ids, target_pts)):
            tx, ty = pt
            arrow = patches.FancyArrowPatch(
                (tx, bus_y), (tx, ty),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.8,
                color=self.colors["bus_line"],
                zorder=3
            )
            self.ax.add_patch(arrow)

            # Badge / label above each target along drop line
            if labels and i < len(labels):
                lbl = labels[i]
                mid_y = (bus_y + ty) / 2
                self.ax.text(tx, mid_y, lbl, ha="center", va="center",
                             fontsize=7.5, fontweight="bold", color="#1E3A8A",
                             bbox=dict(boxstyle="round,pad=0.2,rounding_size=0.1",
                                       fc="#EFF6FF", ec="#93C5FD", lw=1.0, alpha=0.95),
                             zorder=5)

    def draw_collector_n_to_1(self, source_ids: list[str], target_id: str, bus_y: float):
        """
        Draws N-to-1 Collector Bus:
        1. Drops vertical lines from bottom-center of each source block to bus_y.
        2. Extends horizontal collector line across all sources.
        3. Drops single vertical arrow from (target_x, bus_y) to target top-center.
        """
        src_pts = [self.nodes[sid]["bottom"] for sid in source_ids]
        src_xs = [pt[0] for pt in src_pts]
        min_x = min(src_xs)
        max_x = max(src_xs)

        # 1. Vertical lines from each source down to bus_y
        for sx, sy in src_pts:
            self.ax.plot([sx, sx], [sy, bus_y], color=self.colors["bus_line"],
                         linewidth=1.8, zorder=3)

        # 2. Horizontal collector line
        self.ax.plot([min_x, max_x], [bus_y, bus_y], color=self.colors["bus_line"],
                     linewidth=1.8, zorder=3)

        # 3. Single drop arrow into target node
        tgt_x, tgt_y = self.nodes[target_id]["top"]
        arrow = patches.FancyArrowPatch(
            (tgt_x, bus_y), (tgt_x, tgt_y),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.8,
            color=self.colors["arrow"],
            zorder=3
        )
        self.ax.add_patch(arrow)

    def render_header(self, title: str, subtitle: str):
        """Adds a professional academic header bar."""
        self.ax.text(self.width / 2, self.height - 0.65, title, ha="center", va="center",
                     fontsize=14.5, fontweight="bold", color="#0F172A")
        self.ax.text(self.width / 2, self.height - 1.02, subtitle, ha="center", va="center",
                     fontsize=9.0, color="#475569")
    def save(self, png_path: str, pdf_path: str):
        """Saves high-res PNG and vector PDF."""
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
        self.fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.3)
        plt.close(self.fig)


def build_ftmm_compass_flowchart():
    builder = FlowchartBuilder(width=14.5, height=24.5, dpi=300)
    builder.render_header(
        "FTMM COMPASS: Alur Kerja Kecerdasan Buatan Chatbot Akademik",
        "Penyelarasan Bahasa + Pertahanan Bertingkat + Vektor Makna Nomic Embed + Penalaran Llama 3.2 3B"
    )

    # 1. Start Node
    builder.add_terminal(
        node_id="start", cx=7.25, cy=22.80, w=6.4, h=0.75,
        title="Pertanyaan Mahasiswa (Browser / Chatbot UI)",
        subtitle="Pesan teks, riwayat obrolan, dan profil mahasiswa via POST /api/v1/chat/stream"
    )

    # 2. Tier 1 Process: Filter Keamanan Input
    builder.add_process(
        node_id="tier1", cx=7.25, cy=21.25, w=6.4, h=1.15,
        title="Tier 1: Filter Keamanan Input",
        subtitle="Pemeriksaan instan (0.0 ms) untuk memblokir kode berbahaya\ndan manipulasi parameter numerik ilegal",
        badge="Keamanan 0.0 ms"
    )

    # 3. Tier 1 Refusal Node
    builder.add_process(
        node_id="refusal_t1", cx=1.40, cy=21.25, w=2.6, h=0.95,
        title="Penolakan Keamanan",
        subtitle="Ditolak: Pola SQL Injection\nHTTP 400 Bad Request (0.0 ms)",
        theme="refusal"
    )

    # 4. Pre-NLU Academic Normalizer (Step 1b)
    builder.add_process(
        node_id="normalizer", cx=7.25, cy=19.55, w=6.4, h=1.15,
        title="Penyelarasan Bahasa Akademik",
        subtitle="Merapikan singkatan gaul (O(1)), salah dengar suara (STT),\ndan salah ketik kode mata kuliah (Damerau-Levenshtein)",
        badge="Normalisasi ~2 ms"
    )

    # 5. Single-Embed Process: Nomic Embed Text v2 MoE
    builder.add_process(
        node_id="embed", cx=7.25, cy=17.85, w=6.6, h=1.15,
        title="Pembuatan Vektor Makna (Single-Embed)",
        subtitle="Model: Nomic Embed Text v2 MoE (Ollama C++, 768 dimensi)\nMengubah pertanyaan jadi 1 vektor makna untuk dipakai bersama",
        theme="embed", badge="Vektor Laten ~30 ms"
    )

    # 6. Conversation Memory & Context Injection
    builder.add_process(
        node_id="memory", cx=7.25, cy=16.15, w=6.6, h=1.15,
        title="Penyambung Konteks Dialog (Memory Manager)",
        subtitle="Menghubungkan maksud pertanyaan dengan riwayat sesi,\nidentifikasi mata kuliah tersirat, dan profil mahasiswa",
        badge="Memori Sesi"
    )

    # 7. Tier 2 Decision: Penyaring Relevansi & Etika Akademik
    builder.add_decision(
        node_id="tier2", cx=7.25, cy=14.25, w=6.4, h=1.60,
        title="Tier 2: Penyaring Relevansi & Etika Akademik",
        subtitle="Batas semantik kemiripan makna (<35 ms):\nMemastikan topik perkuliahan FTMM & menolak joki tugas"
    )

    # 8. Tier 2 Refusal Node
    builder.add_process(
        node_id="refusal_t2", cx=1.40, cy=14.25, w=2.6, h=0.95,
        title="Penolakan Etika / Topik",
        subtitle="Ditolak: Pertanyaan di luar topik FTMM\natau permintaan joki tugas kuliah",
        theme="refusal"
    )

    # 9. System-1 Decider Decision Node
    builder.add_decision(
        node_id="decider", cx=7.25, cy=12.10, w=6.6, h=1.65,
        title="Pemilih Jalur Cepat (System-1 Decider)",
        subtitle="Klasifikasi instan (<0.1 ms) berdasarkan vektor makna:\nMengarahkan pertanyaan ke jalur penanganan yang paling efisien"
    )

    # 10. Execution Subsystems (4 Parallel Blocks)
    exec_y = 9.00
    exec_h = 2.05
    exec_w = 2.85

    builder.add_process(
        node_id="exec_chat", cx=1.85, cy=exec_y, w=exec_w, h=exec_h,
        title="Jalur Sapaan Langsung",
        subtitle="Model: Llama 3.2 3B\nMenjawab sapaan dan tanya jawab umum\ntanpa memuat alat bantu (~0.5 dtk)",
        theme="route", badge="Jalur Cepat"
    )
    builder.add_process(
        node_id="exec_cag", cx=5.45, cy=exec_y, w=exec_w, h=exec_h,
        title="Jalur Aturan & SOP Resmi",
        subtitle="Model: Llama 3.2 3B + FAISS\nPanduan Akademik, Skripsi, PKL, Yudisium\ndan 8 SOP layanan administrasi FTMM",
        theme="route", badge="Hybrid CAG-RAG"
    )
    builder.add_process(
        node_id="exec_tools", cx=9.05, cy=exec_y, w=exec_w, h=exec_h,
        title="Jalur Info Kurikulum (RAG)",
        subtitle="Model: Llama 3.2 3B + FAISS\nPencarian 369 silabus mata kuliah FTMM,\ncek prasyarat (DAG), dan hitung batas SKS",
        theme="route", badge="Pencarian Cerdas"
    )
    builder.add_process(
        node_id="exec_csp", cx=12.65, cy=exec_y, w=exec_w, h=exec_h,
        title="Jalur Perencana Studi (CSP)",
        subtitle="Penyusun rencana 8 semester otomatis:\nMenjamin batas SKS, prasyarat lulus,\nparitas ganjil/genap, & 1 mata kuliah agama",
        theme="route", badge="Solver Otomatis"
    )

    # 11. Tier 4 Process: Validasi Batas Akademik Fisik
    builder.add_process(
        node_id="tier4", cx=7.25, cy=6.00, w=6.6, h=1.15,
        title="Tier 4: Validasi Aturan Akademik Fisik",
        subtitle="Pengecekan batasan baku sebelum eksekusi sistem:\nSemester legal 1–8 dan batas beban SKS (maks 24 SKS)",
        badge="Invarian Aturan"
    )

    # 12. Asynchronous Token Streamer
    builder.add_process(
        node_id="stream", cx=7.25, cy=4.30, w=6.8, h=1.15,
        title="Pengirim Jawaban Real-Time (SSE Stream)",
        subtitle="Mengalirkan respons kata per kata (efek mengetik langsung)\ndan menyusun data terstruktur untuk kartu rencana studi",
        badge="Streaming Langsung"
    )

    # 13. Terminal End Node
    builder.add_terminal(
        node_id="end", cx=7.25, cy=2.60, w=6.2, h=0.75,
        title="Tampilan Jawaban di Layar Mahasiswa",
        subtitle="Teks jawaban mengalir langsung & kartu visual rencana studi interaktif"
    )

    # CONNECTIONS & ROUTING
    # Start -> Tier 1
    builder.draw_direct_arrow("start", "tier1", from_snap="bottom", to_snap="top")

    # Tier 1 -> Refusal T1 (Blocked)
    builder.draw_horizontal_branch(
        "tier1", "refusal_t1", from_snap="left", to_snap="right",
        label="Serangan SQL Ditolak", label_color="#E11D48"
    )

    # Tier 1 -> Normalizer (Allowed)
    builder.draw_direct_arrow(
        "tier1", "normalizer", from_snap="bottom", to_snap="top",
        label="Lolos Keamanan", label_color="#059669", label_side="right"
    )

    # Normalizer -> Embed
    builder.draw_direct_arrow(
        "normalizer", "embed", from_snap="bottom", to_snap="top",
        label="Teks Rapi", label_color="#059669", label_side="right"
    )

    # Embed -> Memory
    builder.draw_direct_arrow(
        "embed", "memory", from_snap="bottom", to_snap="top",
        label="Vektor Makna d=768", label_color="#0D9488", label_side="right"
    )

    # Memory -> Tier 2
    builder.draw_direct_arrow(
        "memory", "tier2", from_snap="bottom", to_snap="top",
        label="Konteks Lengkap", label_color="#2563EB", label_side="right"
    )

    # Tier 2 -> Refusal T2 (OOD / Homework Blocked)
    builder.draw_horizontal_branch(
        "tier2", "refusal_t2", from_snap="left", to_snap="right",
        label="Topik Luar / Joki Ditolak", label_color="#E11D48"
    )

    # Tier 2 -> Decider (Academic Allowed)
    builder.draw_direct_arrow(
        "tier2", "decider", from_snap="bottom", to_snap="top",
        label="Topik Sah FTMM", label_color="#059669", label_side="right"
    )

    # Decider -> 4 Execution Blocks (1-to-N Branching Bus)
    target_ids = ["exec_chat", "exec_cag", "exec_tools", "exec_csp"]
    branch_labels = ["Sapaan Langsung", "Aturan & SOP", "Katalog & Silabus", "Perencana Studi"]
    builder.draw_fork_1_to_n(
        source_id="decider", target_ids=target_ids,
        bus_y=10.60, labels=branch_labels
    )

    # 4 Execution Blocks -> Tier 4 (N-to-1 Collector Bus)
    builder.draw_collector_n_to_1(
        source_ids=target_ids, target_id="tier4", bus_y=7.25
    )

    # Tier 4 -> Stream
    builder.draw_direct_arrow(
        "tier4", "stream", from_snap="bottom", to_snap="top",
        label="Hasil Tervalidasi", label_color="#059669", label_side="right"
    )

    # Stream -> End
    builder.draw_direct_arrow("stream", "end", from_snap="bottom", to_snap="top")

    # Save artifacts
    png_out = "docs/architecture/ftmm_compass_pipeline_flowchart.png"
    pdf_out = "docs/architecture/ftmm_compass_pipeline_flowchart.pdf"
    builder.save(png_out, pdf_out)
    print(f"Flowchart successfully generated:\n  - PNG: {png_out}\n  - PDF: {pdf_out}")


if __name__ == "__main__":
    build_ftmm_compass_flowchart()
