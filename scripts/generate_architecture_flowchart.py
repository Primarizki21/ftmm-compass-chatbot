#!/usr/bin/env python3
"""
FTMM COMPASS — Pipeline Architecture Flowchart Generator
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
    def __init__(self, width: float = 14.5, height: float = 22.0, dpi: int = 300):
        self.fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        self.ax.set_xlim(0, 14.5)
        self.ax.set_ylim(0, 22.0)
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
            badge_w = min(1.85, w - 0.4)
            badge_h = 0.23
            badge_x = cx - badge_w / 2
            badge_y = cy + h / 2 - badge_h - 0.10
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
                         fontsize=10.0, fontweight="bold", color=title_color, zorder=4)

            # Subtitle placed below title
            if subtitle:
                self.ax.text(cx, cy - h / 2 + 0.35, subtitle, ha="center", va="center",
                             fontsize=7.5, color=self.colors["process_sub"], zorder=4,
                             linespacing=1.25)
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
            self.ax.text(cx, cy + 0.12, title, ha="center", va="center",
                         fontsize=9.8, fontweight="bold", color=self.colors["decision_title"], zorder=4)
            self.ax.text(cx, cy - 0.18, subtitle, ha="center", va="center",
                         fontsize=7.4, color="#1E40AF", zorder=4, linespacing=1.20)
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
        self.ax.text(7.25, 21.4, title, ha="center", va="center",
                     fontsize=15, fontweight="bold", color="#0F172A")
        self.ax.text(7.25, 21.05, subtitle, ha="center", va="center",
                     fontsize=9.2, color="#475569")

    def save(self, png_path: str, pdf_path: str):
        """Saves high-res PNG and vector PDF."""
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
        self.fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.3)
        plt.close(self.fig)


def build_ftmm_compass_flowchart():
    builder = FlowchartBuilder(width=14.5, height=22.0, dpi=300)
    builder.render_header(
        "FTMM COMPASS — AI Engine Microservice End-to-End Pipeline",
        "Single-Embed Shared Latent Vector + Cascaded Guardrails + System-1 Decider + CSP Planner"
    )

    # 1. Start Node
    builder.add_terminal(
        node_id="start", cx=7.25, cy=20.2, w=5.4, h=0.75,
        title="Student Query (Browser / Chatbot UI)",
        subtitle="POST /api/v1/chat/stream { message, history, student_profile }"
    )

    # 2. Tier 1 Process: Deterministic Syntax Filter
    builder.add_process(
        node_id="tier1", cx=7.25, cy=18.6, w=6.2, h=1.2,
        title="Tier 1: Deterministic Syntax Filter",
        subtitle="SQL Injection Sanitization &\nMalicious Parameter Tampering Defense",
        badge="0.0 ms Latency"
    )

    # 3. Tier 1 Refusal Node
    builder.add_process(
        node_id="refusal_t1", cx=1.40, cy=18.6, w=2.6, h=0.9,
        title="Security Refusal",
        subtitle="Blocked: Malicious SQL Syntax\nHTTP 400 Bad Request (0.0 ms)",
        theme="refusal"
    )

    # 4. Single-Embed Process: Ollama C++ Native
    builder.add_process(
        node_id="embed", cx=7.25, cy=16.7, w=6.4, h=1.2,
        title="Single-Embed Step (Ollama C++ Native Daemon)",
        subtitle="1x Inference nomic-embed-text -> L2-Normalized Vector (d=768)\nZero-Redundancy Shared Memory Latent Pipeline",
        theme="embed", badge="~30 ms Latency"
    )

    # 5. Tier 2 Decision: Semantic Dual-Centroid Gate
    builder.add_decision(
        node_id="tier2", cx=7.25, cy=14.65, w=5.8, h=1.75,
        title="Tier 2: Semantic Dual-Centroid Gate",
        subtitle="L2 Cosine Centroid Boundary (<35 ms)\nOOD Detection & Adversarial Jailbreak Defense"
    )

    # 6. Tier 2 Refusal Node
    builder.add_process(
        node_id="refusal_t2", cx=1.40, cy=14.65, w=2.6, h=0.9,
        title="Topic & Scope Refusal",
        subtitle="Blocked: Non-Academic / Jailbreak\nStandard Error Envelope (~30 ms)",
        theme="refusal"
    )

    # 7. Memory Manager Process
    builder.add_process(
        node_id="memory", cx=7.25, cy=12.55, w=6.6, h=1.2,
        title="Conversation Memory & Context Injection",
        subtitle="Dense Gazetteer Linking (369 Kurikulum FTMM)\nStudent Profile & Latent Ellipsis Context",
        badge="In-Memory State"
    )

    # 8. System-1 Decider Decision Node
    builder.add_decision(
        node_id="decider", cx=7.25, cy=10.45, w=6.6, h=1.85,
        title="System-1 Decider (6-Class Intent Matrix)",
        subtitle="Matrix Dot-Product BLAS O(1) (<0.1 ms)\nShared Latent Intent Classification"
    )

    # 9. Execution Subsystems (4 Parallel Blocks)
    exec_y = 7.65
    exec_h = 1.85
    exec_w = 2.85

    builder.add_process(
        node_id="exec_chat", cx=1.85, cy=exec_y, w=exec_w, h=exec_h,
        title="ReAct Direct (Chitchat)",
        subtitle="Intent: CHAT_ONLY\nZero Tool Schema (~400-700 ms)\nDeepSeek-Style Prefix Caching",
        theme="route", badge="Fast Path"
    )
    builder.add_process(
        node_id="exec_cag", cx=5.45, cy=exec_y, w=exec_w, h=exec_h,
        title="Direct CAG Route",
        subtitle="Intent: REGULATION_SOP\nPre-cached 8 SOP FTMM (~811 tok)\n0.0 ms Retrieval Overhead",
        theme="route", badge="Zero-Retrieval"
    )

    builder.add_process(
        node_id="exec_tools", cx=9.05, cy=exec_y, w=exec_w, h=exec_h,
        title="Tools ReAct Agent",
        subtitle="Intent: PREREQUISITE / SKS / DISCOV\nHybrid FAISS Dense + BM25 RRF\nPrerequisite DAG & SKS Constraint Rail",
        theme="route", badge="Tool-Calling"
    )

    builder.add_process(
        node_id="exec_csp", cx=12.65, cy=exec_y, w=exec_w, h=exec_h,
        title="Neuro-Symbolic CSP Planner",
        subtitle="Intent: DEGREE_PLANNING\nDense Gazetteer & FAISS Track Exp\nTopological Placement, Parity & Dedup",
        theme="route", badge="CSP Solver"
    )

    # 10. Tier 4 Process: Physical Parameter Rail
    builder.add_process(
        node_id="tier4", cx=7.25, cy=5.05, w=6.6, h=1.15,
        title="Tier 4: Physical Parameter Rail",
        subtitle="Deterministic Argument Validation\nLegal Semester 1–8 & Capacity Limit SKS 0–24",
        badge="Safety Invariant"
    )

    # 11. Compilation & Stream Node
    builder.add_process(
        node_id="stream", cx=7.25, cy=3.40, w=6.8, h=1.15,
        title="Asynchronous Token Streamer & DegreePlan JSON",
        subtitle="Yielding SSE Event Tokens kata-per-kata\nCompiling DegreePlanPayload JSON Widget",
        badge="SSE Streaming"
    )

    # 12. Terminal End Node
    builder.add_terminal(
        node_id="end", cx=7.25, cy=1.80, w=6.2, h=0.75,
        title="Output SSE Stream & DegreePlanCardWidget",
        subtitle="Live Typing Effect & Visual Semester 1–8 Roadmap"
    )

    # CONNECTIONS & ROUTING
    # Start -> Tier 1
    builder.draw_direct_arrow("start", "tier1", from_snap="bottom", to_snap="top")

    # Tier 1 -> Refusal T1 (Blocked)
    builder.draw_horizontal_branch("tier1", "refusal_t1", from_snap="left", to_snap="right",
                                   label="SQL Injection Attack", label_color="#E11D48")

    # Tier 1 -> Embed (Allowed)
    builder.draw_direct_arrow("tier1", "embed", from_snap="bottom", to_snap="top",
                              label="Allowed", label_color="#059669", label_side="right")

    # Embed -> Tier 2
    builder.draw_direct_arrow("embed", "tier2", from_snap="bottom", to_snap="top",
                              label="Shared Latent d=768", label_color="#0D9488", label_side="right")

    # Tier 2 -> Refusal T2 (OOD Blocked)
    builder.draw_horizontal_branch("tier2", "refusal_t2", from_snap="left", to_snap="right",
                                   label="OOD Non-Academic", label_color="#E11D48")

    # Tier 2 -> Memory (In-Scope)
    builder.draw_direct_arrow("tier2", "memory", from_snap="bottom", to_snap="top",
                              label="In-Scope Academic", label_color="#059669", label_side="right")

    # Memory -> Decider
    builder.draw_direct_arrow("memory", "decider", from_snap="bottom", to_snap="top",
                              label="Context Enriched", label_color="#2563EB", label_side="right")

    # Decider -> 4 Execution Blocks (1-to-N Branching Bus)
    target_ids = ["exec_chat", "exec_cag", "exec_tools", "exec_csp"]
    branch_labels = ["CHAT_ONLY", "REGULATION_SOP", "DISCOVERY / SKS", "DEGREE_PLANNING"]
    builder.draw_fork_1_to_n(source_id="decider", target_ids=target_ids,
                             bus_y=9.05, labels=branch_labels)

    # 4 Execution Blocks -> Tier 4 (N-to-1 Collector Bus)
    builder.draw_collector_n_to_1(source_ids=target_ids, target_id="tier4", bus_y=6.05)

    # Tier 4 -> Stream
    builder.draw_direct_arrow("tier4", "stream", from_snap="bottom", to_snap="top",
                              label="Parameter Valid", label_color="#059669", label_side="right")

    # Stream -> End
    builder.draw_direct_arrow("stream", "end", from_snap="bottom", to_snap="top")

    # Save artifacts
    png_out = "docs/architecture/ftmm_compass_pipeline_flowchart.png"
    pdf_out = "docs/architecture/ftmm_compass_pipeline_flowchart.pdf"
    builder.save(png_out, pdf_out)
    print(f"Flowchart successfully generated:\n  - PNG: {png_out}\n  - PDF: {pdf_out}")


if __name__ == "__main__":
    build_ftmm_compass_flowchart()
