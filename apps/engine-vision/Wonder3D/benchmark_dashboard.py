"""
benchmark_dashboard.py
benchmark_log.json을 읽어 인터랙티브 HTML 대시보드를 생성한다.

사용법:
  python benchmark_dashboard.py
  → benchmark_dashboard.html 생성
"""

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LOG_FILE  = Path(__file__).parent / "benchmark_log.json"
OUT_FILE  = Path(__file__).parent / "benchmark_dashboard.html"


def load_data() -> pd.DataFrame:
    if not LOG_FILE.exists():
        print(f"[ERROR] {LOG_FILE} 없음. benchmark_runner.py를 먼저 실행하세요.")
        return pd.DataFrame()

    with open(LOG_FILE) as f:
        records = json.load(f)

    rows = []
    for r in records:
        p = r.get("params", {})
        m = r.get("mesh",   {})
        g = r.get("gpu_after", r.get("gpu_before", {}))
        rows.append({
            "timestamp":           r.get("timestamp", ""),
            "scene":               r.get("scene", ""),
            "note":                p.get("note", ""),
            # 메쉬 해상도
            "resolution":          p.get("resolution", 0),
            # 해시그리드
            "n_levels":            p.get("n_levels", 0),
            "n_features_per_level":p.get("n_features_per_level", 0),
            "hashmap_size":        p.get("hashmap_size", 0),
            "base_resolution":     p.get("base_resolution", 0),
            # 배치 / 샘플링
            "train_num_rays":      p.get("train_num_rays", 0),
            "max_train_num_rays":  p.get("max_train_num_rays", 0),
            "num_samples_per_ray": p.get("num_samples_per_ray", 0),
            "ray_chunk":           p.get("ray_chunk", 0),
            # 학습
            "max_steps":           p.get("max_steps", 0),
            "precision":           p.get("precision", 0),
            "lr_geometry":         p.get("lr_geometry", 0),
            "lr_texture":          p.get("lr_texture", 0),
            # MLP
            "mlp_neurons":         p.get("mlp_neurons", 0),
            "mlp_hidden_layers":   p.get("mlp_hidden_layers", 0),
            # 결과
            "elapsed_min":         r.get("elapsed_min", 0),
            "elapsed_sec":         r.get("elapsed_sec", 0),
            "step1_sec":           r.get("step1_sec", 0),
            "step2_sec":           r.get("step2_sec", 0),
            "vertices":            m.get("vertices", 0),
            "faces":               m.get("faces", 0),
            "success":             r.get("success", False),
            "gpu_name":            g.get("gpu_name", "N/A"),
            "mem_total_mb":        g.get("mem_total_mb", 0),
            "mem_used_mb":         g.get("mem_used_mb", 0),
        })
    return pd.DataFrame(rows)


def label(row) -> str:
    parts = [f"res{row['resolution']}", f"lv{row['n_levels']}", f"step{row['max_steps']}"]
    if row["note"]:
        parts.append(row["note"])
    return " | ".join(parts)


def build_dashboard(df: pd.DataFrame):
    df["label"] = df.apply(label, axis=1)
    df["color"] = df["success"].map({True: "#2ecc71", False: "#e74c3c"})

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "⏱ 실험별 총 소요 시간 (분)",
            "🔢 Resolution별 소요 시간",
            "📐 n_levels별 소요 시간",
            "🔄 max_steps별 소요 시간",
            "📦 Resolution별 버텍스 수",
            "💾 GPU 메모리 사용량",
        ],
        vertical_spacing=0.14,
        horizontal_spacing=0.1,
    )

    # ── 1. 실험별 막대 ─────────────────────────────────────────────────────────
    fig.add_trace(go.Bar(
        x=df["label"], y=df["elapsed_min"],
        marker_color=df["color"],
        text=df["elapsed_min"].apply(lambda v: f"{v:.1f}m"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>소요: %{y:.2f}분<extra></extra>",
        showlegend=False,
    ), row=1, col=1)

    # ── 2. Resolution 산점도 ───────────────────────────────────────────────────
    for steps, grp in df.groupby("max_steps"):
        fig.add_trace(go.Scatter(
            x=grp["resolution"], y=grp["elapsed_min"],
            mode="markers+lines",
            name=f"steps={steps}",
            marker=dict(size=10),
            hovertemplate="res=%{x}<br>시간=%{y:.2f}분<extra></extra>",
        ), row=1, col=2)

    # ── 3. n_levels 박스 ───────────────────────────────────────────────────────
    for lv in sorted(df["n_levels"].unique()):
        sub = df[df["n_levels"] == lv]
        fig.add_trace(go.Box(
            y=sub["elapsed_min"], name=f"n_levels={lv}",
            boxpoints="all", jitter=0.3,
            hovertemplate="n_levels=%{name}<br>시간=%{y:.2f}분<extra></extra>",
        ), row=2, col=1)

    # ── 4. max_steps 산점도 ────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["max_steps"], y=df["elapsed_min"],
        mode="markers",
        marker=dict(size=10, color=df["resolution"],
                    colorscale="Viridis", showscale=True,
                    colorbar=dict(title="resolution", x=1.02, len=0.3, y=0.16)),
        text=df["label"],
        hovertemplate="<b>%{text}</b><br>steps=%{x}<br>시간=%{y:.2f}분<extra></extra>",
        showlegend=False,
    ), row=2, col=2)

    # ── 5. Resolution vs 버텍스 ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=df["resolution"], y=df["vertices"],
        mode="markers",
        marker=dict(size=10, color="#3498db"),
        text=df["label"],
        hovertemplate="<b>%{text}</b><br>res=%{x}<br>버텍스=%{y:,}<extra></extra>",
        showlegend=False,
    ), row=3, col=1)

    # ── 6. GPU 메모리 ──────────────────────────────────────────────────────────
    gpu_df = df[df["mem_total_mb"] > 0]
    if not gpu_df.empty:
        fig.add_trace(go.Bar(
            x=gpu_df["label"],
            y=gpu_df["mem_used_mb"],
            name="사용",
            marker_color="#e67e22",
            hovertemplate="%{x}<br>사용: %{y:,} MB<extra></extra>",
        ), row=3, col=2)
        fig.add_trace(go.Bar(
            x=gpu_df["label"],
            y=gpu_df["mem_total_mb"] - gpu_df["mem_used_mb"],
            name="여유",
            marker_color="#bdc3c7",
            hovertemplate="%{x}<br>여유: %{y:,} MB<extra></extra>",
        ), row=3, col=2)

    # ── 레이아웃 ───────────────────────────────────────────────────────────────
    gpu_name = df["gpu_name"].iloc[-1] if not df.empty else "N/A"
    fig.update_layout(
        title=dict(
            text=f"Lego Digital Twin — 3D 재구성 벤치마크 대시보드<br>"
                 f"<sup>GPU: {gpu_name} | 총 실험: {len(df)}회</sup>",
            x=0.5, font=dict(size=18),
        ),
        barmode="stack",
        height=1100,
        template="plotly_white",
        legend=dict(orientation="h", y=-0.05),
    )

    # 축 레이블
    fig.update_yaxes(title_text="소요 시간 (분)", row=1, col=1)
    fig.update_yaxes(title_text="소요 시간 (분)", row=1, col=2)
    fig.update_xaxes(title_text="Resolution",    row=1, col=2)
    fig.update_yaxes(title_text="소요 시간 (분)", row=2, col=1)
    fig.update_yaxes(title_text="소요 시간 (분)", row=2, col=2)
    fig.update_xaxes(title_text="max_steps",     row=2, col=2)
    fig.update_yaxes(title_text="버텍스 수",      row=3, col=1)
    fig.update_xaxes(title_text="Resolution",    row=3, col=1)
    fig.update_yaxes(title_text="메모리 (MB)",    row=3, col=2)

    return fig


def build_table(df: pd.DataFrame) -> go.Figure:
    df["label"] = df.apply(label, axis=1)

    col_defs = [
        ("timestamp",           "시각"),
        ("label",               "설정 요약"),
        # 메쉬
        ("resolution",          "해상도(MC)"),
        # 해시그리드
        ("n_levels",            "n_levels"),
        ("n_features_per_level","feat/level"),
        ("hashmap_size",        "hashmap"),
        ("base_resolution",     "base_res"),
        # 배치/샘플링
        ("train_num_rays",      "train_rays"),
        ("max_train_num_rays",  "max_rays"),
        ("num_samples_per_ray", "samples/ray"),
        ("ray_chunk",           "ray_chunk"),
        # 학습
        ("max_steps",           "iterations"),
        ("precision",           "precision"),
        ("lr_geometry",         "lr_geo"),
        ("lr_texture",          "lr_tex"),
        # MLP
        ("mlp_neurons",         "MLP뉴런"),
        ("mlp_hidden_layers",   "MLP레이어"),
        # 시간
        ("step1_sec",           "STEP1(초)"),
        ("step2_sec",           "STEP2(초)"),
        ("elapsed_min",         "총시간(분)"),
        # 결과
        ("vertices",            "버텍스"),
        ("faces",               "면"),
        ("gpu_name",            "GPU"),
        ("mem_used_mb",         "VRAM(MB)"),
        ("success",             "성공"),
    ]
    cols   = [c for c, _ in col_defs]
    headers = [f"<b>{h}</b>" for _, h in col_defs]

    sub = df[cols].copy()
    sub["elapsed_min"] = sub["elapsed_min"].apply(lambda v: f"{v:.2f}")
    sub["vertices"]    = sub["vertices"].apply(lambda v: f"{v:,}")
    sub["faces"]       = sub["faces"].apply(lambda v: f"{v:,}")
    sub["success"]     = sub["success"].map({True: "✅", False: "❌"})

    row_colors  = ["#f0f0f0" if i % 2 == 0 else "white" for i in range(len(sub))]
    cell_colors = [row_colors for _ in cols]

    fig = go.Figure(go.Table(
        columnwidth=[120, 180] + [80] * (len(cols) - 2),
        header=dict(
            values=headers,
            fill_color="#2c3e50",
            font=dict(color="white", size=11),
            align="center",
        ),
        cells=dict(
            values=[sub[c].tolist() for c in cols],
            fill_color=cell_colors,
            align=["left", "left"] + ["center"] * (len(cols) - 2),
            font=dict(size=11),
            height=28,
        ),
    ))
    fig.update_layout(
        title="📋 전체 실험 기록 (파라미터 전체)",
        height=max(300, 80 + len(df) * 30),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def main():
    df = load_data()
    if df.empty:
        return

    charts = build_dashboard(df)
    table  = build_table(df)

    # 두 figure를 하나의 HTML로 합치기
    html_charts = charts.to_html(full_html=False, include_plotlyjs="cdn")
    html_table  = table.to_html(full_html=False, include_plotlyjs=False)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <title>Lego 3D Benchmark Dashboard</title>
  <style>
    body {{ font-family: 'Segoe UI', sans-serif; background: #f5f6fa; margin: 0; padding: 20px; }}
    h1 {{ color: #2c3e50; text-align: center; margin-bottom: 4px; }}
    .subtitle {{ text-align: center; color: #7f8c8d; margin-bottom: 24px; font-size: 14px; }}
    .card {{ background: white; border-radius: 12px; padding: 20px;
             box-shadow: 0 2px 8px rgba(0,0,0,.08); margin-bottom: 24px; }}
  </style>
</head>
<body>
  <h1>🧱 Lego Digital Twin — 3D 재구성 벤치마크</h1>
  <p class="subtitle">파라미터별 소요시간·품질 비교 | 하드웨어 업그레이드 근거 자료</p>
  <div class="card">{html_charts}</div>
  <div class="card">{html_table}</div>
</body>
</html>"""

    OUT_FILE.write_text(html, encoding="utf-8")
    print(f"[대시보드 생성] {OUT_FILE}")
    print(f"Windows에서 열기: \\\\wsl$\\Ubuntu{OUT_FILE}")


if __name__ == "__main__":
    main()
