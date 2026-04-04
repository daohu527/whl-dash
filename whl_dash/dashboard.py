#!/usr/bin/env python3

import os
import dash
import json
import base64
from dash import html, dcc, Input, Output, State, ALL, callback_context, Patch, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from whl_dash.data import RecordDataLoader
from whl_dash.template import TemplateManager, RowConfig, DashboardTemplate, SpatialLayerConfig

_LOADER_CACHE = {}

def get_loader(record_path: str):
    if not record_path:
        return None
    abs_path = os.path.abspath(record_path)
    if abs_path not in _LOADER_CACHE:
        try:
            _LOADER_CACHE[abs_path] = RecordDataLoader(abs_path)
        except Exception as e:
            print(f"Error loading {abs_path}: {e}")
            return None
    return _LOADER_CACHE[abs_path]

def get_available_records(base_dir="."):
    res = []
    if not os.path.isdir(base_dir):
        return res
    try:
        if any(".record" in f for f in os.listdir(base_dir)):
            res.append(base_dir)
        for name in os.listdir(base_dir):
            path = os.path.join(base_dir, name)
            if os.path.isdir(path):
                if any(".record" in f for f in os.listdir(path)):
                    res.append(path)
            elif os.path.isfile(path) and ".record" in name:
                res.append(path)
    except Exception:
        pass

    result_options = [{"label": os.path.basename(p) or p, "value": os.path.abspath(p)} for p in sorted(res)]
    return result_options

def _decimate_series(t: np.ndarray, y: np.ndarray, max_points: int = 4000):
    if t is None or y is None:
        return t, y
    n = len(t)
    if n <= max_points:
        return t, y
    idx = np.linspace(0, n - 1, num=max_points, dtype=int)
    return t[idx], y[idx]

SECTION_HEADER_STYLE = {
    "fontSize": "11px", "fontWeight": "bold", "color": "#666",
    "textTransform": "uppercase", "marginBottom": "6px"
}
PANEL_STYLE = {
    "background": "white", "border": "1px solid #ddd", "borderRadius": "4px",
    "padding": "12px", "marginBottom": "12px", "boxShadow": "0 1px 3px rgba(0,0,0,0.02)"
}
ICON_BTN_STYLE = {
    "cursor": "pointer", "background": "transparent", "border": "none",
    "fontSize": "16px", "padding": "2px", "display": "flex", "alignItems": "center"
}

def create_app(initial_record_path: str):
    mgr = TemplateManager(os.path.join(os.path.dirname(__file__), "templates.json"))
    map_panel_height_px = 360
    chart_toolbar_height_px = 42
    fit_chart_height_css = f"calc(100vh - {map_panel_height_px + chart_toolbar_height_px + 26}px)"

    app = dash.Dash(__name__, title="Apollo Debugger")

    def get_template_options():
        return [{"label": tpl_name, "value": tpl_name} for tpl_name in sorted(mgr.templates.keys())]

    base_dir = os.path.dirname(os.path.abspath(initial_record_path)) if initial_record_path and os.path.isfile(initial_record_path) else (initial_record_path or ".")
    if not os.path.isdir(base_dir):
        base_dir = "."

    records_options = get_available_records(base_dir)
    if initial_record_path and not any(o["value"] == os.path.abspath(initial_record_path) for o in records_options):
        lbl = os.path.basename(initial_record_path) or initial_record_path
        records_options.insert(0, {"label": lbl, "value": os.path.abspath(initial_record_path)})

    default_record = records_options[0]["value"] if records_options else None

    # UI LAYOUT
    app.layout = html.Div(
        [
            dcc.Store(id="current-time-store", data=0),
            # Left Sidebar
            html.Div(
                [
                    html.H3("Apollo Analyzer", style={"margin": "0 0 15px 0", "color": "#333", "fontSize": "20px"}),

                    # 1. Data Source Panel
                    html.Div(
                        [
                            html.Div("Data Source", style=SECTION_HEADER_STYLE),
                            html.Div(
                                [
                                    html.Span("📁", style={"color": "#666", "fontSize": "14px"}),
                                    dcc.Input(id="workspace-input", value=base_dir, placeholder="Workspace path...", style={"flex": "1", "border": "none", "outline": "none", "fontSize": "12px", "padding": "0 5px"}),
                                    html.Button("🔄", id="btn-scan", title="Scan Directory", style=ICON_BTN_STYLE)
                                ],
                                style={"display": "flex", "alignItems": "center", "border": "1px solid #ccc", "borderRadius": "4px", "padding": "4px 8px", "marginBottom": "8px", "background": "#fbfbfb"}
                            ),
                            dcc.Dropdown(
                                id="record-dropdown", options=records_options, value=default_record if not isinstance(default_record, list) else default_record, clearable=True, multi=True,
                                placeholder="Select record to load...",
                                style={"fontSize": "12px", "marginBottom": "4px"}
                            )
                        ], style=PANEL_STYLE
                    ),

                    # 2. Template Manager
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Template Settings", style=SECTION_HEADER_STYLE),
                                    html.Div(
                                        [
                                            dcc.Upload(id="upload-template", children=html.Button("📥", title="Import JSON", style=ICON_BTN_STYLE), style={"display": "inline-block"}),
                                            html.Button("📤", id="btn-export", title="Export JSON", style=ICON_BTN_STYLE),
                                            dcc.Download(id="download-template")
                                        ], style={"display": "flex", "gap": "5px"}
                                    )
                                ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "baseline", "marginBottom": "8px"}
                            ),
                            html.Div(
                                [
                                    dcc.Dropdown(id="tpl-dropdown", options=get_template_options(), value="pose_analysis", clearable=False, style={"flex": "1", "fontSize": "12px", "minWidth": "0"}),
                                    html.Button("💾", id="btn-save", title="Save Template", style=ICON_BTN_STYLE),
                                    html.Button("🗑", id="btn-delete", title="Delete Template", style={"color": "red", **ICON_BTN_STYLE}),
                                ], style={"display": "flex", "alignItems": "center", "marginBottom": "15px", "gap": "4px"}
                            ),

                            dcc.RadioItems(
                                id="editor-mode",
                                options=[
                                    {"label": "👁 Visual UI", "value": "visual"},
                                    {"label": "📝 Raw JSON", "value": "raw"}
                                ],
                                value="visual", inline=True,
                                style={"fontSize": "12px", "display": "flex", "gap": "15px", "borderBottom": "1px solid #eee", "paddingBottom": "10px", "marginBottom": "10px"}
                            ),
                            html.Div(id="json-status-msg", style={"fontSize": "11px", "marginBottom": "10px"}),

                            html.Div(
                                id="visual-editor",
                                children=[
                                    dcc.Tabs([
                                        dcc.Tab(label="Time Series", children=[
                                            html.Div(
                                                [
                                                    html.Button("➕ Add Chart Row", id="btn-add-row", style={"fontSize": "11px", "padding": "4px 8px", "cursor": "pointer"}),
                                                ], style={"display": "flex", "justifyContent": "flex-start", "margin": "10px 0"}
                                            ),
                                            html.Div(id="rows-container")
                                        ]),
                                        dcc.Tab(label="2D Spatial Layers", children=[
                                            html.Div(
                                                [
                                                    html.Button("➕ Blank Layer", id="btn-add-spatial", style={"fontSize": "11px", "padding": "4px 8px", "cursor": "pointer"}),
                                                    html.Button("🛣️ Planning Trajectory", id="btn-add-planning", style={"fontSize": "11px", "padding": "4px 8px", "cursor": "pointer", "marginLeft": "5px"}),
                                                    html.Button("📦 Perception Obstacles", id="btn-add-perception", style={"fontSize": "11px", "padding": "4px 8px", "cursor": "pointer", "marginLeft": "5px"}),
                                                ], style={"display": "flex", "justifyContent": "flex-start", "margin": "10px 0"}
                                            ),
                                            html.Div(id="spatial-container")
                                        ])
                                    ], style={"fontSize": "12px"}),
                                    html.Div(
                                        html.Button("▶ Render Dashboard", id="btn-render", style={"width": "100%", "background": "#28a745", "color": "white", "border": "none", "borderRadius": "3px", "fontWeight": "bold", "fontSize": "12px", "padding": "8px", "marginTop": "15px", "cursor": "pointer"}),
                                    )
                                ],
                                style={"display": "block"}
                            ),

                            # Raw JSON Editor Container
                            html.Div(
                                id="raw-editor",
                                children=[
                                    html.Div("Edit your templates file natively:", style={"fontSize": "11px", "color": "#777", "marginBottom": "5px"}),
                                    dcc.Textarea(
                                        id="raw-json-editor",
                                        style={
                                            "width": "100%", "height": "40vh", "fontFamily": "monospace",
                                            "fontSize": "12px", "whiteSpace": "pre", "boxSizing": "border-box",
                                            "border": "1px solid #ccc", "borderRadius": "4px", "padding": "5px"
                                        }
                                    ),
                                    html.Button("💾 Apply & Save Config", id="btn-save-json", style={"marginTop": "10px", "background": "#007bff", "color": "white", "border": "none", "padding": "6px", "borderRadius": "4px", "cursor": "pointer", "fontWeight": "bold", "width": "100%"})
                                ],
                                style={"display": "none", "flexDirection": "column"}
                            )
                        ], style={**PANEL_STYLE, "flex": "1", "display": "flex", "flexDirection": "column"}
                    ),

                    # 3. Available Signals Reference
                    html.Details(
                        id="signals-details",
                        children=[
                            html.Summary("Available Signals", style=SECTION_HEADER_STYLE),
                            html.Div(id="signals-list-container")
                        ],
                        style=PANEL_STYLE
                    )
                ],
                style={
                    "width": "420px", "padding": "20px 15px", "borderRight": "1px solid #dce0e4",
                    "height": "100vh", "overflowY": "auto", "display": "flex", "flexDirection": "column",
                    "background": "#f6f8fb", "boxSizing": "border-box", "fontFamily": "system-ui, -apple-system, sans-serif"
                },
            ),
            # Main Content
            html.Div(
                id="main-workspace-content",
                children=[
                    # Top: Spatial 2D Map & Playback
                    html.Div(
                        id="map-pane-wrapper",
                        children=[
                            html.Div([
                                html.Button("▶ Play", id="btn-play", style={"marginRight": "8px", "cursor": "pointer", "padding": "4px 12px", "fontWeight": "bold"}),
                                html.Div(dcc.Slider(id="time-slider", min=0, max=100, step=0.1, value=0, marks=None, tooltip={"placement": "bottom", "always_visible": True}), style={"flex": "1"}),
                                dcc.Interval(id="play-interval", interval=100, disabled=True)
                            ], style={"display": "flex", "alignItems": "center", "padding": "10px 40px 10px 10px", "background": "#fafafa", "borderBottom": "1px solid #ddd"}),

                            html.Div(id="map-container", children=[
                                dcc.Graph(
                                    id="map-graph",
                                    style={"height": "100%", "width": "100%"},
                                    config={"scrollZoom": True, "displaylogo": False, "doubleClick": "reset"}
                                )
                            ], style={"height": f"{map_panel_height_px - 40}px", "borderBottom": "2px solid #eee", "backgroundColor": "#fafafa", "resize": "vertical", "overflow": "auto", "minHeight": "150px"})
                        ]
                    ),

                    # Bottom: Time Series Charts
                    html.Div(
                        id="charts-pane-wrapper",
                        children=[
                        html.Div(
                            [
                                html.Label("Chart Layout Mode: ", style={"fontSize": "11px", "fontWeight": "bold", "marginRight": "8px"}),
                                dcc.RadioItems(
                                    id="pane-layout-mode",
                                    options=[
                                        {"label": " ↕ Split Vertically", "value": "vertical"},
                                        {"label": " ↔ Split Horizontally", "value": "horizontal"},
                                        {"label": " 🗺 Map Only", "value": "map-only"},
                                        {"label": " 📈 Charts Only", "value": "charts-only"}
                                    ],
                                    value="vertical", inline=True, style={"fontSize": "11px", "display": "inline-block", "marginRight": "15px", "paddingRight":"15px", "borderRight":"1px solid #ccc"}
                                ),
                                dcc.RadioItems(
                                    id="chart-layout-mode",
                                    options=[
                                        {"label": " Scrollable Charts", "value": "scroll"},
                                        {"label": " Fit Charts", "value": "fit"}
                                    ],
                                    value="scroll", inline=True, style={"fontSize": "11px", "display": "inline-block"}
                                )
                            ], style={"padding": "5px 20px", "backgroundColor": "#f8f9fa", "borderBottom": "1px solid #ddd"}
                        ),
                        html.Div(
                            id="graph-container",
                            style={"flex": "1", "display": "flex", "flexDirection": "column", "overflowY": "auto", "minHeight": "0"},
                            children=[
                                dcc.Loading(
                                    id="loading-graph", type="circle",
                                    children=[
                                        dcc.Graph(id="main-graph", style={"width": "100%", "minHeight": "400px"})
                                    ]
                                )
                            ]
                        )
                    ], style={"flex": "1", "display": "flex", "flexDirection": "column", "minHeight": "0", "overflow": "auto"})
                ],
                style={"flex": "1", "padding": "0px", "overflow": "hidden", "background": "white", "display": "flex", "flexDirection": "column"},
            ),
        ],
        style={"display": "flex", "height": "100vh", "margin": "-8px"},
    )

    # Callbacks
    @app.callback(
        Output("record-dropdown", "options"),
        Output("record-dropdown", "value", allow_duplicate=True),
        Input("btn-scan", "n_clicks"),
        State("workspace-input", "value"),
        prevent_initial_call=True
    )
    def scan_workspace(n_clicks, path):
        opts = get_available_records(path)
        return opts, opts[0]["value"] if opts else None

    @app.callback(
        Output("visual-editor", "style"),
        Output("raw-editor", "style"),
        Input("editor-mode", "value")
    )
    def toggle_mode(mode):
        if mode == "visual":
            return {"display": "block"}, {"display": "none"}
        return {"display": "none"}, {"display": "flex", "flexDirection": "column"}

    def get_current_rows(titles, signals):
        res = []
        if titles and signals:
            for t, s in zip(titles, signals):
                sig_list = [x.strip() for x in (s or "").replace(",", "\n").split("\n") if x.strip() and not x.strip().startswith("#")]
                res.append(RowConfig(title=(t or "Row"), signals=sig_list))
        return res

    def get_current_spatials(s_names, s_types, s_topics, s_bases, s_xs, s_ys, s_modes, s_colors):
        res = []
        if s_names is not None:
            for i in range(len(s_names)):
                res.append(SpatialLayerConfig(
                    name=s_names[i] or "",
                    layer_type=s_types[i] if s_types and len(s_types)>i else "track",
                    topic=s_topics[i] if s_topics and len(s_topics)>i else "",
                    array_base=s_bases[i] if s_bases and len(s_bases)>i else "",
                    x_expr=s_xs[i] if s_xs and len(s_xs)>i else "",
                    y_expr=s_ys[i] if s_ys and len(s_ys)>i else "",
                    mode=s_modes[i] if s_modes and len(s_modes)>i else "markers",
                    color=s_colors[i] if s_colors and len(s_colors)>i else "blue",
                ))
        return res

    @app.callback(
        Output("rows-container", "children"),
        Output("spatial-container", "children"),
        Output("tpl-dropdown", "options"),
        Output("tpl-dropdown", "value"),
        Output("raw-json-editor", "value"),
        Output("json-status-msg", "children"),
        Input("tpl-dropdown", "value"),
        Input("btn-save", "n_clicks"),
        Input("btn-delete", "n_clicks"),
        Input("btn-add-row", "n_clicks"),
        Input("btn-add-spatial", "n_clicks"),
        Input("btn-add-planning", "n_clicks"),
        Input("btn-add-perception", "n_clicks"),
        Input({"type": "btn-rem-row", "index": ALL}, "n_clicks"),
        Input({"type": "btn-rem-spatial", "index": ALL}, "n_clicks"),
        Input("btn-save-json", "n_clicks"),
        State("raw-json-editor", "value"),
        State({"type": "title-in", "index": ALL}, "value"),
        State({"type": "signals-in", "index": ALL}, "value"),
        State({"type": "sl-name", "index": ALL}, "value"),
        State({"type": "sl-type", "index": ALL}, "value"),
        State({"type": "sl-topic", "index": ALL}, "value"),
        State({"type": "sl-base", "index": ALL}, "value"),
        State({"type": "sl-x", "index": ALL}, "value"),
        State({"type": "sl-y", "index": ALL}, "value"),
        State({"type": "sl-mode", "index": ALL}, "value"),
        State({"type": "sl-color", "index": ALL}, "value"),
        State("tpl-dropdown", "value")
    )
    def handle_ui_state(
        sel_tpl, save_c, del_c, add_r, add_s, add_plan, add_percep, rem_r, rem_s, save_json,
        json_val, titles, signals,
        s_names, s_types, s_topics, s_bases, s_xs, s_ys, s_modes, s_colors,
        current_tpl_dropdown_val
    ):
        ctx = callback_context
        # Robust trigger detection: support normal ids and pattern-matching ids
        trig = "init"
        try:
            tlist = ctx.triggered
            if tlist:
                prop = tlist[0].get("prop_id", "")
                if prop:
                    pid = prop.split('.')[0]
                    try:
                        trig = json.loads(pid)
                    except Exception:
                        trig = pid
        except Exception:
            trig = "init"

        # Decide active template: if the tpl-dropdown triggered, use sel_tpl, otherwise keep previous
        if isinstance(trig, str) and trig == "tpl-dropdown":
            active_tpl = sel_tpl
        else:
            active_tpl = current_tpl_dropdown_val

        current_rows = get_current_rows(titles, signals)
        current_spatials = get_current_spatials(s_names, s_types, s_topics, s_bases, s_xs, s_ys, s_modes, s_colors)

        out_sel = active_tpl
        new_rows = current_rows
        new_spatials = current_spatials
        status_msg = ""

        try:
            if trig == "btn-save-json":
                data = json.loads(json_val)
                mgr.templates.clear()
                for k, v in data.items():
                    r_cfgs = [RowConfig(title=r.get("title", ""), signals=r.get("signals", [])) for r in v.get("rows", [])]
                    s_cfgs = [SpatialLayerConfig(**s) for s in v.get("spatial_layers", [])]
                    mgr.templates[k] = DashboardTemplate(name=k, rows=r_cfgs, spatial_layers=s_cfgs)
                mgr.save()
                status_msg = html.Span("✅ Config saved!", style={"color": "green"})
                if active_tpl not in mgr.templates and mgr.templates:
                    out_sel = list(mgr.templates.keys())[0]
                new_rows = mgr.templates.get(out_sel, DashboardTemplate("", [])).rows
                new_spatials = mgr.templates.get(out_sel, DashboardTemplate("", [])).spatial_layers

            elif trig == "init" or trig == "tpl-dropdown":
                tpl = mgr.templates.get(sel_tpl)
                if tpl:
                    new_rows = tpl.rows
                    new_spatials = tpl.spatial_layers
                out_sel = sel_tpl

            elif trig == "btn-add-row":
                new_rows.append(RowConfig(title="New Row", signals=["# signal"]))

            elif trig == "btn-add-spatial":
                new_spatials.append(SpatialLayerConfig("New Layer", "track", "", "", "", "", "lines", "blue"))

            elif isinstance(trig, dict) and trig.get("type") == "btn-rem-row":
                idx = trig.get("index")
                if 0 <= idx < len(new_rows): new_rows.pop(idx)

            elif isinstance(trig, dict) and trig.get("type") == "btn-rem-spatial":
                idx = trig.get("index")
                if 0 <= idx < len(new_spatials): new_spatials.pop(idx)

            elif trig == "btn-save" and active_tpl:
                mgr.templates[active_tpl] = DashboardTemplate(active_tpl, current_rows, spatial_layers=current_spatials)
                mgr.save()
                status_msg = html.Span(f"✅ Saved '{active_tpl}'", style={"color": "green"})

        except Exception as e:
            status_msg = html.Span(f"❌ Error: {str(e)}", style={"color": "red"})

        # Build Visual Controls for Rows
        r_children = []
        for i, r in enumerate(new_rows):
            chunk = html.Div([
                html.Div([
                    dcc.Input(type="text", value=r.title, id={"type": "title-in", "index": i}, style={"flex": "1", "marginRight": "5px"}),
                    html.Button("✖", id={"type": "btn-rem-row", "index": i}, style={"color": "red"})
                ], style={"display": "flex", "marginBottom": "5px"}),
                dcc.Textarea(value="\n".join(r.signals), id={"type": "signals-in", "index": i}, style={"width": "100%", "height": "40px"})
            ], style={"border": "1px solid #ddd", "padding": "5px", "marginBottom": "5px"})
            r_children.append(chunk)

        # Build Visual Controls for Spatials
        s_children = []
        for i, s in enumerate(new_spatials):
            chunk = html.Div([
                html.Div([
                    dcc.Input(type="text", value=s.name, id={"type": "sl-name", "index": i}, placeholder="Layer Name", style={"flex": "1", "marginRight": "5px", "fontWeight":"bold"}),
                    dcc.RadioItems(options=[{"label": "Track", "value": "track"}, {"label": "Frame", "value": "frame"}], value=s.layer_type, id={"type": "sl-type", "index": i}, style={"marginRight": "5px", "fontSize": "11px", "display": "flex", "alignItems": "center", "gap": "5px"}),
                    html.Button("✖", id={"type": "btn-rem-spatial", "index": i}, style={"color": "red"})
                ], style={"display": "flex", "marginBottom": "5px"}),
                html.Div([
                    dcc.Input(type="text", value=s.topic, id={"type": "sl-topic", "index": i}, placeholder="Topic (for frame mode)", style={"flex": "1"}),
                    dcc.Input(type="text", value=s.array_base, id={"type": "sl-base", "index": i}, placeholder="Array Base (e.g. perception_obstacle)", style={"flex": "1"}),
                ], style={"display": "flex", "gap":"5px", "marginBottom": "5px"}),
                html.Div([
                    dcc.Input(type="text", value=s.x_expr, id={"type": "sl-x", "index": i}, placeholder="X Expr", style={"flex": "1"}),
                    dcc.Input(type="text", value=s.y_expr, id={"type": "sl-y", "index": i}, placeholder="Y Expr", style={"flex": "1"}),
                ], style={"display": "flex", "gap":"5px", "marginBottom": "5px"}),
                html.Div([
                    dcc.RadioItems(options=["lines", "markers", "lines+markers"], value=s.mode, id={"type": "sl-mode", "index": i}, inline=True, style={"flex": "1", "fontSize":"11px"}),
                    dcc.Input(type="text", value=s.color, id={"type": "sl-color", "index": i}, placeholder="color", style={"flex": "1"}),
                ], style={"display": "flex", "gap":"5px"}),
            ], style={"border": "1px solid #007bff", "padding": "5px", "marginBottom": "5px", "background": "#f8f9fc"})
            s_children.append(chunk)

        opts = get_template_options()

        # Raw Dict build
        state_dict = {}
        for k, v in mgr.templates.items():
            state_dict[k] = {
                "name": v.name,
                "rows": [r.__dict__ for r in v.rows],
                "spatial_layers": [s.__dict__ for s in v.spatial_layers]
            }
        raw_json_str = json.dumps(state_dict, indent=2, ensure_ascii=False)

        return r_children, s_children, opts, out_sel, raw_json_str, status_msg

    # Base Render (When Record or Template changes materially)
    @app.callback(
        Output("main-graph", "figure"),
        Output("map-graph", "figure"),
        Output("time-slider", "max"),
        Output("time-slider", "value", allow_duplicate=True),
        Output("signals-list-container", "children", allow_duplicate=True),
        Input("btn-render", "n_clicks"),
        Input("record-dropdown", "value"),
        State("tpl-dropdown", "value"),
        State("chart-layout-mode", "value"),
        prevent_initial_call='initial_duplicate'
    )
    def render_base_graphs(n_clicks, cb_record_paths, active_tpl, layout_mode):
        fig_map = go.Figure()
        fig_map.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=30, b=10), dragmode="pan", uirevision="constant")

        fig_main = make_subplots(rows=1, cols=1)
        fig_main.update_layout(template="plotly_white", title="No Data Rows", uirevision="constant")

        if not cb_record_paths:
            return fig_main, fig_map, 100, 0, "No records"

        record_paths = [cb_record_paths] if isinstance(cb_record_paths, str) else list(cb_record_paths)
        loaders = [get_loader(p) for p in record_paths if get_loader(p)]
        if not loaders:
            return fig_main, fig_map, 100, 0, "No records"

        # Define MAX time based on primary loader (first track)
        loader = loaders[0]
        max_time = 0.0
        if not loader.master_df.empty:
            max_time = loader.master_df["relative_time_sec"].max()

        tpl = mgr.templates.get(active_tpl)
        if not tpl:
            return fig_main, fig_map, max_time, 0, "No template matched"

        # 1. Render Main Graph (Bottom Time Series)
        num_rows = len(tpl.rows)
        if num_rows > 0:
            fig_main = make_subplots(rows=num_rows, cols=1, shared_xaxes=True, subplot_titles=[r.title for r in tpl.rows], vertical_spacing=0.03)
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

            for i, r in enumerate(tpl.rows):
                for j, expr in enumerate(r.signals):
                    y = loader.evaluate(expr)
                    t = loader.master_df["relative_time_sec"].values
                    if y is not None and len(y) == len(t):
                        t_arr, y_arr = _decimate_series(t, y)
                        fig_main.add_trace(go.Scatter(x=t_arr, y=y_arr, name=expr, mode='lines', line=dict(color=colors[j % len(colors)])), row=i+1, col=1)

            fig_main.update_xaxes(showspikes=True, spikemode="across", spikethickness=1, spikedash="solid", spikecolor="#999")
            fig_main.update_layout(hovermode="x unified", height=max(450, num_rows * 280), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        # 2. Render Map Graph (Tracks only!)
        for sl_idx, sl in enumerate(tpl.spatial_layers):
            if sl.layer_type == 'track':
                x, y = loader.get_spatial_frame(0, sl)  # Track ignores time
                fig_map.add_trace(go.Scatter(
                    x=x, y=y, mode=sl.mode, name=sl.name,
                    line=dict(color=sl.color), marker=dict(color=sl.color, size=4),
                    hoverinfo='skip'
                ))
            elif sl.layer_type == 'frame':
                # Empty placeholder for frame, to be patched
                fig_map.add_trace(go.Scatter(
                    x=[], y=[], mode=sl.mode, name=sl.name,
                    line=dict(color=sl.color), marker=dict(color=sl.color, size=6)
                ))

        fig_map.update_yaxes(scaleanchor="x", scaleratio=1)
        fig_map.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        # Build signals html
        sig_list = [html.Li(s, style={"fontSize": "12px", "wordBreak": "break-all"}) for s in loader.available_signals]
        sig_ui = html.Ul(sig_list, style={"paddingLeft": "15px", "margin": "0"})
        return fig_main, fig_map, float(max_time), 0, sig_ui



    # --- Pane Layout Switcher --- #
    @app.callback(
        Output("main-workspace-content", "style"),
        Output("map-pane-wrapper", "style"),
        Output("charts-pane-wrapper", "style"),
        Input("pane-layout-mode", "value")
    )
    def switch_pane_layout(mode):
        base_main = {"flex": "1", "padding": "0px", "overflow": "hidden", "background": "white", "display": "flex"}

        if mode == "horizontal":
            base_main["flexDirection"] = "row"
            return base_main, {"flex": "1", "position": "relative", "minWidth": "30%", "borderRight": "2px solid #ccc", "resize": "horizontal", "overflow": "auto"}, {"flex": "1", "display": "flex", "flexDirection": "column", "minWidth": "0"}
        elif mode == "map-only":
            base_main["flexDirection"] = "column"
            return base_main, {"flex": "1", "position": "relative"}, {"display": "none"}
        elif mode == "charts-only":
            base_main["flexDirection"] = "column"
            return base_main, {"display": "none"}, {"flex": "1", "display": "flex", "flexDirection": "column", "minHeight": "0"}
        else:
            # vertical default
            base_main["flexDirection"] = "column"
            return base_main, {"position": "relative", "flexShrink": "0", "resize": "vertical", "overflow": "auto", "minHeight": "20vh", "height": "45vh"}, {"flex": "1", "display": "flex", "flexDirection": "column", "minHeight": "0"}



# --- Playback Logic --- #
# --- Playback Logic --- #
    @app.callback(
        Output("play-interval", "disabled"),
        Output("btn-play", "children"),
        Input("btn-play", "n_clicks"),
        State("play-interval", "disabled")
    )
    def toggle_play(n_clicks, disabled):
        if n_clicks:
            return not disabled, "⏸ Pause" if disabled else "▶ Play"
        return True, "▶ Play"

    @app.callback(
        Output("time-slider", "value"),
        Input("play-interval", "n_intervals"),
        State("time-slider", "value"),
        State("time-slider", "max")
    )
    def on_tick(n_intervals, current_time, max_t):
        new_time = current_time + 0.1
        if new_time > max_t:
            new_time = 0
        return round(new_time, 2)

    # --- Sync Slider To Store And Render Frames --- #
    @app.callback(
        Output("current-time-store", "data"),
        Output("map-graph", "figure", allow_duplicate=True),
        Output("main-graph", "figure", allow_duplicate=True),
        Input("time-slider", "value"),
        State("record-dropdown", "value"),
        State("tpl-dropdown", "value"),
        State("map-graph", "figure"),
        prevent_initial_call=True
    )
    def update_dynamic_frames(time_val, cb_record_paths, active_tpl, current_map_fig):
        if time_val is None: return no_update, no_update, no_update

        # 1. Update Crosshair on Main Graph
        patched_main = Patch()
        patched_main["layout"]["shapes"] = [{
            "type": "line", "xref": "x", "yref": "paper",
            "x0": time_val, "x1": time_val, "y0": 0, "y1": 1,
            "line": {"color": "red", "width": 2, "dash": "dash"}
        }]

        # 2. Update Map Frames
        record_paths = [cb_record_paths] if isinstance(cb_record_paths, str) else list(cb_record_paths or [])
        if not record_paths: return time_val, no_update, patched_main

        loader = get_loader(record_paths[0])
        tpl = mgr.templates.get(active_tpl)
        if not loader or not tpl: return time_val, no_update, patched_main

        patched_map = Patch()
        trace_idx = 0

        # We must align with how traces were added in `render_base_graphs`
        for sl in tpl.spatial_layers:
            if sl.layer_type == 'frame':
                x_arr, y_arr = loader.get_spatial_frame(time_val, sl)
                patched_map["data"][trace_idx]["x"] = x_arr
                patched_map["data"][trace_idx]["y"] = y_arr
            trace_idx += 1

        return time_val, patched_map, patched_main

    return app
