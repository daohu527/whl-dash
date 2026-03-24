#!/usr/bin/env python3

import os
import dash
import json
import base64
from dash import html, dcc, Input, Output, State, ALL, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from whl_dash.data import RecordDataLoader
from whl_dash.template import TemplateManager, RowConfig, DashboardTemplate

# A simple cache to avoid continuously reloading the same record
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
        # Include current base_dir if it has records
        if any(".record" in f for f in os.listdir(base_dir)):
            res.append(base_dir)
        # Scan immediate subdirectories
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


# Helpers for performance
def _decimate_series(t: np.ndarray, y: np.ndarray, max_points: int = 4000):
    """Reduce series to at most `max_points` by picking evenly spaced indices.
    Keeps endpoints. Returns (t_dec, y_dec).
    """
    if t is None or y is None:
        return t, y
    n = len(t)
    if n <= max_points:
        return t, y
    idx = np.linspace(0, n - 1, num=max_points, dtype=int)
    return t[idx], y[idx]

# Reusable UI Styles (VS Code / Modern IDE theme)
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

                    # 2. Template Manager & Editor Panel
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
                            # Template Selector Row
                            html.Div(
                                [
                                    dcc.Dropdown(id="tpl-dropdown", options=get_template_options(), value="control_feedback", clearable=False, style={"flex": "1", "fontSize": "12px", "minWidth": "0"}),
                                    html.Button("💾", id="btn-save", title="Save Template", style=ICON_BTN_STYLE),
                                    html.Button("🗑", id="btn-delete", title="Delete Template", style={"color": "red", **ICON_BTN_STYLE}),
                                ], style={"display": "flex", "alignItems": "center", "marginBottom": "15px", "gap": "4px"}
                            ),

                            # Segmented Control for Mode
                            dcc.RadioItems(
                                id="editor-mode",
                                options=[
                                    {"label": html.Span(["👁", html.Span(" Visual UI", style={"marginLeft":"4px"})], style={"display":"flex", "alignItems":"center"}), "value": "visual"},
                                    {"label": html.Span(["📝", html.Span(" Raw JSON", style={"marginLeft":"4px"})], style={"display":"flex", "alignItems":"center"}), "value": "raw"}
                                ],
                                value="visual",
                                inline=True,
                                style={"fontSize": "12px", "display": "flex", "gap": "15px", "borderBottom": "1px solid #eee", "paddingBottom": "10px", "marginBottom": "10px"}
                            ),

                            html.Div(id="json-status-msg", style={"fontSize": "11px", "marginBottom": "10px"}),

                            # Visual Editor Container
                            html.Div(
                                id="visual-editor",
                                children=[
                                    html.Div(
                                        [
                                            html.Button("➕ Add Chart Row", id="btn-add-row", style={"fontSize": "11px", "padding": "4px 8px", "cursor": "pointer"}),
                                            html.Button("▶ Render Canvas", id="btn-render", style={"background": "#28a745", "color": "white", "border": "none", "borderRadius": "3px", "fontWeight": "bold", "fontSize": "11px", "padding": "4px 10px", "cursor": "pointer"}),
                                        ],
                                        style={"display": "flex", "justifyContent": "space-between", "marginBottom": "10px"}
                                    ),
                                    html.Div(id="rows-container")
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
            # Main Content Graph (Split: Top Map, Bottom Charts)
            html.Div(
                [
                    html.Div([
                        html.Button(
                            "🗺️ Hide Map", id="btn-toggle-map",
                            style={
                                "position": "absolute", "top": "10px", "right": "16px", "zIndex": "1000",
                                "padding": "4px 8px", "fontSize": "12px", "cursor": "pointer",
                                "fontWeight": "bold", "color": "#444",
                                "background": "white", "border": "1px solid #ccc", "borderRadius": "4px", "boxShadow": "0 1px 3px rgba(0,0,0,0.1)"
                            }
                        ),
                        html.Div(id="map-container", children=[
                            dcc.Graph(
                                id="map-graph",
                                style={"height": "100%", "width": "100%"},
                                config={
                                    "scrollZoom": True,
                                    "displaylogo": False,
                                    "doubleClick": "reset",
                                    "responsive": True
                                }
                            )
                        ], style={"height": f"{map_panel_height_px}px", "borderBottom": "2px solid #eee", "backgroundColor": "#fafafa", "flexShrink": "0"})
                    ], style={"position": "relative", "flexShrink": "0"}),

                    html.Div([
                        html.Div(
                            [
                                html.Label("Chart Layout Mode: ", style={"fontSize": "11px", "fontWeight": "bold", "marginRight": "8px"}),
                                dcc.RadioItems(
                                    id="chart-layout-mode",
                                    options=[
                                        {"label": " Scrollable Details (Fixed Height)", "value": "scroll"},
                                        {"label": " Fit to Screen (Compare All)", "value": "fit"}
                                    ],
                                    value="scroll",
                                    inline=True,
                                    style={"fontSize": "11px", "display": "inline-block"}
                                )
                            ], style={"padding": "5px 20px", "backgroundColor": "#f8f9fa", "borderBottom": "1px solid #ddd", "display": "flex", "alignItems": "center"}
                        ),
                        html.Div(
                            id="graph-container",
                            style={"flex": "1", "display": "flex", "flexDirection": "column", "overflowY": "hidden", "minHeight": "0"},
                            children=[
                                dcc.Loading(
                                    id="loading-graph", type="circle",
                                    style={"height": "100%"},
                                    children=[
                                        dcc.Graph(id="main-graph", style={"width": "100%", "height": "100%", "minHeight": "0"})
                                    ]
                                )
                            ]
                        )
                    ], style={"flex": "1", "display": "flex", "flexDirection": "column", "minHeight": "0"})
                ],
                style={"flex": "1", "padding": "0px", "overflow": "hidden", "background": "white", "display": "flex", "flexDirection": "column"},
            ),
        ],
        style={"display": "flex", "height": "100vh", "margin": "-8px"},
    )

    # ---------------- CALLBACKS ---------------- #

    @app.callback(
        Output("map-container", "style"),
        Output("btn-toggle-map", "children"),
        Input("btn-toggle-map", "n_clicks"),
    )
    def toggle_map_panel(n_clicks):
        is_hidden = bool(n_clicks and n_clicks % 2 == 1)
        if is_hidden:
            return {"display": "none"}, "🗺️ Show Map"
        return {
            "height": f"{map_panel_height_px}px",
            "borderBottom": "2px solid #eee",
            "backgroundColor": "#fafafa",
            "flexShrink": "0"
        }, "🗺️ Hide Map"

    # 1. Workspace Scanner
    @app.callback(
        Output("record-dropdown", "options"),
        Output("record-dropdown", "value", allow_duplicate=True),
        Input("btn-scan", "n_clicks"),
        State("workspace-input", "value"),
        prevent_initial_call=True
    )
    def scan_workspace(n_clicks, path):
        opts = get_available_records(path)
        val = opts[0]["value"] if opts else None
        return opts, val

    # 2. View Toggle
    @app.callback(
        Output("visual-editor", "style"),
        Output("raw-editor", "style"),
        Input("editor-mode", "value")
    )
    def toggle_mode(mode):
        if mode == "visual":
            return {"display": "block"}, {"display": "none"}
        return {"display": "none"}, {"display": "flex", "flexDirection": "column"}

    # 3. Export Logic
    @app.callback(
        Output("download-template", "data"),
        Input("btn-export", "n_clicks"),
        prevent_initial_call=True
    )
    def export_templates(n_clicks):
        try:
            with open(mgr.path, "r", encoding="utf-8") as f:
                content = f.read()
            return dcc.send_string(content, "apollo_dash_templates.json")
        except Exception as e:
            return dash.no_update

    # 4. Signals Glossary updates when record changes
    @app.callback(
        Output("signals-list-container", "children"),
        Output("signals-details", "children"),
        Input("record-dropdown", "value")
    )
    def update_signals_list(record_paths):
        if not record_paths:
            return html.Div("No record selected", style={"color": "#999", "fontSize": "11px"}), dash.no_update

        record_path = record_paths[0] if isinstance(record_paths, list) else record_paths
        loader = get_loader(record_path)
        if not loader:
            return html.Div("Failed to load record or no signals found", style={"color": "red", "fontSize": "11px"}), dash.no_update

        roots = [k for k in loader.eval_env.keys() if k not in ("np", "math", "relative_time_sec")]

        content = html.Div([
            html.Div("Variables: " + ", ".join(roots) + " | Globals: np, math", style={"fontSize": "10px", "marginBottom": "8px", "color": "#0056b3"}),
            html.Pre(
                "\n".join(loader.available_signals),
                style={"height": "200px", "overflowY": "scroll", "fontSize": "11px", "background": "#fff", "border": "1px solid #ddd", "padding": "8px", "borderRadius": "4px"}
            )
        ])
        summary = [html.Summary(f"Available Signals ({len(loader.available_signals)})", style={"cursor": "pointer", **SECTION_HEADER_STYLE}), content]
        return content, summary

    # Helper: read UI Visual Rows
    def get_current_rows(titles, signals):
        current_rows = []
        if titles is not None and signals is not None:
            for t, s in zip(titles, signals):
                sig_list = [x.strip() for x in (s or "").replace(",", "\n").split("\n") if x.strip() and not x.strip().startswith("#")]
                current_rows.append(RowConfig(title=(t or "Row"), signals=sig_list))
        return current_rows

    # 5. Core Engine: Manage Template UI States (Visual vs JSON vs Import)
    @app.callback(
        Output("rows-container", "children"),
        Output("tpl-dropdown", "options"),
        Output("tpl-dropdown", "value"),
        Output("raw-json-editor", "value"),
        Output("json-status-msg", "children"),
        Input("tpl-dropdown", "value"),
        Input("btn-save", "n_clicks"),
        Input("btn-delete", "n_clicks"),
        Input("btn-add-row", "n_clicks"),
        Input({"type": "btn-rem", "index": ALL}, "n_clicks"),
        Input("btn-save-json", "n_clicks"),
        Input("upload-template", "contents"),
        State("raw-json-editor", "value"),
        State({"type": "title-in", "index": ALL}, "value"),
        State({"type": "signals-in", "index": ALL}, "value"),
        State("tpl-dropdown", "value"),
    )
    def handle_ui_state(
        sel_tpl, save_c, del_c, add_c, rem_c, save_json_c, upload_contents,
        json_val, titles, signals, current_tpl_dropdown_val
    ):
        ctx = callback_context
        trig = ctx.triggered_id if ctx.triggered else "init"

        # Note: input string from trigger replaces 'sel_tpl' during its trigger.
        # But 'current_tpl_dropdown_val' is stable.
        active_tpl = sel_tpl if trig == "tpl-dropdown" else current_tpl_dropdown_val

        current_rows = get_current_rows(titles, signals)
        out_sel = active_tpl
        new_rows = current_rows
        status_msg = ""

        try:
            # Handle JSON Save
            if trig == "btn-save-json":
                data = json.loads(json_val)
                mgr.templates.clear()
                for k, v in data.items():
                    r_cfgs = [RowConfig(title=r.get("title", ""), signals=r.get("signals", [])) for r in v.get("rows", [])]
                    mgr.templates[k] = DashboardTemplate(name=k, rows=r_cfgs)
                mgr.save()
                status_msg = html.Span("✅ Config applied and saved!", style={"color": "green"})
                if active_tpl not in mgr.templates and mgr.templates:
                    out_sel = list(mgr.templates.keys())[0]
                new_rows = mgr.templates.get(out_sel, DashboardTemplate("", [])).rows

            # Handle JSON Upload
            elif trig == "upload-template" and upload_contents:
                content_type, content_string = upload_contents.split(',')
                decoded = base64.b64decode(content_string).decode('utf-8')
                data = json.loads(decoded)
                # Overwrite or merge? We overwrite entirely for consistency
                mgr.templates.clear()
                for k, v in data.items():
                    r_cfgs = [RowConfig(title=r.get("title", ""), signals=r.get("signals", [])) for r in v.get("rows", [])]
                    mgr.templates[k] = DashboardTemplate(name=k, rows=r_cfgs)
                mgr.save()
                status_msg = html.Span("✅ Imported successfully!", style={"color": "green"})
                out_sel = list(mgr.templates.keys())[0] if mgr.templates else None
                new_rows = mgr.templates.get(out_sel, DashboardTemplate("", [])).rows if out_sel else []

            # Handle Initialize / Normal View Switch
            elif trig == "init" or trig == "tpl-dropdown":
                tpl = mgr.templates.get(sel_tpl)
                if tpl:
                    new_rows = tpl.rows
                out_sel = sel_tpl

            # Handle Visual Add Row
            elif trig == "btn-add-row":
                new_rows.append(RowConfig(title="New Row", signals=["# Enter signal here"]))

            # Handle Visual Delete Row
            elif isinstance(trig, dict) and trig.get("type") == "btn-rem":
                idx = trig.get("index")
                if 0 <= idx < len(new_rows):
                    new_rows.pop(idx)

            # Handle Toolbar Save
            elif trig == "btn-save":
                # Save just uses current_rows with current template name (overwrites it)
                if active_tpl:
                    mgr.templates[active_tpl] = DashboardTemplate(active_tpl, current_rows)
                    mgr.save()
                    status_msg = html.Span(f"✅ Saved '{active_tpl}'", style={"color": "green"})

            # Handle Toolbar Delete
            elif trig == "btn-delete":
                if active_tpl in mgr.templates:
                    del mgr.templates[active_tpl]
                    mgr.save()
                    out_sel = list(mgr.templates.keys())[0] if mgr.templates else None
                    if out_sel:
                        new_rows = mgr.templates[out_sel].rows
                        status_msg = html.Span(f"✅ Deleted '{active_tpl}'", style={"color": "green"})
                    else:
                        new_rows = [RowConfig("Empty", [])]

        except Exception as e:
            status_msg = html.Span(f"❌ Error: {str(e)}", style={"color": "red"})

        # Build Visual Controls
        children = []
        for i, r in enumerate(new_rows):
            chunk = html.Div(
                [
                    html.Div(
                        [
                            dcc.Input(type="text", value=r.title, id={"type": "title-in", "index": i}, placeholder="Chart Title", style={"flex": "1", "marginRight": "5px", "fontWeight": "bold", "border": "1px solid #ddd", "borderRadius": "3px", "padding": "2px 5px", "fontSize": "11px"}),
                            html.Button("✖", id={"type": "btn-rem", "index": i}, title="Remove Row", style={"color": "#dc3545", "background": "transparent", "border": "none", "cursor": "pointer", "fontSize": "12px"}),
                        ],
                        style={"display": "flex", "marginBottom": "5px"}
                    ),
                    dcc.Textarea(
                        value="\n".join(r.signals), id={"type": "signals-in", "index": i},
                        style={"width": "100%", "height": "40px", "fontFamily": "monospace", "boxSizing": "border-box", "fontSize": "11px", "border": "1px solid #ddd", "borderRadius": "3px"}
                    ),
                ],
                style={"border": "1px solid #ddd", "padding": "8px", "marginBottom": "8px", "background": "white", "borderRadius": "4px"}
            )
            children.append(chunk)

        opts = get_template_options()
        # Build Raw JSON from Memory state
        raw_json_str = json.dumps({k: {"name": v.name, "rows": [r.__dict__ for r in v.rows]} for k,v in mgr.templates.items()}, indent=2, ensure_ascii=False)

        return children, opts, out_sel, raw_json_str, status_msg

    # 6. Global Renderer
    @app.callback(
        [Output("main-graph", "figure"), Output("map-graph", "figure"), Output("main-graph", "style"), Output("graph-container", "style")],
        [Input("btn-render", "n_clicks"),
         Input("tpl-dropdown", "value"),
         Input("record-dropdown", "value"),
         Input("chart-layout-mode", "value")],
        [State({"type": "title-in", "index": ALL}, "value"),
         State({"type": "signals-in", "index": ALL}, "value")],
         prevent_initial_call=False
    )
    def render_graph(n_clicks, tpl_change, cb_record_paths, layout_mode, titles, signals):
        fig_map = go.Figure()
        fig_map.update_layout(
            template="plotly_white",
            margin=dict(l=10, r=10, t=30, b=10),
            title="3D / Geo Map Trajectory",
            dragmode="pan",
            uirevision="map-user-scale"
        )

        if layout_mode == "fit":
            main_style = {"width": "100%", "height": fit_chart_height_css, "minHeight": fit_chart_height_css}
            graph_container_style = {"flex": "1", "display": "flex", "flexDirection": "column", "overflowY": "hidden", "minHeight": "0"}
        else:
            scroll_h = max(450, (len(titles or []) or 1) * 280)
            main_style = {"width": "100%", "height": f"{scroll_h}px", "minHeight": f"{scroll_h}px"}
            graph_container_style = {"flex": "1", "display": "flex", "flexDirection": "column", "overflowY": "auto", "minHeight": "0"}

        if not titles:
            fig = make_subplots(rows=1, cols=1)
            fig.update_layout(template="plotly_white", title="No Data Rows")
            return fig, fig_map, main_style, graph_container_style

        if isinstance(cb_record_paths, str):
            record_paths = [cb_record_paths]
        elif isinstance(cb_record_paths, (list, tuple)):
            record_paths = list(cb_record_paths)
        else:
            record_paths = []

        loaders = []
        for p in record_paths:
            loader = get_loader(p)
            if loader:
                loaders.append((p, loader))
        if not loaders:
            fig = make_subplots(rows=1, cols=1)
            fig.update_layout(template="plotly_white", title="No Data Loaded. Please select a record source.")
            return fig, fig_map, main_style, graph_container_style

        num_rows = len(titles)

        # Compute dynamic row_heights when in 'fit' mode so subplots compress
        row_heights = None
        if layout_mode == "fit":
            # base fraction per row, clamp to avoid invisible rows
            base = 1.0 / max(1, num_rows)
            min_frac = 0.06
            raw = [max(base, min_frac) for _ in range(num_rows)]
            s = sum(raw)
            row_heights = [r / s for r in raw]

        fig = make_subplots(
            rows=max(1, num_rows),
            cols=1,
            shared_xaxes=True,
            subplot_titles=titles,
            vertical_spacing=0.035,
            row_heights=row_heights,
        )

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        dash_styles = ["solid", "dash", "dot", "dashdot"]

        valid_plots = 0
        for i, (title, sig_block) in enumerate(zip(titles, signals)):
            row_idx = i + 1
            sig_list = [
                x.strip() for x in (sig_block or "").replace(",", "\n").split("\n")
                if x.strip() and not x.strip().startswith("#")
            ]

            for loader_idx, (rec_path, loader) in enumerate(loaders):
                t = loader.master_df["relative_time_sec"].values
                rec_name = os.path.basename(rec_path)
                line_dash = dash_styles[loader_idx % len(dash_styles)]

                for trace_idx, expr in enumerate(sig_list):
                    y = loader.evaluate(expr)
                    if y is not None and len(y) == len(t):
                        # Decimate long series for rendering performance
                        max_points = 4000
                        t_arr, y_arr = _decimate_series(t, y, max_points=max_points)

                        valid_y = y_arr[np.isfinite(y_arr)]
                        unique_vals = np.unique(valid_y) if len(valid_y) > 0 else []
                        is_step = len(unique_vals) <= 12 and all((v % 1 == 0) for v in unique_vals) if len(unique_vals) > 0 else False

                        shape = "hv" if is_step else "linear"
                        c = colors[trace_idx % len(colors)]

                        # Identify A/B testing traces
                        trace_name = f"{expr} [{rec_name}]" if len(loaders) > 1 else expr

                        # Use WebGL-backed Scatter for very large traces
                        trace_cls = go.Scattergl if len(t_arr) > 2000 else go.Scatter

                        fig.add_trace(
                            trace_cls(
                                x=t_arr, y=y_arr, mode="lines", name=trace_name,
                                line={"shape": shape, "color": c, "width": 1.5 if loader_idx == 0 else 2.0, "dash": line_dash},
                                hovertemplate=f"{trace_name}<br>t=%{{x:.3f}}s<br>val=%{{y:.5f}}<extra></extra>",
                                showlegend=True,
                            ),
                            row=row_idx, col=1,
                        )
                        valid_plots += 1
        if valid_plots > 0:
            fig.update_xaxes(title_text="Relative Time [s]", row=num_rows, col=1)
            # Sync Cursor Across Subplots (Spikelines)
            fig.update_xaxes(showspikes=True, spikemode="across", spikethickness=1, spikedash="solid", spikecolor="#999")
            fig.update_layout(hovermode="x unified")
            fig.update_traces(xaxis="x")

        layout_height = None if layout_mode == "fit" else max(450, num_rows * 280)
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            height=layout_height,
            autosize=True,
            margin=dict(l=40, r=40, t=50, b=40)
        )

        # Map Layout Logic
        has_map_data = False
        for loader_idx, (rec_path, loader) in enumerate(loaders):
            df = loader.master_df
            x_col, y_col = None, None
            for c in df.columns:
                cl = c.lower()
                if ('position.x' in cl or 'utm_x' in cl or cl.endswith('.x') or cl == 'x') and not x_col: x_col = c
                if ('position.y' in cl or 'utm_y' in cl or cl.endswith('.y') or cl == 'y') and not y_col: y_col = c

            if x_col and y_col:
                has_map_data = True
                # Map traces can also be very long; decimate for performance
                max_map_points = 6000
                xs = np.array(df[x_col].tolist())
                ys = np.array(df[y_col].tolist())
                custom = np.array(df["relative_time_sec"].tolist()) if "relative_time_sec" in df.columns else None
                if len(xs) > max_map_points:
                    idx = np.linspace(0, len(xs) - 1, num=max_map_points, dtype=int)
                    xs = xs[idx]
                    ys = ys[idx]
                    custom = custom[idx] if custom is not None else None

                trace_cls_map = go.Scattergl if len(xs) > 2000 else go.Scatter
                fig_map.add_trace(trace_cls_map(
                    x=xs.tolist(), y=ys.tolist(), mode="lines", name=os.path.basename(rec_path),
                    line=dict(color=colors[loader_idx % len(colors)], width=2.5, dash=dash_styles[loader_idx % len(dash_styles)]),
                    customdata=custom.tolist() if custom is not None else [],
                    hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>t=%{customdata:.3f}s<extra></extra>"
                ))

        if has_map_data:
            fig_map.update_yaxes(scaleanchor="x", scaleratio=1)
            fig_map.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            fig_map.update_xaxes(fixedrange=False)
            fig_map.update_yaxes(fixedrange=False)

        return fig, fig_map, main_style, graph_container_style

    # 7. Sync Cursor Map to Time (Client-side for performance)
    app.clientside_callback(
        """
        function(hoverData, mapFig) {
            if (!hoverData || !mapFig || !mapFig.data || !hoverData.points) {
                return window.dash_clientside.no_update;
            }
            const time = hoverData.points[0].x;

            let cursorX = null;
            let cursorY = null;

            for (let i = 0; i < mapFig.data.length; i++) {
                let trace = mapFig.data[i];
                if (!trace.customdata) continue;

                let minDiff = Infinity;
                let closestIdx = -1;
                for (let j = 0; j < trace.customdata.length; j++) {
                    let t_val = Array.isArray(trace.customdata[j]) ? trace.customdata[j][0] : trace.customdata[j];
                    let diff = Math.abs(t_val - time);
                    if (diff < minDiff) {
                        minDiff = diff;
                        closestIdx = j;
                    }
                }

                if (closestIdx >= 0 && minDiff < 0.5) {
                    cursorX = trace.x[closestIdx];
                    cursorY = trace.y[closestIdx];
                    break;
                }
            }

            if (cursorX !== null && cursorY !== null) {
                let newFig = JSON.parse(JSON.stringify(mapFig)); // Deep copy to ensure Dash detects update

                // Add Crosshair Lines
                if (!newFig.layout) newFig.layout = {};
                newFig.layout.shapes = [
                    {
                        type: 'line', xref: 'x', yref: 'paper',
                        x0: cursorX, x1: cursorX, y0: 0, y1: 1,
                        line: {color: 'rgba(255,0,0,0.5)', width: 2, dash: 'dot'}
                    },
                    {
                        type: 'line', xref: 'paper', yref: 'y',
                        x0: 0, x1: 1, y0: cursorY, y1: cursorY,
                        line: {color: 'rgba(255,0,0,0.5)', width: 2, dash: 'dot'}
                    }
                ];

                // Add Tooltip Annotation
                newFig.layout.annotations = [
                    {
                        x: cursorX,
                        y: cursorY,
                        xref: 'x',
                        yref: 'y',
                        showarrow: true,
                        arrowhead: 2,
                        arrowsize: 1,
                        arrowwidth: 2,
                        arrowcolor: 'red',
                        ax: 40,
                        ay: -40,
                        text: 'T: ' + time.toFixed(3) + 's<br>X: ' + cursorX.toFixed(2) + '<br>Y: ' + cursorY.toFixed(2),
                        font: {family: 'monospace', size: 12, color: 'white'},
                        bgcolor: 'rgba(0, 0, 0, 0.75)',
                        bordercolor: 'red',
                        borderwidth: 1,
                        borderpad: 4,
                        opacity: 1.0,
                        align: 'left'
                    }
                ];

                return newFig;
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("map-graph", "figure", allow_duplicate=True),
        Input("main-graph", "hoverData"),
        State("map-graph", "figure"),
        prevent_initial_call=True
    )

    return app
