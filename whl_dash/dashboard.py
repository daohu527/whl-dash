import os
import dash
from dash import html, dcc, Input, Output, State, ALL, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

from data import RecordDataLoader
from template import TemplateManager, RowConfig, DashboardTemplate


def create_app(record_path: str):
    loader = RecordDataLoader(record_path)
    mgr = TemplateManager(os.path.join(os.path.dirname(__file__), "templates.json"))

    app = dash.Dash(__name__, title="Apollo Debugger")

    def get_template_options():
        return [
            {"label": tpl_name, "value": tpl_name}
            for tpl_name in sorted(mgr.templates.keys())
        ]

    # Generate reference text for available roots (to guide users writing formulas)
    roots = [
        k
        for k in loader.eval_env.keys()
        if k not in ("np", "math", "relative_time_sec")
    ]

    app.layout = html.Div(
        [
            html.Div(
                [
                    html.H3("Apollo Analyzer Core"),
                    html.Div(
                        "Record: " + os.path.basename(loader.record_path),
                        style={
                            "fontSize": "11px",
                            "color": "#666",
                            "marginBottom": "10px",
                        },
                    ),
                    html.Div("Templates:"),
                    dcc.Dropdown(
                        id="tpl-dropdown",
                        options=get_template_options(),
                        value="control_feedback",
                        clearable=False,
                    ),
                    html.Div(
                        [
                            dcc.Input(
                                id="tpl-name",
                                placeholder="Save as...",
                                style={"width": "150px"},
                            ),
                            html.Button(
                                "💾 Save", id="btn-save", style={"margin": "0 5px"}
                            ),
                            html.Button(
                                "🗑 Delete", id="btn-delete", style={"color": "red"}
                            ),
                        ],
                        style={"margin": "10px 0"},
                    ),
                    html.Hr(),
                    html.Div(
                        [
                            html.H4("Formula Editors", style={"margin": "0"}),
                            html.Div(
                                [
                                    html.Button("➕ Add Row", id="btn-add-row"),
                                    html.Button(
                                        "▶ Run & Render",
                                        id="btn-render",
                                        style={
                                            "background": "#28a745",
                                            "color": "white",
                                            "fontWeight": "bold",
                                            "marginLeft": "10px",
                                        },
                                    ),
                                ],
                                style={"marginTop": "5px"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "alignItems": "center",
                            "marginBottom": "10px",
                        },
                    ),
                    html.Div(id="rows-container"),
                    html.Hr(),
                    html.Details(
                        [
                            html.Summary(
                                f"Available Signals ({len(loader.available_signals)})"
                            ),
                            html.Div(
                                "Roots: " + ", ".join(roots) + " | Functions: np, math",
                                style={"fontSize": "10px", "marginBottom": "5px"},
                            ),
                            html.Pre(
                                "\n".join(loader.available_signals),
                                style={
                                    "height": "250px",
                                    "overflowY": "scroll",
                                    "fontSize": "11px",
                                    "background": "#f0f0f0",
                                    "padding": "5px",
                                },
                            ),
                        ]
                    ),
                ],
                style={
                    "width": "380px",
                    "padding": "15px",
                    "borderRight": "1px solid #ccc",
                    "height": "100vh",
                    "overflowY": "auto",
                    "display": "flex",
                    "flexDirection": "column",
                    "background": "#fafafa",
                },
            ),
            html.Div(
                [dcc.Graph(id="main-graph", style={"height": "95vh"})],
                style={"flex": "1", "padding": "10px"},
            ),
        ],
        style={"display": "flex", "height": "100vh", "margin": "-8px"},
    )

    @app.callback(
        Output("rows-container", "children"),
        Output("tpl-dropdown", "options"),
        Output("tpl-dropdown", "value"),
        Input("tpl-dropdown", "value"),
        Input("btn-save", "n_clicks"),
        Input("btn-delete", "n_clicks"),
        Input("btn-add-row", "n_clicks"),
        Input({"type": "btn-rem", "index": ALL}, "n_clicks"),
        State({"type": "title-in", "index": ALL}, "value"),
        State({"type": "signals-in", "index": ALL}, "value"),
        State("tpl-name", "value"),
        prevent_initial_call=False,
    )
    def handle_ui_state(
        sel_tpl, save_c, del_c, add_c, rem_c, titles, signals, tpl_name
    ):
        ctx = callback_context
        trig = ctx.triggered_id if ctx.triggered else "init"

        # Capture current rows from UI (each textarea is split by newline)
        current_rows = []
        if titles is not None and signals is not None:
            for t, s in zip(titles, signals):
                sig_list = [
                    x.strip()
                    for x in (s or "").replace(",", "\n").split("\n")
                    if x.strip() and not x.strip().startswith("#")
                ]
                current_rows.append(RowConfig(title=(t or "Row"), signals=sig_list))

        out_sel = sel_tpl
        new_rows = current_rows

        if trig == "init" or trig == "tpl-dropdown":
            tpl = mgr.templates.get(sel_tpl)
            if tpl:
                new_rows = tpl.rows
            out_sel = sel_tpl

        elif trig == "btn-add-row":
            new_rows.append(RowConfig(title="New Row", signals=["chassis.speed_mps"]))
            out_sel = dash.no_update

        elif isinstance(trig, dict) and trig.get("type") == "btn-rem":
            idx = trig.get("index")
            if 0 <= idx < len(new_rows):
                new_rows.pop(idx)
            out_sel = dash.no_update

        elif trig == "btn-save":
            name = (tpl_name or sel_tpl or "custom_template").strip()
            if name:
                mgr.templates[name] = DashboardTemplate(name, current_rows)
                mgr.save()
                out_sel = name

        elif trig == "btn-delete":
            if sel_tpl in mgr.templates:
                del mgr.templates[sel_tpl]
                mgr.save()
                out_sel = list(mgr.templates.keys())[0] if mgr.templates else None
                if out_sel:
                    new_rows = mgr.templates[out_sel].rows
                else:
                    new_rows = [RowConfig("Empty", [])]

        children = []
        for i, r in enumerate(new_rows):
            chunk = html.Div(
                [
                    html.Div(
                        [
                            dcc.Input(
                                type="text",
                                value=r.title,
                                id={"type": "title-in", "index": i},
                                style={
                                    "flex": "1",
                                    "marginRight": "5px",
                                    "fontWeight": "bold",
                                },
                            ),
                            html.Button(
                                "X",
                                id={"type": "btn-rem", "index": i},
                                style={
                                    "color": "white",
                                    "background": "#dc3545",
                                    "border": "none",
                                    "borderRadius": "3px",
                                    "cursor": "pointer",
                                },
                            ),
                        ],
                        style={"display": "flex", "marginBottom": "5px"},
                    ),
                    dcc.Textarea(
                        value="\n".join(r.signals),
                        id={"type": "signals-in", "index": i},
                        style={
                            "width": "100%",
                            "height": "50px",
                            "fontFamily": "monospace",
                        },
                    ),
                ],
                style={
                    "border": "1px solid #ccc",
                    "padding": "8px",
                    "marginBottom": "10px",
                    "background": "white",
                    "borderRadius": "4px",
                },
            )
            children.append(chunk)

        opts = get_template_options()
        return children, opts, out_sel

    @app.callback(
        Output("main-graph", "figure"),
        Input("btn-render", "n_clicks"),
        Input("tpl-dropdown", "value"),  # switch templates -> auto render
        State({"type": "title-in", "index": ALL}, "value"),
        State({"type": "signals-in", "index": ALL}, "value"),
    )
    def render_graph(n_clicks, tpl_change, titles, signals):
        if not titles:
            fig = make_subplots(rows=1, cols=1)
            fig.update_layout(template="plotly_white", title="No Data Rows")
            return fig

        num_rows = len(titles)
        fig = make_subplots(
            rows=num_rows,
            cols=1,
            shared_xaxes=True,
            subplot_titles=titles,
            vertical_spacing=0.035,
        )
        t = loader.master_df["relative_time_sec"].values

        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        for i, (title, sig_block) in enumerate(zip(titles, signals), start=1):
            sig_list = [
                x.strip()
                for x in (sig_block or "").replace(",", "\n").split("\n")
                if x.strip() and not x.strip().startswith("#")
            ]
            for trace_idx, expr in enumerate(sig_list):
                y = loader.evaluate(expr)
                if y is not None and len(y) == len(t):
                    # Check if categorical/step
                    valid_y = y[np.isfinite(y)]
                    unique_vals = np.unique(valid_y)
                    is_step = len(unique_vals) <= 12 and all(
                        (v % 1 == 0) for v in unique_vals
                    )

                    shape = "hv" if is_step else "linear"
                    c = colors[trace_idx % len(colors)]

                    fig.add_trace(
                        go.Scatter(
                            x=t,
                            y=y,
                            mode="lines",
                            name=expr,
                            line={"shape": shape, "color": c, "width": 1.5},
                            hovertemplate=f"{expr}<br>t=%{{x:.3f}}s<br>val=%{{y:.5f}}<extra></extra>",
                            showlegend=True,
                        ),
                        row=i,
                        col=1,
                    )

        fig.update_layout(
            height=max(800, 260 * num_rows),
            template="plotly_white",
            hovermode="x unified",
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.02,
                "xanchor": "right",
                "x": 1,
            },
        )
        fig.update_xaxes(title_text="Relative Time [s]", row=num_rows, col=1)
        return fig

    return app
