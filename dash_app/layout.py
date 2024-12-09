from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

def create_header():
    return dbc.Navbar(
        dbc.Container(
            [
                dbc.NavbarBrand("Spotify Podcast Analysis"),
                dbc.Nav(
                    [
                        dbc.NavItem(
                            dbc.Button(
                                "Podcast Clustering",
                                id="btn-tab-cluster",
                                color="dark",
                                style={"border": "1px solid #ddd"},
                            )
                        ),
                    ],
                    className="ms-auto",
                ),
            ]
        ),
        color="primary",
        dark=True,
    )

def create_sidebar():
    return dbc.Col(
        [
            html.H3("Spotify Tools", className="text-center mt-3"),
            html.Hr(),
            dbc.InputGroup(
                [
                    dbc.InputGroupText("\ud83c\udfa7"),
                    dbc.Input(
                        id="input-podcast",
                        placeholder="Enter Podcast Name",
                        type="text",
                    ),
                ],
                className="mb-3",
            ),
            dbc.Button(
                "Search Podcast",
                id="btn-search-podcast",
                color="success",
                className="w-100 mb-3",
            ),
            html.Div(id="search-results"),
        ],
        width=3,
        style={
            "background-color": "#f8f9fa",
            "height": "100%",
            "padding": "15px",
            "box-shadow": "2px 0px 5px rgba(0, 0, 0, 0.1)",
        },
    )

def create_content():
    return dbc.Col(
        [
            dcc.Tabs(
                [
                    dcc.Tab(
                        label="Podcast Clustering",
                        children=[
                            html.Div(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader("Select Podcast"),
                                                        dbc.CardBody(
                                                            dcc.Dropdown(
                                                                id="podcast-dropdown",
                                                                placeholder="Choose a Podcast",
                                                            )
                                                        ),
                                                    ]
                                                ),
                                                width=6,
                                            ),
                                            dbc.Col(
                                                dbc.Card(
                                                    [
                                                        dbc.CardHeader("Select PCA Features for Plot"),
                                                        dbc.CardBody(
                                                            dbc.Row(
                                                                [
                                                                    dbc.Col(
                                                                        dcc.Dropdown(
                                                                            id="pca-x-feature-dropdown",
                                                                            options=[
                                                                                {"label": f"PCA Feature {i+1}", "value": i}
                                                                                for i in range(50)  # Adjust based on PCA model
                                                                            ],
                                                                            placeholder="X-axis PCA Feature",
                                                                        ),
                                                                        width=6,
                                                                    ),
                                                                    dbc.Col(
                                                                        dcc.Dropdown(
                                                                            id="pca-y-feature-dropdown",
                                                                            options=[
                                                                                {"label": f"PCA Feature {i+1}", "value": i}
                                                                                for i in range(50)
                                                                            ],
                                                                            placeholder="Y-axis PCA Feature",
                                                                        ),
                                                                        width=6,
                                                                    ),
                                                                ]
                                                            )
                                                        ),
                                                    ]
                                                ),
                                                width=6,
                                            ),
                                        ],
                                    ),
                                    html.Br(),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Card(
                                                    dbc.CardBody(
                                                        dcc.Graph(id="metrics-visualization")
                                                    )
                                                ),
                                                width=12,
                                            ),
                                        ]
                                    ),
                                    html.Br(),
                                    html.H3("Nearest Podcasts/Episodes", className="text-center"),
                                    dash_table.DataTable(
                                        id="nearest-table",
                                        columns=[
                                            {"name": "Podcast", "id": "Podcast"},
                                            {"name": "Episode", "id": "Episode"},
                                            {"name": "Distance", "id": "Distance"},
                                        ],
                                        style_table={"overflowY": "auto", "overflowX": "auto"},
                                        style_cell={"textAlign": "left", "padding": "10px"},
                                    ),
                                ],
                                style={"padding": "20px"},
                            )
                        ],
                    )
                ]
            )
        ],
        width=9,
    )

def create_layout():
    return dbc.Container(
        [
            dcc.Store(id="clustering-data-store"),
            create_header(),
            dbc.Row(
                [
                    create_sidebar(),
                    create_content(),
                ],
                style={"height": "100vh"},
            ),
        ],
        fluid=True,
    )
