import os
import dash
import dash_bootstrap_components as dbc
from layout import create_layout
from callbacks import register_callbacks

# Initialize Dash App
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Set Layout
app.layout = create_layout()

# Register Callbacks
register_callbacks(app)

# Run the App
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))  # Defaults to 8050 if PORT is not set
    app.run_server(debug=False, host="0.0.0.0", port=port)
