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
    app.run_server(debug=True)
