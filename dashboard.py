import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import plotly.graph_objects as go
import numpy as np
import scipy.io
import spectral.io as spio
import os
from PIL import Image
import rasterio
from tkinter import filedialog
import tkinter as tk
import base64
import io
from dash.long_callback import DiskcacheLongCallbackManager
import diskcache

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# CSS for dark theme
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Dark theme styles */
            [data-theme="dark"] .radio-items input[type="radio"] {
                accent-color: #3498db;
            }
            [data-theme="dark"] .radio-items label {
                color: #ffffff;
            }
            [data-theme="dark"] .rc-slider {
                background-color: transparent;
            }
            [data-theme="dark"] .rc-slider-rail {
                background-color: #444444;
            }
            [data-theme="dark"] .rc-slider-track {
                background-color: #3498db;
            }
            [data-theme="dark"] .rc-slider-handle {
                border-color: #3498db;
                background-color: #3498db;
            }
            [data-theme="dark"] .rc-slider-mark-text {
                color: #ffffff;
            }
            [data-theme="dark"] .rc-slider-dot {
                border-color: #444444;
                background-color: #444444;
            }
            [data-theme="dark"] .rc-slider-dot-active {
                border-color: #3498db;
                background-color: #3498db;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Style definitions with theme support
STYLE = {
    'container': {
        'margin': '20px',
        'padding': '20px',
        'fontFamily': 'Arial',
        'maxWidth': '1200px',
        'margin': 'auto'
    },
    'image-container': {
        'position': 'relative',
        'width': '100%',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'center',
        'gap': '10px'
    },
    'button': {
        'margin': '5px',
        'padding': '8px 15px',
        'cursor': 'pointer',
        'backgroundColor': '#3498db',
        'color': 'white',
        'border': 'none',
        'borderRadius': '5px',
        'transition': 'background-color 0.3s'
    },
    'button-hover': {
        'backgroundColor': '#2980b9'
    },
    'section': {
        'marginBottom': '20px',
        'padding': '20px',
        'borderRadius': '8px',
        'backgroundColor': '#f8f9fa',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    },
    'input': {
        'padding': '8px',
        'borderRadius': '4px',
        'border': '1px solid #ddd',
        'marginRight': '10px'
    },
    'label': {
        'fontWeight': 'bold',
        'marginBottom': '5px',
        'display': 'block'
    },
    'tooltip': {
        'position': 'relative',
        'display': 'inline-block',
        'marginLeft': '5px',
        'cursor': 'help'
    },
    'slider-component': {
        'width': '200px',
        'margin': '0 10px'
    }

}

STYLE.update({
    'header': {
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'marginBottom': '10px',
        'padding': '10px 0',
        'borderBottom': '2px solid #eee'
    },
    'header-title': {
        'textAlign': 'left',
        'color': '#2c3e50',
        'margin': '0',
        'flex': '1',
        'fontSize': '24px'
    }
})

STYLE['container'].update({
    'transition': 'background-color 0.3s, color 0.3s'
})

# Theme definitions
LIGHT_THEME = {
    'backgroundColor': '#ffffff',
    'color': '#000000',
    'padding': '20px',
    'minHeight': '100vh'
}

DARK_THEME = {
    'backgroundColor': '#1a1a1a',
    'color': '#ffffff',
    'padding': '20px',
    'minHeight': '100vh'
}

DARK_STYLE = {
    'section': {
        'marginBottom': '20px',
        'padding': '20px',
        'borderRadius': '8px',
        'backgroundColor': '#2d2d2d',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.2)',
        'color': '#ffffff'
    },
    'panel': {  # Add this new style for panels
        'backgroundColor': '#1a1a1a',
        'padding': '20px',
        'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.2)',
        'color': '#ffffff'
    },
    'button': {
        'margin': '5px',
        'padding': '8px 15px',
        'cursor': 'pointer',
        'backgroundColor': '#3498db',
        'color': 'white',
        'border': 'none',
        'borderRadius': '5px',
        'transition': 'background-color 0.3s'
    },
    'input': {
        'padding': '8px',
        'borderRadius': '4px',
        'border': '1px solid #444',
        'marginRight': '10px',
        'backgroundColor': '#333',
        'color': '#ffffff'
    },
    'label': {
        'fontWeight': 'bold',
        'marginBottom': '5px',
        'display': 'block',
        'color': '#ffffff'
    }
}

DARK_STYLE['button'].update({
    'transition': 'all 0.3s ease',
    'hover': {
        'backgroundColor': '#2980b9',
        'boxShadow': '0 4px 8px rgba(0,0,0,0.2)'
    }
})

# Helper functions
def validate_wavelength_input(start_wl, end_wl):
    """Validate wavelength input values."""
    if start_wl is None or end_wl is None:
        return False, "Both wavelength values must be provided"
    if start_wl >= end_wl:
        return False, "Start wavelength must be less than end wavelength"
    if start_wl < 0 or end_wl < 0:
        return False, "Wavelengths must be positive"
    return True, ""

def load_data(path, format):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        if format == 'npy':
            data = np.load(path, allow_pickle=True)
            if not isinstance(data, np.ndarray):
                raise ValueError("Loaded NPY file does not contain a numpy array")
            return data
        elif format == 'mat':
            mat_data = scipy.io.loadmat(path)
            return max((v for k, v in mat_data.items()
                       if isinstance(v, np.ndarray) and len(v.shape) == 3),
                      key=lambda x: x.size)
        elif format == 'hdr':
            return spio.envi.open(path).load()
        elif format == 'tif':
            with rasterio.open(path) as src:
                return src.read()
        else:
            raise ValueError(f"Unsupported file format: {format}")
    except Exception as e:
        raise Exception(f"Error loading {format} file: {str(e)}")

def normalize_image(image_data):
    """Normalize image data to [0, 1] range."""
    min_val = np.min(image_data)
    max_val = np.max(image_data)
    if max_val == min_val:
        return np.zeros_like(image_data)
    return (image_data - min_val) / (max_val - min_val)

def enhance_image(image_data, contrast=1.0, brightness=0.0):
    """Apply contrast and brightness adjustments to image."""
    enhanced = image_data * contrast + brightness
    return np.clip(enhanced, 0, 1)

# Layout
app.layout = html.Div(id='container', children=[
    # Header
    html.Div([
        # Title and Theme Buttons in one line
        html.Div([
            # Left side - Title
            html.H1("Hyperspectral Image Analysis Dashboard",
                    style={
                        'textAlign': 'left',
                        'color': '#2c3e50',
                        'margin': '0',
                        'flex': '1',
                        'fontSize': '24px'
                    }),
            # Right side - Theme Buttons (only in setup section)
            html.Div([
                html.Button('Light Theme',
                            id='light-theme',
                            style={**STYLE['button'], 'marginRight': '10px'}),
                html.Button('Dark Theme',
                            id='dark-theme',
                            style=STYLE['button']),
            ], style={
                'display': 'flex',
                'alignItems': 'center'
            })
        ], style={
            'display': 'flex',
            'justifyContent': 'space-between',
            'alignItems': 'center',
            'marginBottom': '10px',
            'padding': '10px 0',
            'borderBottom': '2px solid #eee'
        }),
    ]),

    # Initial Setup Section
    html.Div([
        html.Div([
            # Left Panel - File and Format Selection
            html.Div([
                # File Path Selection
                html.Div([
                    html.Label("HSI Data Path:", style=STYLE['label']),
                    html.Div([
                        html.Button('Select Folder', id='folder-select',
                                    style={**STYLE['button'], 'marginRight': '10px'}),
                        html.Div(id='selected-path', style={'display': 'inline-block'}),
                        dcc.Loading(id='loading-path', type='circle')
                    ], style={'display': 'flex', 'alignItems': 'center'}),
                ], style={'marginBottom': '20px'}),

                # File Format Selection
                html.Div([
                    html.Label("File Format:", style=STYLE['label']),
                    html.Div([
                        dcc.RadioItems(
                            id='file-format',
                            options=[
                                {'label': ' NPY (.npy) ', 'value': 'npy'},
                                {'label': ' MATLAB (.mat) ', 'value': 'mat'},
                                {'label': ' HSD (.hsd) ', 'value': 'hsd'},
                                {'label': ' TIFF (.tif) ', 'value': 'tif'},
                                {'label': ' ENVI (.hdr) ', 'value': 'hdr'},
                                {'label': ' RAW (.raw) ', 'value': 'raw'}
                            ],
                            value='npy',
                            style={'margin': '10px 0'},
                            className='radio-items'
                        ),
                        html.Span("ⓘ", id='format-tooltip', style=STYLE['tooltip']),
                        dcc.Tooltip(
                            id='format-tooltip-content',
                            children='Select the format of your hyperspectral image file'
                        )
                    ])
                ], id='file-format-panel', style={
                    'width': '30%',
                    'padding': '20px',
                    'backgroundColor': '#f8f9fa',
                    'borderRadius': '8px',
                    'marginRight': '20px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'height': 'fit-content'
                }),

                # Wavelength Input Section
                html.Div([
                    html.Label("Spectral Range:", style=STYLE['label']),
                    dcc.Input(
                        id='start-wavelength',
                        type='number',
                        placeholder='Start wavelength (nm)',
                        style={
                            **STYLE['input'],
                            'width': '150px',
                        }
                    ),
                    dcc.Input(
                        id='end-wavelength',
                        type='number',
                        placeholder='End wavelength (nm)',
                        style={**STYLE['input'], 'width': '150px'}
                    ),
                    html.Div(id='wavelength-error', style={'color': 'red', 'marginTop': '5px'})
                ], id='wavelength-inputs'),
            ], style={
                'width': '30%',
                'padding': '20px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '8px',
                'marginRight': '20px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                'height': 'fit-content'
            }),

            # Right Panel - Preview and Dimension Order
            html.Div([
                # Dimension Order Selection
                html.Div([
                    html.Label("Dimension Order:", style=STYLE['label']),
                    html.Div([
                        dcc.RadioItems(
                            id='dim-order',
                            options=[
                                {'label': ' [C, H, W] ', 'value': 'chw'},
                                {'label': ' [C, W, H] ', 'value': 'cwh'},
                                {'label': ' [H, W, C] ', 'value': 'hwc'},
                                {'label': ' [W, H, C] ', 'value': 'whc'}
                            ],
                            value='chw',
                            style={
                                'margin': '10px 0',
                            }),
                        html.Span("ⓘ", id='dim-order-tooltip', style=STYLE['tooltip']),
                        dcc.Tooltip(
                            id='dim-order-tooltip-content',
                            children='C=Channels, H=Height, W=Width'
                        )
                    ]),
                ], id='preview-panel', style={
                    'width': '65%',
                    'padding': '20px',
                    'backgroundColor': '#f8f9fa',
                    'borderRadius': '8px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
                }),

                # Preview Image
                html.Div([
                    html.Label("Preview:", style=STYLE['label']),
                    html.Div(
                        style={
                            'height': '400px',
                            'overflow': 'hidden',
                            'backgroundColor': '#ffffff',
                            'borderRadius': '4px',
                            'padding': '10px',
                            'boxShadow': 'inset 0 0 5px rgba(0,0,0,0.1)'
                        },
                        children=[
                            html.Div(
                                id='preview-loading',
                                children=[
                                    dcc.Graph(
                                        id='preview-image',
                                        style={'height': '100%'},
                                        config={
                                            'displayModeBar': True,
                                            'scrollZoom': True,
                                            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
                                        }
                                    )
                                ]
                            )
                        ]
                    )
                ]),
            ], style={
                'width': '65%',
                'padding': '20px',
                'backgroundColor': '#f8f9fa',
                'borderRadius': '8px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
            }),
        ], style={
            'display': 'flex',
            'flexDirection': 'row',
            'marginBottom': '20px',
            'gap': '20px'
        }),

        # Load Button (outside the panels)
        html.Div([
            html.Button(
                'Load Data',
                id='load-button',
                style={
                    **STYLE['button'],
                    'backgroundColor': '#2ecc71',
                    'width': '100%',
                    'padding': '15px',
                    'fontSize': '16px',
                    'fontWeight': 'bold',
                    'transition': 'all 0.3s ease',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'hover': {
                        'backgroundColor': '#27ae60',
                        'boxShadow': '0 4px 8px rgba(0,0,0,0.2)'
                    }
                }
            ),
            html.Div(id='load-error', style={'color': 'red', 'marginTop': '10px', 'textAlign': 'center'})
        ], style={'padding': '0 20px'})
        ], id='setup-section', style={
            **STYLE['section'],
            'backgroundColor': '#ffffff',
            'padding': '30px',
            'margin': '20px auto',
            'maxWidth': '1400px'
        }),

        # Main Analysis Section
        html.Div([
            # Top Section - Controls and Image side by side
            html.Div([
                # Left Panel - Controls
                html.Div([
                    html.H4("Image Controls", style={'marginBottom': '15px'}),
                    # Image Enhancement Controls
                    html.Div([
                        html.Label("Enhancement:", style=STYLE['label']),
                        html.Div([
                            html.Label("Contrast:", style={'marginBottom': '5px'}),
                            html.Div(
                                dcc.Slider(
                                    id='contrast-slider',
                                    min=0.1,
                                    max=3.0,
                                    step=0.1,
                                    value=1.0,
                                    marks={i / 10: str(i / 10) for i in range(1, 31, 5)}
                                ),
                                id='contrast-slider-container'
                            ),
                            html.Label("Brightness:", style={'marginTop': '10px', 'marginBottom': '5px'}),
                            html.Div(
                                dcc.Slider(
                                    id='brightness-slider',
                                    min=-1.0,
                                    max=1.0,
                                    step=0.1,
                                    value=0,
                                    marks={i / 10: str(i / 10) for i in range(-10, 11, 5)}
                                ),
                                id='brightness-slider-container'
                            )
                        ], style={'width': '100%', 'marginBottom': '20px'}),
                    ]),

                    # Image Orientation Controls
                    html.Div([
                        html.Label("Orientation:", style=STYLE['label']),
                        html.Button('Vertical Flip', id='vertical-flip',
                                  style={**STYLE['button'], 'width': '100%', 'marginBottom': '5px'}),
                        html.Button('Horizontal Flip', id='horizontal-flip',
                                  style={**STYLE['button'], 'width': '100%', 'marginBottom': '5px'}),
                        html.Button('Rotate 90°', id='rotate-90',
                                  style={**STYLE['button'], 'width': '100%'})
                    ])
                ], style={
                    'width': '20%',
                    'padding': '15px',
                    'backgroundColor': '#f8f9fa',
                    'borderRadius': '8px',
                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                    'marginRight': '20px'
                }),

                # Right Panel - Image Display
                html.Div([
                    html.Div([
                        html.Button('←', id='prev-channel', style=STYLE['button']),
                        html.Div([
                            html.H4(id='dimension-info', style={'textAlign': 'center', 'fontSize': '0.9em'}),
                            html.H4(id='channel-info', style={'textAlign': 'center'}),
                            dcc.Loading(
                                id='image-loading',
                                type='circle',
                                children=[
                                    dcc.Graph(id='hsi-image',
                                             config={'displayModeBar': True, 'scrollZoom': True},
                                             style={'height': '45vh'})  # Reduced height
                                ]
                            )
                        ], style={'flex': '1'}),
                        html.Button('→', id='next-channel', style=STYLE['button'])
                    ], style={'display': 'flex', 'alignItems': 'center'})
                ], style={'width': '75%'})
            ], style={
                'display': 'flex',
                'flexDirection': 'row',
                'marginBottom': '10px',
                'height': '55vh'
            }),

            # Bottom Section - Spectral Plot
            html.Div([
                dcc.Loading(
                    id='plot-loading',
                    type='circle',
                    children=[
                        dcc.Graph(id='spectral-plot', style={'height': '28vh'})
                    ]
                ),
                html.Div([
                    html.Button('Undo', id='undo-button',
                               style={'backgroundColor': '#e74c3c', 'color': 'white', **STYLE['button']}),
                    html.Button('Clear', id='clear-button',
                               style={'backgroundColor': '#e67e22', 'color': 'white', **STYLE['button']}),
                    html.Button('Export Data', id='export-button',
                               style={'backgroundColor': '#3498db', 'color': 'white', **STYLE['button']})
                ], style={'textAlign': 'center'})
            ], style={'height': '28vh'})  # Adjusted height
        ], id='analysis-section', style={
            'display': 'none',
            'height': '90vh',
            'padding': '10px',
            'overflow': 'hidden'
        }),

    # Store components
    dcc.Store(id='hsi-data'),
    dcc.Store(id='current-channel', data=0),
    dcc.Store(id='clicked-points', data=[]),
    dcc.Store(id='wavelength-data'),
    dcc.Store(id='theme', data='light'),
    dcc.Download(id='download-data'),
], style=LIGHT_THEME)

@callback(
    [Output('setup-section', 'style'),
     Output('analysis-section', 'style'),
     Output('wavelength-inputs', 'style'),
     Output('dimension-info', 'style'),
     Output('channel-info', 'style')],
    Input('theme', 'data'),
    prevent_initial_call=True
)
def update_panel_styles(theme):
    is_dark = theme == 'dark'

    # Define panel style based on theme
    panel_style = {
        'backgroundColor': '#1a1a1a' if is_dark else '#f8f9fa',
        'padding': '20px',
        'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.2)',
        'color': '#ffffff' if is_dark else '#000000'
    }

    # Setup section style
    base_section_style = DARK_STYLE['section'] if is_dark else STYLE['section']
    setup_style = {
        **base_section_style,
        'display': 'block',
        'backgroundColor': '#2d2d2d' if is_dark else '#ffffff',
        'padding': '30px',
        'margin': '20px auto',
        'maxWidth': '1400px'
    }

    # Analysis section style
    analysis_style = {
        'display': 'none',
        'height': '90vh',
        'padding': '10px',
        'overflow': 'hidden',
        'backgroundColor': '#2d2d2d' if is_dark else '#ffffff',
        'color': '#ffffff' if is_dark else '#000000'
    }

    # Wavelength inputs style
    wavelength_style = {
        'backgroundColor': '#1a1a1a' if is_dark else '#ffffff',
        'padding': '15px',
        'borderRadius': '8px',
        'color': '#ffffff' if is_dark else '#000000'
    }

    # Text style for info elements
    text_style = {
        'color': '#ffffff' if is_dark else '#000000',
        'textAlign': 'center'
    }

    return setup_style, analysis_style, wavelength_style, text_style, text_style

@callback(
    [Output('file-format-panel', 'style'),
     Output('preview-panel', 'style'),
     Output('spectral-plot', 'style')],
    Input('theme', 'data'),
    prevent_initial_call=True
)
def update_panel_backgrounds(theme):
    is_dark = theme == 'dark'
    panel_style = {
        'backgroundColor': '#1a1a1a' if is_dark else '#f8f9fa',
        'padding': '20px',
        'borderRadius': '8px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.2)',
        'color': '#ffffff' if is_dark else '#000000'
    }
    return panel_style, panel_style, panel_style

@callback(
    [Output('loading-path', 'style'),
     Output('image-loading', 'style'),
     Output('plot-loading', 'style')],
    Input('theme', 'data'),
    prevent_initial_call=True
)
def update_loading_styles(theme):
    loading_style = {
        'color': '#ffffff' if theme == 'dark' else '#000000'
    }
    return loading_style, loading_style, loading_style


@callback(
    [Output('contrast-slider-container', 'style'),
     Output('brightness-slider-container', 'style')],
    Input('theme', 'data'),
    prevent_initial_call=True
)
def update_slider_styles(theme):
    slider_container_style = {
        'backgroundColor': '#333' if theme == 'dark' else '#ffffff',
        'padding': '10px',
        'borderRadius': '4px',
        'marginBottom': '10px'
    }
    return slider_container_style, slider_container_style

@callback(
    [Output('start-wavelength', 'style'),
     Output('end-wavelength', 'style'),
     Output('dim-order', 'style'),
     Output('file-format', 'style')],
    Input('theme', 'data'),
    prevent_initial_call=True
)
def update_input_styles(theme):
    input_style = {
        **STYLE['input'],
        'width': '150px',
        'backgroundColor': '#333' if theme == 'dark' else '#ffffff',
        'color': '#ffffff' if theme == 'dark' else '#000000'
    }

    radio_style = {
        'margin': '10px 0',
        'color': '#ffffff' if theme == 'dark' else '#000000'
    }

    return input_style, input_style, radio_style, radio_style
def create_dark_theme_layout():
    return {
        'plot_bgcolor': '#2d2d2d',
        'paper_bgcolor': '#2d2d2d',
        'font': {'color': '#ffffff'},
        'xaxis': {'gridcolor': '#444444', 'zerolinecolor': '#444444'},
        'yaxis': {'gridcolor': '#444444', 'zerolinecolor': '#444444'}
    }

# Callback for theme switching
@callback(
    [Output('container', 'style'),
     Output('container', 'data-theme'),
     Output('theme', 'data')],
    [Input('light-theme', 'n_clicks'),
     Input('dark-theme', 'n_clicks')],
    [State('theme', 'data')],
    prevent_initial_call=True
)
def update_theme(light_clicks, dark_clicks, current_theme):
    trigger_id = ctx.triggered_id

    if trigger_id == 'light-theme':
        return LIGHT_THEME, 'light', 'light'
    elif trigger_id == 'dark-theme':
        return DARK_THEME, 'dark', 'dark'
    return dash.no_update, dash.no_update, dash.no_update

# Callback for folder selection
@callback(
    [Output('selected-path', 'children'),
     Output('loading-path', 'children')],
    Input('folder-select', 'n_clicks'),
    prevent_initial_call=True
)
def select_folder(n_clicks):
    try:
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        folder_path = filedialog.askdirectory(master=root)
        root.destroy()
        if not folder_path:
            return "No folder selected", ""
        return folder_path, ""
    except Exception as e:
        return f"Error: {str(e)}", ""

# Callback for preview image
@callback(
    Output('preview-image', 'figure'),
    [Input('selected-path', 'children'),
     Input('dim-order', 'value'),
     Input('file-format', 'value'),
     Input('theme', 'data')],
    prevent_initial_call=True
)
def update_preview(path, dim_order, format, theme):
    if path == "No folder selected":
        return go.Figure()

    try:
        # Find the first file with the specified format
        if os.path.isdir(path):
            files = [f for f in os.listdir(path) if f.endswith(f'.{format}')]
            if not files:
                raise Exception(f"No .{format} files found")
            file_path = os.path.join(path, files[0])
        else:
            file_path = path

        data = load_data(file_path, format)

        # Transform data based on dimension order
        if dim_order == 'chw':
            preview = data[0, :, :]
        elif dim_order == 'cwh':
            preview = data[0, :, :].T
        elif dim_order == 'hwc':
            preview = data[:, :, 0]
        else:  # whc
            preview = data[:, :, 0].T

        preview = normalize_image(preview)

        fig = go.Figure(data=go.Heatmap(
            z=preview,
            colorscale='Gray',
            showscale=True
        ))
        fig.update_layout(
            title='Preview (First Channel)',
            margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        fig = apply_theme_to_figure(fig, theme)
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig = apply_theme_to_figure(fig, theme)
        return fig

# Main data loading callback
@app.long_callback(
    output=[
        Output('hsi-data', 'data'),
        Output('dimension-info', 'children'),
        Output('setup-section', 'style', allow_duplicate=True),
        Output('analysis-section', 'style', allow_duplicate=True),
        Output('load-error', 'children'),
        Output('wavelength-data', 'data')
    ],
    inputs=[
        Input('load-button', 'n_clicks')
    ],
    state=[
        State('selected-path', 'children'),
        State('file-format', 'value'),
        State('dim-order', 'value'),
        State('start-wavelength', 'value'),
        State('end-wavelength', 'value')
    ],
    manager=long_callback_manager,
    prevent_initial_call=True
)
def load_hsi_data(n_clicks, path, format, dim_order, start_wl, end_wl):
    if path == "No folder selected":
        return [dash.no_update] * 6

    try:
        # Validate wavelength inputs
        is_valid, message = validate_wavelength_input(start_wl, end_wl)
        if not is_valid and (start_wl is not None or end_wl is not None):
            raise ValueError(message)

        # Find and load the file
        if os.path.isdir(path):
            files = [f for f in os.listdir(path) if f.endswith(f'.{format}')]
            if not files:
                raise Exception(f"No .{format} files found in directory")
            file_path = os.path.join(path, files[0])
        else:
            file_path = path

        data = load_data(file_path, format)
        original_shape = data.shape

        # Check for wavelength information in metadata
        wavelength_data = None
        if format == 'hdr':
            try:
                with open(file_path.replace('.hdr', '.hdr'), 'r') as f:
                    header = f.read()
                    if 'wavelength' in header.lower():
                        wavelength_data = {'start': start_wl, 'end': end_wl}
            except:
                pass

        # If no wavelength data in metadata, use user input
        if wavelength_data is None and start_wl is not None and end_wl is not None:
            wavelength_data = {'start': start_wl, 'end': end_wl}

        # Standardize to [H, W, C] format
        if dim_order == 'chw':
            data = np.transpose(data, (1, 2, 0))
        elif dim_order == 'cwh':
            data = np.transpose(data, (2, 1, 0))
        elif dim_order == 'whc':
            data = np.transpose(data, (1, 0, 2))

        dim_info = (f"Original dimensions: {original_shape} ({dim_order}) → "
                   f"Standardized [H, W, C]: {data.shape}")

        return (data.tolist(), dim_info,
                {'display': 'none'}, {'display': 'block'}, "",
                wavelength_data)

    except Exception as e:
        return [dash.no_update] * 5 + [f"Error: {str(e)}"]

# Image enhancement callback
@callback(
    Output('hsi-image', 'figure', allow_duplicate=True),
    [Input('contrast-slider', 'value'),
     Input('brightness-slider', 'value'),
     Input('theme', 'data')],
    [State('hsi-data', 'data'),
     State('current-channel', 'data')],
    prevent_initial_call=True
)
def update_image_enhancement(contrast, brightness, theme, data, current_channel):
    if not data:
        return dash.no_update

    data = np.array(data)
    channel_data = data[:, :, current_channel]
    channel_data = normalize_image(channel_data)
    enhanced_data = enhance_image(channel_data, contrast, brightness)

    fig = go.Figure(data=go.Heatmap(
        z=enhanced_data,
        colorscale='Gray',
        showscale=True,
        hoverongaps=False
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    fig = apply_theme_to_figure(fig, theme)
    return fig

# Image orientation callback
@callback(
    [Output('hsi-data', 'data', allow_duplicate=True),
     Output('current-channel', 'data', allow_duplicate=True)],
    [Input('vertical-flip', 'n_clicks'),
     Input('horizontal-flip', 'n_clicks'),
     Input('rotate-90', 'n_clicks')],
    [State('hsi-data', 'data'),
     State('current-channel', 'data')],
    prevent_initial_call=True
)
def apply_orientation(v_flip, h_flip, rotate, data, current_channel):
    if not data:
        return dash.no_update, dash.no_update

    data = np.array(data)
    trigger_id = ctx.triggered_id

    if trigger_id == 'vertical-flip':
        data = np.flip(data, axis=0)
    elif trigger_id == 'horizontal-flip':
        data = np.flip(data, axis=1)
    elif trigger_id == 'rotate-90':
        data = np.rot90(data)

    return data.tolist(), current_channel

def cleanup_data():
    import gc
    gc.collect()

# Channel navigation and display callback
@callback(
    [Output('hsi-image', 'figure'),
     Output('channel-info', 'children'),
     Output('current-channel', 'data')],
    [Input('hsi-data', 'data'),
     Input('current-channel', 'data'),
     Input('prev-channel', 'n_clicks'),
     Input('next-channel', 'n_clicks'),
     Input('theme', 'data')],
    prevent_initial_call=True
)
def update_image(data, current_channel, prev_clicks, next_clicks, theme):
    if not data:
        return dash.no_update, dash.no_update, dash.no_update

    data = np.array(data)
    num_channels = data.shape[2]
    trigger_id = ctx.triggered_id

    if trigger_id == 'prev-channel' and current_channel > 0:
        current_channel -= 1
    elif trigger_id == 'next-channel' and current_channel < num_channels - 1:
        current_channel += 1

    channel_data = data[:, :, current_channel]
    channel_data = normalize_image(channel_data)

    fig = go.Figure(data=go.Heatmap(
        z=channel_data,
        colorscale='Gray',
        showscale=True,
        hoverongaps=False
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showticklabels=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    # Call cleanup_data() after creating the figure
    cleanup_data()

    fig = apply_theme_to_figure(fig, theme)
    return fig, f"Channel: {current_channel + 1} / {num_channels}", current_channel

# Spectral plot callback
@callback(
    [Output('spectral-plot', 'figure'),
     Output('clicked-points', 'data')],
    [Input('hsi-image', 'clickData'),
     Input('undo-button', 'n_clicks'),
     Input('clear-button', 'n_clicks'),
     Input('theme', 'data')],
    [State('hsi-data', 'data'),
     State('clicked-points', 'data'),
     State('wavelength-data', 'data')],
    prevent_initial_call=True
)

def update_spectral_plot(click_data, undo_clicks, clear_clicks, theme, hsi_data, clicked_points, wavelength_data):
    if not hsi_data:
        return dash.no_update, dash.no_update

    trigger_id = ctx.triggered_id
    data = np.array(hsi_data)

    if trigger_id == 'clear-button':
        clicked_points = []
    elif trigger_id == 'undo-button' and clicked_points:
        clicked_points.pop()
    elif trigger_id == 'hsi-image' and click_data:
        point = click_data['points'][0]
        x, y = point['x'], point['y']
        spectral_signature = data[int(y), int(x), :]
        clicked_points.append({
            'x': x,
            'y': y,
            'spectrum': spectral_signature.tolist()
        })

    fig = go.Figure()

    if wavelength_data:
        x_axis = np.linspace(wavelength_data['start'], wavelength_data['end'],
                           len(clicked_points[0]['spectrum']) if clicked_points else 0)
        x_label = 'Wavelength (nm)'
    else:
        x_axis = list(range(len(clicked_points[0]['spectrum']))) if clicked_points else []
        x_label = 'Channel'

    for i, point in enumerate(clicked_points):
        fig.add_trace(go.Scatter(
            y=point['spectrum'],
            x=x_axis if wavelength_data else list(range(len(point['spectrum']))),
            name=f"Point {i+1} ({int(point['x'])}, {int(point['y'])})",
            mode='lines'
        ))

    fig.update_layout(
        title='Spectral Signatures',
        xaxis_title=x_label,
        yaxis_title='Intensity',
        showlegend=True,
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )

    # Apply theme-specific layout
    if theme == 'dark':
        fig.update_layout(create_dark_theme_layout())

    return fig, clicked_points

# Export data callback
@callback(
    Output('download-data', 'data'),
    Input('export-button', 'n_clicks'),
    [State('clicked-points', 'data'),
     State('wavelength-data', 'data')],
    prevent_initial_call=True
)
def export_data(n_clicks, clicked_points, wavelength_data):
    if not clicked_points:
        return dash.no_update

    # Create CSV content
    import io
    import pandas as pd

    # Create wavelength or channel numbers for x-axis
    if wavelength_data:
        num_points = len(clicked_points[0]['spectrum'])
        x_values = np.linspace(wavelength_data['start'], wavelength_data['end'], num_points)
        x_label = 'Wavelength (nm)'
    else:
        x_values = range(len(clicked_points[0]['spectrum']))
        x_label = 'Channel'

    # Create DataFrame
    data_dict = {x_label: x_values}
    for i, point in enumerate(clicked_points):
        data_dict[f"Point {i+1} ({int(point['x'])}, {int(point['y'])})"] = point['spectrum']

    df = pd.DataFrame(data_dict)

    # Save to string buffer
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)

    return dict(
        content=buffer.getvalue(),
        filename='spectral_signatures.csv'
    )

def apply_theme_to_figure(fig, theme):
    if theme == 'dark':
        fig.update_layout(
            plot_bgcolor='#2d2d2d',
            paper_bgcolor='#2d2d2d',
            font={'color': '#ffffff'},
            xaxis={'gridcolor': '#444444', 'zerolinecolor': '#444444'},
            yaxis={'gridcolor': '#444444', 'zerolinecolor': '#444444'}
        )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
