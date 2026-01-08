# -*- coding: utf-8 -*-
import math
from pathlib import Path

import pandas as pd
import networkx as nx

from dash import Dash, dcc, html, Input, Output, dash_table
import plotly.graph_objs as go


# ==============================================================================
# Helpers
# ==============================================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def read_csv_auto(path):
    try:
        return pd.read_csv(path, sep=',')
    except Exception:
        return pd.read_csv(path, sep=';')


def build_full_graph(stops_df, stop_times_df):
    stop_times_df = stop_times_df.sort_values(['trip_id', 'stop_sequence'])
    stop_times_df['next_stop_id'] = stop_times_df.groupby('trip_id')['stop_id'].shift(-1)
    edges_df = stop_times_df.dropna(subset=['next_stop_id'])
    unique_edges = edges_df[['stop_id', 'next_stop_id']].drop_duplicates()

    G_full = nx.Graph()
    G_full.add_edges_from(zip(unique_edges['stop_id'], unique_edges['next_stop_id']))

    cols_to_keep = ['stop_name', 'stop_lat', 'stop_lon']
    available_cols = [c for c in cols_to_keep if c in stops_df.columns]
    if available_cols:
        stops_attr = stops_df.set_index('stop_id')[available_cols].to_dict('index')
        filtered_attr = {k: v for k, v in stops_attr.items() if k in G_full.nodes()}
        nx.set_node_attributes(G_full, filtered_attr)

    return G_full


def build_spatial_index(G_full, dist_threshold_m):
    nodes_with_coords = []
    coords = {}
    lats = []

    for node, data in G_full.nodes(data=True):
        if 'stop_lat' not in data or 'stop_lon' not in data:
            continue
        lat = data['stop_lat']
        lon = data['stop_lon']
        coords[node] = (lat, lon)
        nodes_with_coords.append(node)
        lats.append(lat)

    if not nodes_with_coords:
        return {}, {}, {}

    lat0 = sum(lats) / len(lats)
    cos_lat0 = math.cos(math.radians(lat0))
    meters_per_deg = 111_320

    def to_xy(lat, lon):
        return (lon * cos_lat0 * meters_per_deg, lat * meters_per_deg)

    positions = {}
    cell_id = {}
    index = {}
    cell_size = dist_threshold_m

    for node in nodes_with_coords:
        lat, lon = coords[node]
        x, y = to_xy(lat, lon)
        positions[node] = (x, y)
        ix = int(x // cell_size)
        iy = int(y // cell_size)
        cell = (ix, iy)
        cell_id[node] = cell
        index.setdefault(cell, []).append(node)

    return coords, cell_id, index


def reduce_graph(G_full, dist_threshold_m):
    node_degrees = dict(G_full.degree())
    sorted_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)

    coords, cell_id, index = build_spatial_index(G_full, dist_threshold_m)
    if not coords:
        return nx.Graph(), {}

    mapping = {}
    processed = set()

    for pivot_node in sorted_nodes:
        if pivot_node in processed:
            continue
        if pivot_node not in coords:
            continue

        processed.add(pivot_node)
        mapping[pivot_node] = pivot_node

        pivot_lat, pivot_lon = coords[pivot_node]
        pivot_cell = cell_id[pivot_node]

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cell = (pivot_cell[0] + dx, pivot_cell[1] + dy)
                for candidate in index.get(cell, []):
                    if candidate in processed:
                        continue
                    cand_lat, cand_lon = coords[candidate]
                    if haversine(pivot_lat, pivot_lon, cand_lat, cand_lon) <= dist_threshold_m:
                        processed.add(candidate)
                        mapping[candidate] = pivot_node

    G_reduced = nx.Graph()
    for u, v in G_full.edges():
        new_u = mapping.get(u)
        new_v = mapping.get(v)
        if not new_u or not new_v or new_u == new_v:
            continue
        G_reduced.add_edge(new_u, new_v)

    for node in G_reduced.nodes():
        if node in G_full.nodes:
            G_reduced.nodes[node].update(G_full.nodes[node])

    return G_reduced, mapping


def closest_freguesia(lat, lon, freguesias_list):
    closest_name = None
    min_dist = None
    for name, (f_lat, f_lon) in freguesias_list:
        dist = haversine(lat, lon, f_lat, f_lon)
        if min_dist is None or dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name


def compute_freguesia_stats(G_reduced, node_df):
    summary_rows = []
    for freg in sorted(node_df['freguesia'].dropna().unique()):
        node_ids = node_df.loc[node_df['freguesia'] == freg, 'id'].tolist()
        G_freg = G_reduced.subgraph(node_ids)

        if len(G_freg) <= 1:
            radius = 0
            diameter = 0
            con_status = 'Single node'
        else:
            if nx.is_connected(G_freg):
                radius = nx.radius(G_freg)
                diameter = nx.diameter(G_freg)
                con_status = 'Connected'
            else:
                largest_cc = max(nx.connected_components(G_freg), key=len)
                G_lcc = G_freg.subgraph(largest_cc)
                radius = nx.radius(G_lcc)
                diameter = nx.diameter(G_lcc)
                con_status = f'Fragmented (LCC {len(G_lcc)}/{len(G_freg)})'

        sub = node_df[node_df['freguesia'] == freg]
        summary_rows.append({
            'freguesia': freg,
            'nodes': len(sub),
            'avg_degree': round(sub['degree'].mean(), 6),
            'avg_betweenness': round(sub['betweenness'].mean(), 6),
            'avg_closeness': round(sub['closeness'].mean(), 6),
            'radius': radius,
            'diameter': diameter,
            'connectivity': con_status,
        })

    return pd.DataFrame(summary_rows)


def build_network_figure(G_reduced, node_df, selected_freg, search_text):
    filtered_df = node_df.copy()

    if selected_freg and selected_freg != 'All':
        filtered_df = filtered_df[filtered_df['freguesia'] == selected_freg]

    if search_text:
        search = search_text.strip().lower()
        if search:
            filtered_df = filtered_df[filtered_df['name'].str.lower().str.contains(search, na=False)]

    filtered_ids = set(filtered_df['id'].tolist())
    pos_geo = {}
    for node, data in G_reduced.nodes(data=True):
        if node in filtered_ids and 'stop_lon' in data and 'stop_lat' in data:
            pos_geo[node] = (data['stop_lon'], data['stop_lat'])

    edge_x = []
    edge_y = []
    for u, v in G_reduced.edges():
        if u not in pos_geo or v not in pos_geo:
            continue
        edge_x.extend([pos_geo[u][0], pos_geo[v][0], None])
        edge_y.extend([pos_geo[u][1], pos_geo[v][1], None])

    edge_trace = go.Scattergl(
        x=edge_x,
        y=edge_y,
        mode='lines',
        line=dict(width=0.6, color='rgba(120,120,120,0.35)'),
        hoverinfo='skip',
        name='Connections'
    )

    node_trace = go.Scattergl(
        x=[pos_geo[n][0] for n in pos_geo.keys()],
        y=[pos_geo[n][1] for n in pos_geo.keys()],
        mode='markers',
        marker=dict(
            size=6,
            color=filtered_df['degree'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Degree')
        ),
        text=[
            f"{row['name']}<br>Freguesia: {row['freguesia']}<br>"
            f"Degree: {row['degree']:.5f}<br>"
            f"Betweenness: {row['betweenness']:.5f}<br>"
            f"Closeness: {row['closeness']:.5f}"
            for _, row in filtered_df.iterrows()
        ],
        hoverinfo='text',
        name='Hubs'
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title='Rede Carris Simplificada (por freguesia)',
        font=dict(family='Georgia, Palatino, "Times New Roman", serif', size=14, color='#2b2b2b'),
        paper_bgcolor='#f6f2ea',
        plot_bgcolor='#f6f2ea',
        xaxis=dict(title='Longitude', showgrid=False, zeroline=False),
        yaxis=dict(title='Latitude', showgrid=False, zeroline=False, scaleanchor='x', scaleratio=1),
        margin=dict(l=10, r=10, t=40, b=10),
        height=650,
        showlegend=False
    )

    return fig


def build_data():
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"

    file_stops = DATA_DIR / "stops.txt"
    file_times = DATA_DIR / "stop_times.txt"

    if not file_stops.exists():
        raise FileNotFoundError(f"Ficheiro não encontrado: {file_stops}")

    if not file_times.exists():
        raise FileNotFoundError(f"Ficheiro não encontrado: {file_times}")

    stops_df = read_csv_auto(file_stops)
    stop_times_df = read_csv_auto(file_times)

    G_full = build_full_graph(stops_df, stop_times_df)
    G_reduced, _ = reduce_graph(G_full, dist_threshold_m=150)

    deg_cent = nx.degree_centrality(G_reduced)
    bet_cent = nx.betweenness_centrality(G_reduced, normalized=True)
    clo_cent = nx.closeness_centrality(G_reduced)

    freguesias_coords = {
        "Ajuda": (38.7068, -9.2016), "Alcântara": (38.7058, -9.1775),
        "Alvalade": (38.7533, -9.1438), "Areeiro": (38.7423, -9.1305),
        "Arroios": (38.7290, -9.1345), "Avenidas Novas": (38.7366, -9.1486),
        "Beato": (38.7303, -9.1126), "Belém": (38.6974, -9.2188),
        "Benfica": (38.7513, -9.2012), "Campo de Ourique": (38.7188, -9.1663),
        "Campolide": (38.7317, -9.1631), "Carnide": (38.7656, -9.1906),
        "Estrela": (38.7126, -9.1584), "Lumiar": (38.7756, -9.1587),
        "Marvila": (38.7483, -9.1098), "Misericórdia": (38.7118, -9.1469),
        "Olivais": (38.7708, -9.1213), "Parque das Nações": (38.7675, -9.0945),
        "Penha de França": (38.7265, -9.1287), "Santa Clara": (38.7783, -9.1444),
        "Santa Maria Maior": (38.7144, -9.1360), "Santo António": (38.7236, -9.1492),
        "São Domingos de Benfica": (38.7456, -9.1764), "São Vicente": (38.7186, -9.1274),
        "Margem Sul / Ponte": (38.6700, -9.1670)
    }

    freguesias_list = list(freguesias_coords.items())

    node_rows = []
    for node, data in G_reduced.nodes(data=True):
        lat = data.get('stop_lat')
        lon = data.get('stop_lon')
        name = data.get('stop_name', str(node))
        freg = None
        if lat is not None and lon is not None:
            freg = closest_freguesia(lat, lon, freguesias_list)

        node_rows.append({
            'id': node,
            'name': name,
            'lat': lat,
            'lon': lon,
            'freguesia': freg,
            'degree': deg_cent.get(node, 0.0),
            'betweenness': bet_cent.get(node, 0.0),
            'closeness': clo_cent.get(node, 0.0),
        })

    node_df = pd.DataFrame(node_rows)
    summary_df = compute_freguesia_stats(G_reduced, node_df)

    return G_reduced, node_df, summary_df


# ==============================================================================
# Dash app
# ==============================================================================

G_REDUCED, NODE_DF, SUMMARY_DF = build_data()

app = Dash(__name__)
app.title = 'Rede Carris - Dashboard'

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --ink: #2b2b2b;
                --muted: #6a6a6a;
                --card: #ffffff;
                --accent: #9c6b3f;
                --paper: #f6f2ea;
                --border: #e6dccb;
            }
            body {
                margin: 0;
                background: var(--paper);
                color: var(--ink);
                font-family: Georgia, Palatino, "Times New Roman", serif;
                -webkit-font-smoothing: antialiased;
                text-rendering: optimizeLegibility;
            }
            .page {
                max-width: 1200px;
                margin: 0 auto;
                padding: 28px 20px 40px;
            }
            .title {
                font-size: 28px;
                letter-spacing: 0.3px;
                margin: 6px 0 4px;
            }
            .subtitle {
                color: var(--muted);
                margin: 0 0 18px;
                font-size: 15px;
            }
            .card {
                background: var(--card);
                border: 1px solid var(--border);
                border-radius: 14px;
                padding: 16px;
                box-shadow: 0 6px 18px rgba(0,0,0,0.04);
            }
            .controls {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 16px;
                margin: 16px 0;
            }
            .label {
                font-size: 12px;
                text-transform: uppercase;
                letter-spacing: 1.2px;
                color: var(--muted);
                margin-bottom: 6px;
                display: block;
            }
            .section-title {
                font-size: 18px;
                margin: 18px 0 10px;
                border-left: 3px solid var(--accent);
                padding-left: 10px;
            }
            .table-note {
                color: var(--muted);
                font-size: 12px;
                margin-top: 6px;
            }
            .dash-table-container .dash-spreadsheet-container {
                border: 1px solid var(--border);
                border-radius: 10px;
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

freguesia_options = ['All'] + sorted([f for f in NODE_DF['freguesia'].dropna().unique()])

app.layout = html.Div(
    className='page',
    children=[
        html.H2('Dashboard de Centralidade e Estrutura da Rede', className='title'),
        html.P('Explore a rede por freguesia, consulte top hubs e compare raio e diametro.', className='subtitle'),

        html.Div(
            className='card',
            children=[
                html.Div(
                    className='controls',
                    children=[
                        html.Div([
                            html.Label('Freguesia', className='label'),
                            dcc.Dropdown(
                                id='freguesia-dropdown',
                                options=[{'label': f, 'value': f} for f in freguesia_options],
                                value='All',
                                clearable=False
                            ),

                        ])
                    ]
                ),
                dcc.Graph(id='network-graph')
            ]
        ),

        html.H3('Resumo por freguesia', className='section-title'),
        html.Div(
            className='card',
            children=[
                dash_table.DataTable(
                    id='summary-table',
                    columns=[{'name': c, 'id': c} for c in SUMMARY_DF.columns],
                    data=SUMMARY_DF.to_dict('records'),
                    sort_action='native',
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_header={
                        'backgroundColor': '#f0e6d6',
                        'fontWeight': 'bold',
                        'border': '1px solid #e6dccb'
                    },
                    style_cell={
                        'textAlign': 'left',
                        'fontFamily': 'Georgia, Palatino, "Times New Roman", serif',
                        'fontSize': 13,
                        'padding': '8px'
                    },
                    style_data={'border': '1px solid #f1e8da'}
                ),
                html.Div('Media das centralidades e medidas topologicas por freguesia.', className='table-note')
            ]
        ),

        html.H3('Top hubs da freguesia selecionada', className='section-title'),
        html.Div(
            className='card',
            children=[
                dash_table.DataTable(
                    id='hubs-table',
                    columns=[
                        {'name': 'name', 'id': 'name'},
                        {'name': 'degree', 'id': 'degree'},
                        {'name': 'betweenness', 'id': 'betweenness'},
                        {'name': 'closeness', 'id': 'closeness'},
                    ],
                    data=[],
                    sort_action='native',
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_header={
                        'backgroundColor': '#f0e6d6',
                        'fontWeight': 'bold',
                        'border': '1px solid #e6dccb'
                    },
                    style_cell={
                        'textAlign': 'left',
                        'fontFamily': 'Georgia, Palatino, "Times New Roman", serif',
                        'fontSize': 13,
                        'padding': '8px'
                    },
                    style_data={'border': '1px solid #f1e8da'}
                ),
                html.Div('Ordene por qualquer coluna para comparar hubs rapidamente.', className='table-note')
            ]
        )
    ]
)


@app.callback(
    Output('network-graph', 'figure'),
    Output('hubs-table', 'data'),
    Input('freguesia-dropdown', 'value')

)
def update_dashboard(selected_freg):
    fig = build_network_figure(G_REDUCED, NODE_DF, selected_freg, None)

    hubs_df = NODE_DF.copy()
    if selected_freg and selected_freg != 'All':
        hubs_df = hubs_df[hubs_df['freguesia'] == selected_freg]

    hubs_df = hubs_df.sort_values('degree', ascending=False).head(10)
    hubs_df = hubs_df[['name', 'degree', 'betweenness', 'closeness']].copy()
    hubs_df['degree'] = hubs_df['degree'].round(6)
    hubs_df['betweenness'] = hubs_df['betweenness'].round(6)
    hubs_df['closeness'] = hubs_df['closeness'].round(6)

    return fig, hubs_df.to_dict('records')

server = app.server  # necessário para gunicorn no Render

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)

