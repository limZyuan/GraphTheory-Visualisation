import pandas as pd
import numpy as np
import networkx as nx
from bokeh.io import output_file, show
from bokeh.models import (Circle, HoverTool, WheelZoomTool, PanTool, TapTool,
                          MultiLine, Plot, Range1d, ResetTool, ColorBar, NumeralTickFormatter)
from bokeh.models.graphs import (
    from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes)
from bokeh.models import ColumnDataSource, LabelSet
from bokeh.transform import linear_cmap

CUSTOM_PALETTE = ['#92c5de', '#67fd8c', '#f8d56c', '#ff9f40', '#ca0020']

# Prepare Data
df = pd.read_table("Air_2021_CountryNames.csv", delimiter=',', index_col=0)
df_countries = pd.read_table("countries.csv", delimiter=',', index_col=0)

# Filter by HS6
HS6_CODE = '851712'
df_HS6 = df[df['hs6'] == HS6_CODE]

# filter for tradelanes
ASIA = ['Minor Pacific Islands', 'Australia', 'Brunei', 'China', 'Hong Kong', 'Indonesia', 'Japan', 'Cambodia', 'North Korea', 'South Korea', 'Laos', 'Myanmar', 'Mongolia',
        'Macao', 'Malaysia', 'French Polynesia & New Caledonia', 'New Zealand', 'Papua New Guinea', 'Philippines', 'Singapore', 'Thailand',  'East Timor', 'Taiwan', 'Vietnam', 'India']
EUROPE = ['Albania', 'Austria', 'Bosnia Herzegovina', 'Finland', 'Belgium + Luxembourg', 'Bulgaria', 'Belarus', 'Switzerland', 'Serbia, Montenegro and Kosovo', 'Cyprus', 'Czech Republic', 'Germany', 'Denmark', 'Estonia', 'Spain', 'Faroe Islands', 'France', 'United Kingdom',
          'Guernsey', 'Greenland', 'Greece', 'Croatia', 'Hungary', 'Irish Republic', 'Iceland', 'Italy', 'Lithuania', 'Latvia', 'Moldova', 'Macedonia', 'Malta', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania', 'Russia', 'Sweden', 'Slovenia', 'Slovakia', 'Ukraine']
USA = ['USA']

#df_ori = df_HS6[df_HS6['origin'].isin(ASIA)]
#df_final = df_ori[df_ori['destination'].isin(ASIA)]

#df_asiaeu = df_HS6[df_HS6['origin'].isin(ASIA) & df_HS6['destination'].isin(EUROPE)]
#df_euasia = df_HS6[df_HS6['origin'].isin(EUROPE) & df_HS6['destination'].isin(ASIA)]
#df_final = pd.concat([df_asiaeu,df_euasia])

df_asiaus = df_HS6[df_HS6['origin'].isin(
    ASIA) & df_HS6['destination'].isin(USA)]
df_usasia = df_HS6[df_HS6['origin'].isin(
    USA) & df_HS6['destination'].isin(ASIA)]
df_final = pd.concat([df_asiaus, df_usasia])

M = nx.from_pandas_edgelist(df_final, 'origin', 'destination', edge_attr=[
                            'origin', 'destination', 'air_value_usd', 'air_weight_kg'], create_using=nx.MultiGraph())

# create weighted graph from M to aggregate the parallel edges into 1 edge between each pair of nodes
G = nx.Graph()
for u, v, data in M.edges(data=True):
    w = data['air_value_usd'] if 'air_value_usd' in data else 1.0

    # remove routes that have no air value
    if w == 0 or pd.isna(w):
        continue

    if G.has_edge(u, v):
        G[u][v]['air_value_usd'] += w
    else:
        G.add_edge(u, v, air_value_usd=w,
                   air_weight_kg=data['air_weight_kg'], origin=data['origin'], destination=data['destination'])

# Node attributes
node_attrs = {}
node_attrs_coord = {}
SPACER = 0
for i in G.nodes(data=True):
    if i[0] == 'Singapore':
        node_attrs[i[0]] = 'firebrick'
    else:
        node_attrs[i[0]] = 'lightblue'

    NODE_LOG = df_countries[df_countries['name'] == i[0]
                            ]['long_140-0.3'][0] if not df_countries[df_countries['name'] == i[0]]['long_140-0.3'].empty else 0
    NODE_LAT = df_countries[df_countries['name'] == i[0]
                            ]['lat_55-0.2'][0] if not df_countries[df_countries['name'] == i[0]]['lat_55-0.2'].empty else -1.7

    if NODE_LOG == 0:
        NODE_LOG += SPACER
        SPACER += 0.06

    node_attrs_coord[i[0]] = (NODE_LOG, NODE_LAT)


nx.set_node_attributes(G, node_attrs_coord, "node_coordinates")
nx.set_node_attributes(G, node_attrs, "node_color")

# for color mapper percentile cal below
allTheSummedEdgeAirValueDataForColorBar = []
for start_node, end_node, data in G.edges(data=True):

    w = data['air_value_usd'] if 'air_value_usd' in data else 1.0
    allTheSummedEdgeAirValueDataForColorBar.append(w)

# Edge attributes
edge_attrs = {}
for start_node, end_node, data in G.edges(data=True):
    edge_opacity = 0.3
    w = data['air_value_usd'] if 'air_value_usd' in data else 1.0

    if w < np.nanpercentile(allTheSummedEdgeAirValueDataForColorBar, 75):
        edge_attrs[(start_node, end_node)] = 0
    else:
        edge_attrs[(start_node, end_node)] = edge_opacity

nx.set_edge_attributes(G, edge_attrs, "edge_opacity")


# Graph Visualisation Tool

'''
# Colorbar (Use the field name of the column source)
bounds = [np.percentile(df_HS6['air_value_usd'], 20),np.percentile(df_HS6['air_value_usd'], 40),np.percentile(df_HS6['air_value_usd'], 60),np.percentile(df_HS6['air_value_usd'], 80),np.percentile(df_HS6['air_value_usd'], 100)]
 
low = 0
high =  np.percentile(df_HS6['air_value_usd'], 100).astype(int)
bound_colors = []
j = 0
for i in range(low, high, 100):
    if i >= bounds[j+1]:
        j += 1
    bound_colors.append(CUSTOM_PALETTE[j])
'''

mapper = linear_cmap(field_name='air_value_usd', palette=CUSTOM_PALETTE, low=np.nanpercentile(
    allTheSummedEdgeAirValueDataForColorBar, 75), high=np.nanpercentile(allTheSummedEdgeAirValueDataForColorBar, 99))
# mapper = linear_cmap(field_name='air_value_usd', palette= CUSTOM_PALETTE ,low= min(allTheSummedEdgeAirValueDataForColorBar) ,high = max(allTheSummedEdgeAirValueDataForColorBar))

# Show with Bokeh
plot = Plot(width=1250, height=550,
            x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))
plot.title.text = f'Flow of {HS6_CODE} (Air Value) GEO'

node_hover_tool = HoverTool(tooltips=[("Route", "Between @origin and @destination"), (
    "Air Value", "@air_value_usd{1,11} USD"), ("Air Weight", "@air_weight_kg{1,11} Kg")], show_arrow=False)
node_hover_tool.line_policy = 'interp'
plot.add_tools(node_hover_tool, ResetTool(),
               WheelZoomTool(), PanTool(), TapTool())
plot.toolbar.active_scroll = plot.select_one(WheelZoomTool)
graph_renderer = from_networkx(G, nx.get_node_attributes(
    G, 'node_coordinates'), scale=1, center=(0, 0))

graph_renderer.node_renderer.glyph = Circle(size=13, fill_color="node_color")
graph_renderer.node_renderer.hover_glyph = Circle(
    size=15, fill_color="#4575b4")
graph_renderer.node_renderer.selection_glyph = Circle(
    size=15, fill_color="#4575b4")

graph_renderer.edge_renderer.glyph = MultiLine(
    line_color=mapper, line_alpha="edge_opacity", line_width=1)
graph_renderer.edge_renderer.hover_glyph = MultiLine(
    line_color=mapper, line_alpha=1, line_width=3.5)
graph_renderer.edge_renderer.selection_glyph = MultiLine(
    line_color=mapper, line_alpha=1, line_width=2)

graph_renderer.inspection_policy = EdgesAndLinkedNodes()
graph_renderer.selection_policy = NodesAndLinkedEdges()
plot.renderers.append(graph_renderer)

# Node labels
x, y = zip(*graph_renderer.layout_provider.graph_layout.values())
node_labels = list(graph_renderer.layout_provider.graph_layout.keys())
source = ColumnDataSource({'x': x, 'y': y,
                           'airport':  [node_labels[i] for i in range(len(x))]})
labels = LabelSet(x='x', y='y',  x_offset=5, y_offset=5, text='airport', source=source,
                  render_mode='canvas', background_fill_alpha=0, level='glyph', text_font_size="5pt")

plot.renderers.append(labels)

# Color bar
color_bar = ColorBar(color_mapper=mapper['transform'], width=5, label_standoff=10,
                     formatter=NumeralTickFormatter(format='0 a'), location=(0, 0))
plot.add_layout(color_bar, 'right')

output_file(f'{HS6_CODE}_GEO.html', mode='inline',
            title=f'Flow of {HS6_CODE} (Air Value) GEO')
show(plot)
