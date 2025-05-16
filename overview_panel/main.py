from plotly.subplots import make_subplots
import plotly.graph_objects as go
from PIL import Image

def add_image(fig, source=None, x=0, y=0, sizex=0.15, sizey=0.15):
  """
  The source attribute of a go.layout.Image can be the URL of an image, or
  a PIL Image object (from PIL import Image; img = Image.open('filename.png')).
  """
  if source:
    img = Image.open(source)

  fig.add_layout_image(
    dict(
        source=img,
        xref="paper",
        yref="paper",
        x=x,
        y=y,
        sizex=sizex,
        sizey=sizey,
        xanchor="right",
        yanchor="bottom"
    )
)

def add_shape_object(fig, x0=0, y0=0, type='line', color='#676767', width=2):
  fig.add_shape(
    type=type,
    x0=x0, x1=x0+0.36,
    y0=y0, y1=y0,
    xref='paper',
    yref='paper',
    line=dict(color=color, width=width)
)

def add_header(fig, header_text, x=0, y=0, font_size=20, font_color='#7C7C7C', weight=None):
  if weight=='bold':
    header_text = f'<b>{header_text}</b>'
  elif weight=='italic':
    header_text = f'<i>{header_text}</i>'
  fig.add_annotation(text=header_text,
                    showarrow=False,
                    font=dict(size=font_size, color=font_color),
                    xref='paper',
                    x=x,
                    yref='paper',
                    y=y,
                    xanchor='center')


def get_color(value):
    return '#1DB954' if value >= 0 else '#BC2D24'


revenue = 137
rev_growth = 26
rev_change = -23
oi = 26
oi_growth = 26
oi_change = 23
ni = 31
ni_growth = 26
ni_change = -23
eps = 43.70
eps_growth = 26
eps_change = 23

unit = 'Bn'
year = 2024

rows = 5
cols = 5
n_grids = rows*cols+1

fig = make_subplots(
    rows=rows,
    cols=cols,
    horizontal_spacing=0.12,
    vertical_spacing=0.15,
    column_widths=[0.10, 0.2, 0.25, 0.2, 0.25]
)

add_image(fig, source='./rsc/financial-statement.png', x=0.48, y=0.9)
add_image(fig, source='./rsc/pie-chart.png', x=0.88, y=0.9)

add_header(fig, header_text=f'{year} ($)', x=0.4, y=0.88, weight='bold')
add_header(fig, header_text=f'{year} (% of Revenue)', x=0.83, y=0.88, weight='bold')

add_shape_object(fig, x0=0.2, y0=0.82)
add_shape_object(fig, x0=0.63, y0=0.82)

add_header(fig, header_text=f'Revenue', x=0.0, y=0.75, weight='bold')
add_header(fig, header_text=f'Operating<br>Income', x=0.0, y=0.50, weight='bold')
add_header(fig, header_text=f'Net<br>Income', x=0.0, y=0.22, weight='bold')
add_header(fig, header_text=f'Earnings<br>Per Share', x=0.0, y=0.0, weight='bold')

fig.add_trace(go.Bar(y=[2,3,1], marker_color='lightgrey'), row=2, col=2)
fig.add_trace(go.Bar(y=[1,3,1], marker_color='lightgrey'), row=3, col=2)
fig.add_trace(go.Bar(y=[1,2,3], marker_color='lightgrey'), row=4, col=2)
fig.add_trace(go.Bar(y=[1,2,3], marker_color='lightgrey'), row=5, col=2)

third_col_xposition = 0.46
value_font_size = 26
value_font_color = '#004080'
add_header(fig, header_text=f'${revenue:.1f}{unit}', x=third_col_xposition, y=0.78, font_size=value_font_size, font_color=value_font_color, weight='bold')
add_header(fig, header_text=f'${oi:.1f}{unit}', x=third_col_xposition, y=0.52, font_size=value_font_size, font_color=value_font_color, weight='bold')
add_header(fig, header_text=f'${ni:.1f}{unit}', x=third_col_xposition, y=0.26, font_size=value_font_size, font_color=value_font_color, weight='bold')
add_header(fig, header_text=f'${eps:.1f}{unit}', x=third_col_xposition, y=0.03, font_size=value_font_size, font_color=value_font_color, weight='bold')


relative_value_font_size = 16
rev_font_color = get_color(rev_change)
ni_font_color = get_color(ni_change)
oi_font_color = get_color(oi_change)
eps_font_color = get_color(eps)

add_header(fig, header_text=f'${rev_growth:.1f}, {rev_change:.1f}%', x=third_col_xposition, y=0.73, font_size=relative_value_font_size, font_color=rev_font_color, weight='bold')
add_header(fig, header_text=f'${oi_growth:.1f}, {oi_change:.1f}%', x=third_col_xposition, y=0.48, font_size=relative_value_font_size, font_color=oi_font_color, weight='bold')
add_header(fig, header_text=f'${ni_growth:.1f}, {ni_change:.1f}%', x=third_col_xposition, y=0.23, font_size=relative_value_font_size, font_color=ni_font_color, weight='bold')
add_header(fig, header_text=f'${eps_growth:.1f}, {eps_change:.1f}%', x=third_col_xposition, y=0.0, font_size=relative_value_font_size, font_color=eps_font_color, weight='bold')

fig.add_trace(go.Scatter(y=[2,3,1], marker_color='lightgrey'), row=2, col=4)
fig.add_trace(go.Scatter(y=[1,3,1], marker_color='lightgrey'), row=3, col=4)
fig.add_trace(go.Scatter(y=[1,2,3], marker_color='lightgrey'), row=4, col=4)
fig.add_trace(go.Scatter(y=[1,2,3], marker_color='lightgrey'), row=5, col=4)

last_col_xposition = 0.9

add_header(fig, header_text=f'${oi_growth:.1f}%', x=last_col_xposition, y=0.52, font_size=value_font_size, font_color=value_font_color, weight='bold')
add_header(fig, header_text=f'${oi_change:.1f}pp', x=last_col_xposition, y=0.48, font_size=relative_value_font_size, font_color=oi_font_color, weight='bold')
add_header(fig, header_text=f'${ni_growth:.1f}%', x=last_col_xposition, y=0.26, font_size=value_font_size, font_color=value_font_color, weight='bold')
add_header(fig, header_text=f'${ni_change:.1f}pp', x=last_col_xposition, y=0.23, font_size=relative_value_font_size, font_color=oi_font_color, weight='bold')


for i in range(1, n_grids):
    fig.update_layout({
        f'xaxis{i}': dict(showgrid=False, zeroline=False, showticklabels=False, ticks='', showline=False),
        f'yaxis{i}': dict(showgrid=False, zeroline=False, showticklabels=False, ticks='', showline=False)
    })

fig.update_layout(
    title_text="Apple's 2024 Earnings Overview",
    title_x=0.5,
    font=dict(size=28, color='#535353'),
    width=800,
    height=750,
    paper_bgcolor='white',
    plot_bgcolor='white',
    showlegend=False,
)

# Source
fig.add_annotation(text="Source: SEC (EDGAR) Filings",
                   showarrow=False,
                   font=dict(size=16, color="grey"),
                   xref='paper',
                   x=1.0,
                   yref='paper',
                   y=-0.1)


fig.show()
