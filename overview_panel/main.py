import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from PIL import Image

from edgar import *

class EdgarData:
  def __init__(self, identity, ticker, data_columns, data_rows):

    self.identity = identity
    self.ticker = ticker
    self.data_columns = data_columns
    self.data_rows = data_rows

    self.set_identity_edgar()

  def set_identity_edgar(self):
    """"
    Set the identity of the EdgarData object. Requires the identity by SEC
    """
    set_identity(self.identity)

  def fetch_company(self):
    self.company = Company(self.ticker)

  def fetch_financials(self):
    return self.company.get_financials()

  def fetch_filings(self):
    return self.company.get_filings()

  def fetch_financial_data(self):
    self.fetch_company()
    return self.fetch_financials().income_statement().to_dataframe()

  def processing_data(self):

    df = self.fetch_financial_data()

    # subset by columns
    cols = [col for col in df.columns if any(data_col in col for data_col in self.data_columns)]
    df = df[cols].set_index(cols[0])

    # subset by rows
    df = df.loc[self.data_rows,:]

    return df


class DataAnalysis:

  @staticmethod
  def extract_raw_data(df):
    return df.iloc[:,0].to_list()

  @staticmethod
  def calculate_change(df):
    return (df.iloc[:,0] - df.iloc[:,1]).to_list()

  @staticmethod
  def calculate_growth(df):
    return (df.pct_change(axis=1)*-1*100).iloc[:,1].to_list() # -1 for change sign

  @staticmethod
  def calculate_margin(df, base, relative):
    return (df.loc[relative,:]/df.loc[base,:])

  @staticmethod
  def calculate_margin_stats(df, base, relative):
    df_margin = DataAnalysis.calculate_margin(df, base, relative)
    margin = df_margin.copy()
    margin_diff = df_margin.copy().diff(-1)*100

    return (margin , margin_diff)

  @staticmethod
  def prepare_earning_data(df):
    raw_data = DataAnalysis.extract_raw_data(df)
    change = DataAnalysis.calculate_change(df)
    growth = DataAnalysis.calculate_growth(df)

    unit, base_unit = DataAnalysis.unit_formatter(raw_data)

    scaled_raw_data = [d/ base_unit if d > 200 else d for d in raw_data] # to avoid dividing eps by base_unit
    scaled_change = [c/base_unit if abs(c) > 200 else c for c in change] # to avoid dividing eps by base_unit

    return scaled_raw_data, scaled_change, growth, unit

  @staticmethod
  def unit_formatter(data:list):
    if data[0] > 1e9:
      unit = 'Bn'
      base_unit = 1e9
    elif 1e6 < data[0] & data[0] > 1e9:
      unit = 'M'
      base_unit = 1e6
    else:
      unit = ''
      base_unit = 1

    return unit, base_unit


class PlotlyUtils:
  @staticmethod
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

  @staticmethod
  def add_shape_object(fig, x0=0, y0=0, type='line', color='#676767', width=2):
    fig.add_shape(
      type=type,
      x0=x0, x1=x0+0.36,
      y0=y0, y1=y0,
      xref='paper',
      yref='paper',
      line=dict(color=color, width=width)
  )

  @staticmethod
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

  @staticmethod
  def get_color(value):
      return '#1DB954' if value >= 0 else '#BC2D24'

  @staticmethod
  def prepare_plotting_data(df):
    raw_values, change, growth, unit = DataAnalysis.prepare_earning_data(df)
    revenue, operating_income, net_income, eps = raw_values
    revenue_change, operating_income_change, net_income_change, eps_change = change
    revenue_growth, operating_income_growth, net_income_growth, eps_growth = growth

    operating_margin, operating_margin_diff = DataAnalysis.calculate_margin_stats(df, 'Net sales', 'Operating Income')
    income_margin, income_margin_diff = DataAnalysis.calculate_margin_stats(df, 'Net sales', 'Net Income')

  @staticmethod
  def create_figure(rows, cols, df, year, title):

    # figure setting
    n_grids = rows*cols+1

    fig = make_subplots(
        rows=rows,
        cols=cols,
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
        column_widths=[0.10, 0.2, 0.25, 0.2, 0.25]
    )

    # data preparation
    raw_values, change, growth, unit = DataAnalysis.prepare_earning_data(df)
    revenue, operating_income, net_income, eps = raw_values
    revenue_change, operating_income_change, net_income_change, eps_change = change
    revenue_growth, operating_income_growth, net_income_growth, eps_growth = growth

    operating_margin, operating_margin_diff = DataAnalysis.calculate_margin_stats(df, 'Net sales', 'Operating Income')
    income_margin, income_margin_diff = DataAnalysis.calculate_margin_stats(df, 'Net sales', 'Net Income')

    PlotlyUtils.add_image(fig, source='financial-statement.png', x=0.48, y=0.9)
    PlotlyUtils.add_image(fig, source='pie-chart.png', x=0.88, y=0.9)

    PlotlyUtils.add_header(fig, header_text=f'{year} ($)', x=0.4, y=0.88, weight='bold')
    PlotlyUtils.add_header(fig, header_text=f'{year} (% of Revenue)', x=0.83, y=0.88, weight='bold')

    PlotlyUtils.add_shape_object(fig, x0=0.2, y0=0.82)
    PlotlyUtils.add_shape_object(fig, x0=0.63, y0=0.82)

    PlotlyUtils.add_header(fig, header_text=f'Revenue', x=0.0, y=0.75, weight='bold')
    PlotlyUtils.add_header(fig, header_text=f'Operating<br>Income', x=0.0, y=0.50, weight='bold')
    PlotlyUtils.add_header(fig, header_text=f'Net<br>Income', x=0.0, y=0.22, weight='bold')
    PlotlyUtils.add_header(fig, header_text=f'Earnings<br>Per Share', x=0.0, y=0.0, weight='bold')

    fig.add_trace(go.Bar(y=df.loc['Net sales',:].iloc[::-1], marker_color='lightgrey'), row=2, col=2)
    fig.add_trace(go.Bar(y=df.loc['Operating Income',:].iloc[::-1], marker_color='lightgrey'), row=3, col=2)
    fig.add_trace(go.Bar(y=df.loc['Net Income',:].iloc[::-1], marker_color='lightgrey'), row=4, col=2)
    fig.add_trace(go.Bar(y=df.loc['Diluted (in dollars per share)',:].iloc[::-1], marker_color='lightgrey'), row=5, col=2)

    third_col_xposition = 0.46
    value_font_size = 26
    value_font_color = '#004080'
    PlotlyUtils.add_header(fig, header_text=f'${revenue:.1f}{unit}', x=third_col_xposition, y=0.78, font_size=value_font_size, font_color=value_font_color, weight='bold')
    PlotlyUtils.add_header(fig, header_text=f'${operating_income:.1f}{unit}', x=third_col_xposition, y=0.52, font_size=value_font_size, font_color=value_font_color, weight='bold')
    PlotlyUtils.add_header(fig, header_text=f'${net_income:.1f}{unit}', x=third_col_xposition, y=0.26, font_size=value_font_size, font_color=value_font_color, weight='bold')
    PlotlyUtils.add_header(fig, header_text=f'${eps:.2f}', x=third_col_xposition, y=0.03, font_size=value_font_size, font_color=value_font_color, weight='bold')

    relative_value_font_size = 16
    rev_font_color = PlotlyUtils.get_color(revenue_change)
    operating_income_font_color = PlotlyUtils.get_color(operating_income_change)
    net_income_font_color = PlotlyUtils.get_color(net_income_change)
    eps_font_color = PlotlyUtils.get_color(eps)

    PlotlyUtils.add_header(fig, header_text=f'${revenue_change:.2f}{unit}, {revenue_growth:.1f}%', x=third_col_xposition, y=0.73, font_size=relative_value_font_size, font_color=rev_font_color, weight='bold')
    PlotlyUtils.add_header(fig, header_text=f'${operating_income_change:.2f}{unit}, {operating_income_growth:.1f}%', x=third_col_xposition, y=0.48, font_size=relative_value_font_size, font_color=operating_income_font_color, weight='bold')
    PlotlyUtils.add_header(fig, header_text=f'${net_income_change:.2f}{unit}, {net_income_growth:.1f}%', x=third_col_xposition, y=0.23, font_size=relative_value_font_size, font_color=net_income_font_color, weight='bold')
    PlotlyUtils.add_header(fig, header_text=f'${eps_change:.2f}, {eps_growth:.1f}%', x=third_col_xposition, y=0.0, font_size=relative_value_font_size, font_color=eps_font_color, weight='bold')

    fig.add_trace(go.Scatter(y=operating_margin.iloc[::-1], marker_color='lightgrey'), row=3, col=4)
    fig.add_trace(go.Scatter(y=income_margin.iloc[::-1], marker_color='lightgrey'), row=4, col=4)

    last_col_xposition = 0.9
    operating_income_margin_font_color = PlotlyUtils.get_color(operating_margin_diff.iloc[0])
    net_income_margin_font_color = PlotlyUtils.get_color(income_margin_diff.iloc[0])

    PlotlyUtils.add_header(fig, header_text=f'{operating_margin.iloc[0]:.1f}%', x=last_col_xposition, y=0.52, font_size=value_font_size, font_color=value_font_color, weight='bold')
    PlotlyUtils.add_header(fig, header_text=f'{operating_margin_diff.iloc[0]:.1f}pp', x=last_col_xposition, y=0.48, font_size=relative_value_font_size, font_color=operating_income_margin_font_color, weight='bold')
    PlotlyUtils.add_header(fig, header_text=f'{income_margin.iloc[0]:.1f}%', x=last_col_xposition, y=0.26, font_size=value_font_size, font_color=value_font_color, weight='bold')
    PlotlyUtils.add_header(fig, header_text=f'{income_margin_diff.iloc[0]:.1f}pp', x=last_col_xposition, y=0.23, font_size=relative_value_font_size, font_color=net_income_margin_font_color, weight='bold')


    for i in range(1, n_grids):
        fig.update_layout({
            f'xaxis{i}': dict(showgrid=False, zeroline=False, showticklabels=False, ticks='', showline=False),
            f'yaxis{i}': dict(showgrid=False, zeroline=False, showticklabels=False, ticks='', showline=False)
        })

    fig.update_layout(
        title_text=title,
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


data_columns = ['label','2024', '2023', '2022']
year = '2024'
data_rows = ['Net sales','Operating Income','Net Income', 'Diluted (in dollars per share)']
edgar = EdgarData(identity="your.name@example.com", ticker="AAPL", data_columns=data_columns, data_rows=data_rows)
df_income = edgar.processing_data()

PlotlyUtils.create_figure(rows=5, cols=5, df=df_income, year=2024, title="Apple's 2024 Earnings Overview")
