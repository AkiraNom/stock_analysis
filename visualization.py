import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional
import io
import os
import json
from datetime import datetime
import matplotlib.font_manager as fm
import matplotlib as mpl

from custom_utils import DataProcessing, logger

class DataVisualization:
    def __init__(self, data_processor: DataProcessing, title="Financial Dashboard", figsize=(18, 14)):

        self.data_processor = data_processor
        self.income_statement = self.data_processor.income_statement
        self.balance_sheet = self.data_processor.balance_sheet
        self.cash_flow = self.data_processor.cash_flow
        self.dates = self.data_processor.dates
        self.title = title
        self.figsize = figsize

        # for continuous data
        self.color_sequence = [
            "#E9EFF9",
            "#D0DCF1",
            "#B6C9E9",
            "#9DB6E1",
            "#83A3D9",
            "#6990D1",
            "#4472C4",
            "#3B65AF",
            "#335899",
            "#2A4B84",
            "#203E6E"
            ]

        # for discrete data
        self.color_sequence_short = [
          '#2980B9',  # Blue
          '#E67E22',  # Orange
          '#27AE60',  # Green
          '#C0392B',  # 🔴 Muted red (NEW)
          '#95A5A6',  # Light gray
          '#2C3E50'   # Dark gray
        ]

    def _init_figure(self):
        fig = plt.figure(figsize=self.figsize, constrained_layout=True)
        fig.patch.set_facecolor('white')
        fig.suptitle(self.title, fontsize=20, fontweight='bold', y=1.04)
        fig.get_layout_engine().set(hspace=0.15,wspace=0.05)
        return fig

    def _init_gridspec(self, fig):
        gs = gridspec.GridSpec(3, 5, figure=fig)
        return gs

    def generate_plot(self, ax, data, data_map, plot_columns, data_transform=True, type='bar', title=None, y_unit='Milions', y_lim=True, show_legend=True, show_values=False):

      if y_unit in ('Millions','Billions', 'Percent'):
        ylabel = None
        y_formatter = y_unit.lower()
      elif y_unit:
        ylabel = y_unit
        y_formatter = 'normal'
      else:
        ylabel = None
        y_formatter = 'normal'

      if data_transform:
        plot_data = DataProcessing.transpose_metrics_for_plotting(data, data_map, plot_columns)
      else:
        plot_columns = list(data_map)
        plot_data = {}
        for col, feature_dict in data.items():
            vals = []
            for feature in data_map.values():
                _vals = data[col][feature]
                vals.append(_vals)
            plot_data[str(col)] = vals

      if type == 'multi_bar':
        self._create_multi_bar_chart(ax, plot_data, plot_columns, title=title, ylabel=ylabel, y_formatter=y_formatter, show_legend=show_legend, show_values=show_values)
      elif type == 'line':
        self._create_line_chart(ax, plot_data, plot_columns, title=title, ylabel=ylabel, y_formatter=y_formatter, show_legend=show_legend, y_lim=y_lim, show_values=show_values)
      elif type == 'stacked':
        self._create_stacked_area_chart(ax, plot_data, plot_columns, title=title, ylabel=ylabel, y_formatter=y_formatter, normalize=True)
      else:
        self._create_bar_chart(ax, plot_data[list(data_map)[0]], plot_columns, title=title, ylabel=ylabel, y_formatter=y_formatter, show_values=show_values)
    def _format_number(self, num):
        """
        Convert large numbers to human-readable format (B for billions, M for millions)
        """
        if num >= 1e9:
            return f"{num/1e9:.1f}B"
        elif num >= 1e6:
            return f"{num/1e6:.1f}M"
        else:
            return f"{num:.1f}"

    def _format_axis(self, ax, title=None, xlabel=None, ylabel=None, y_formatter='normal'):
        """Format axis with title and labels"""
        if title:
            ax.set_title(title, fontsize=10, fontweight='bold')
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=8)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=8)

        # Format y-axis
        if y_formatter == 'percent':
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
        elif y_formatter == 'millions':
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y/1e6:.1f}M'))
        elif y_formatter == 'billions':
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y/1e9:.1f}B'))

        # Clean up the grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Set tick sizes
        ax.tick_params(axis='both', which='major', labelsize=8)

    def _format_legend(self, ax):

      return ax.legend(fontsize=7)

    def _add_text_box(self, ax, text, fontsize=12, loc='upper right', frameon=True, color='black'):
        """Add a text box to the axis"""
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.95, 0.95, text, transform=ax.transAxes, fontsize=fontsize,
                verticalalignment='top', horizontalalignment='right',
                bbox=props if frameon else None, color=color)

    def _create_multi_bar_chart(self, ax, data, plot_columns, title=None, ylabel=None, y_formatter='normal',
                              colors=None, width=0.8, show_values=False, multiplier=1, show_legend=False):
        """Create a multi-series bar chart"""
        if colors is None:
            colors = self.color_sequence_short[:len(plot_columns)]

        categories = list(data.keys())
        n_cols = len(plot_columns)
        n_cats = len(categories)

        # Position of groups
        ind = np.arange(n_cats)
        # Width of each individual bar
        bar_width = width / n_cols

        # Create bars for each year
        for i, col in enumerate(plot_columns):
            col_data = [data[cat][i] for cat in categories]
            x_pos = ind + (i - n_cols/2 + 0.5) * bar_width
            bars = ax.bar(x_pos, col_data, bar_width, label=col, color=colors[i])

            if show_values:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2.,
                            height + 0.01 * max([max(data[cat]) for cat in categories]),
                            self._human_readable_number(height),
                            ha='center', va='bottom', fontsize=6, rotation=90)

        # Set x-axis ticks and labels
        ax.set_xticks(ind)
        ax.set_xticklabels(categories, fontsize=8)

        if show_legend:
          self._format_legend(ax)

        self._format_axis(ax, title=title, ylabel=ylabel, y_formatter=y_formatter)

        return ax

    def _create_bar_chart(self, ax, data, x_labels, title=None, ylabel=None, color=None,
                        y_formatter='normal', show_values=False):
        """Create a single series bar chart"""
        if color is None:
            color = self.color_sequence_short[0]

        bars = ax.bar(range(len(x_labels)), data, color=color)

        if show_values:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(data),
                        self._format_number(height), ha='center', va='bottom',
                        fontsize=6, rotation=90)

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)

        self._format_axis(ax, title=title, ylabel=ylabel, y_formatter=y_formatter)

        return ax

    def _create_stacked_area_chart(self, ax, data, x_labels, title=None, ylabel=None, y_formatter='normal',
                                 colors=None, alpha=0.7, normalize=False):
        """Create a stacked area chart"""
        if colors is None:
            colors = self.color_sequence_short[:len(data)]

        # Create numpy arrays for stacking
        y = np.row_stack([values for label, values in data.items()])

        # Normalize to percentages if requested
        if normalize:
            sums = y.sum(axis=0)
            y_pct = np.zeros_like(y, dtype=float)
            for i in range(y.shape[1]):
                if sums[i] > 0:  # Avoid division by zero
                    y_pct[:, i] = y[:, i] / sums[i] * 100
            y = y_pct

        ax.stackplot(range(len(x_labels)), y, labels=[label for label, _ in data.items()],
                     colors=colors, alpha=alpha)

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)

        self._format_legend(ax)

        self._format_axis(ax, title=title, ylabel=ylabel, y_formatter='percent' if normalize else y_formatter)

        return ax

    def _create_line_chart(self, ax, data, x_labels, title=None, ylabel=None, y_formatter='normal', y_lim=True,
                         colors=None, marker='o', markersize=4, show_values=False, show_legend=True):

        if colors is None:
            colors = self.color_sequence_short[:len(data)]

        for i, (label, values) in enumerate(data.items()):
            line = ax.plot(range(len(x_labels)), values, marker=marker, markersize=markersize,
                   label=label, color=colors[i])

            if show_values:
                for x, y in enumerate(values):
                    ax.text(x, y + 0.02 * max(values), self._format_number(y),
                            ha='center', va='bottom', fontsize=6)

        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7)

        if y_lim:
          ax.set_ylim(0)

        if show_legend:
          self._format_legend(ax)

        self._format_axis(ax, title=title, ylabel=ylabel, y_formatter=y_formatter)

        return ax


    def create_dashboard(self):
        """Create the full financial dashboard."""
        fig = self._init_figure()
        gs = self._init_gridspec(fig)

        # 1. Income Statement Overview
        ax = fig.add_subplot(gs[0, :3])
        data_map = {
            'Revenue': 'revenue',
            'Gross Profit': 'gross_profit',
            'Operating Income': 'operating_income',
            'EBIT': 'ebit',
            'EBITDA':'ebitda',
            'Net Income': 'net_income',
        }
        self.generate_plot(ax, self.data_processor.metrics, data_map, self.dates, data_transform=True,
                        type='line',title='Sales', y_unit='Billions')

        # 2. REVENUE GROWTH
        ax = fig.add_subplot(gs[0, 3])
        data_map = {'Revenue Growth %': 'revenue_growth'}
        self.generate_plot(ax, self.data_processor.metrics, data_map, self.dates,
                        type='bar',title='Revenue Growth %', y_unit='Percent')

        # 3. MARGIN TRENDS
        ax = fig.add_subplot(gs[0, 4])
        data_map ={
          'Gross Profit Margin %' : 'gross_profit_margin',
          'Operating Margin %' : 'operating_margin',
          'EBIT Margin %': 'ebit_margin',
          'EBITDA Margin %': 'ebitda_margin',
          'Net Profit Margin %': 'net_profit_margin',
        }
        self.generate_plot(ax, self.data_processor.metrics, data_map, self.dates,
                        type='line',title='Sales Trend', y_unit='Percent')

        # 4. VALUATION RATIOS - EPS
        ax = fig.add_subplot(gs[1, 0])
        data_map = {
            'EPS' : 'diluted_eps',
        }
        self.generate_plot(ax, self.data_processor.metrics, data_map, self.dates,
                        type='bar',title='EPS', y_unit='EPS ($)')

        # 5. VALUATION RATIOS - BPS
        ax = fig.add_subplot(gs[1, 1])
        data_map = {
            'Book Value per Share' : 'bps',
        }
        self.generate_plot(ax, self.data_processor.metrics, data_map, self.dates,
                        type='bar',title='BPS', y_unit='BPS ($)')

        # 6. VALUATION RATIOS - dividend per share
        ax = fig.add_subplot(gs[1, 2])
        data_map = {
            'Dividend per Share' : 'dividend',
        }
        self.generate_plot(ax, self.data_processor.metrics, data_map, self.dates,
                        type='bar',title='BPS', y_unit='BPS ($)')

        # 6. SHARE
        ax = fig.add_subplot(gs[1, 3])
        data_map = {
            'Shares': 'shares_outstanding'
        }
        self.generate_plot(ax, self.data_processor.metrics, data_map, self.dates,
                        type='line', title='Shares', y_unit='Billions', show_legend=False)

        # 7. PROFITABILITY METRICS
        ax = fig.add_subplot(gs[1, 4])
        data_map = {
            'ROA': 'roa',
            'ROE': 'roe',
            'ROIC': 'roic'
        }
        self.generate_plot(ax, self.data_processor.metrics, data_map, self.dates,
                        type='line',title='Profitability', y_unit='Percent')

        # 8. RELATIVE COMPOSITION
        ax = fig.add_subplot(gs[2, 0])
        data_map = {
          'Short Debt': 'short_term_debt',
          'Long Debt': 'long_term_debt',
          'Equity': 'equity'
        }
        self.generate_plot(ax, self.data_processor.metrics, data_map, self.dates,
                        type='stacked',title='Assets Composition', y_unit='Percent')

        # 9. SOLVENCY METRICS
        ax = fig.add_subplot(gs[2, 1])
        data_map = {
            'D/E ratio' : 'de_ratio',
            'D/A ratio': 'da_ratio',
            'Equity ratio': 'equity_ratio'
        }
        self.generate_plot(ax, self.data_processor.metrics, data_map, self.dates,
                        type='line',title='Solvency', y_unit='Percent')

        # 10. EFFICIENCY METRICS
        ax = fig.add_subplot(gs[2, 2])
        data_map = {
            'FCF/Net Income ratio': 'fcf_ni_ratio',
            'Asset turnover ratio': 'asset_turnover_ratio'
        }
        self.generate_plot(ax, self.data_processor.metrics, data_map, self.dates,
                        type='line',title='Efficiency', y_unit='Percent')

        # 11. CASH FLOW METRICS
        ax = fig.add_subplot(gs[2, 3])
        data_map = {
          'CFO': 'cfo',
          'CFI': 'cfi',
          'CFF': 'cff'
        }
        self.generate_plot(ax, self.data_processor.metrics, data_map, self.dates,
                        type='line',title='Cash Flow', y_unit='Billions', y_lim=False)

        # 12. FCF TREND
        ax = fig.add_subplot(gs[2, 4])
        data_map = {'FCF': 'fcf' }
        self.generate_plot(ax, self.data_processor.metrics, data_map, self.dates,
                        type='line',title='Free Cash Flow', y_unit='Billions', show_legend=False)

        return fig

    @logger(separator='*')
    def show_dashboard(self, save=False, filename='financial_dashboard.png', dpi=150):
        """Display the dashboard"""
        fig = self.create_dashboard()

        if save:
          try:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight', pad_inches=0.5)
            message= f"Dashboard saved to {filename}"

          except Exception as e:
            message = f"Error saving dashboard: {e}"

        plt.show()
        plt.close(fig)
        return message
