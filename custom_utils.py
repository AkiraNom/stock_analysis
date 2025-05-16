import pandas as pd
import numpy as np
from datetime import datetime

def logger(separator: str='-'):
  def decorator(func):
    def wrapper(*args, **kwargs):
      
      result = func(*args, **kwargs)
      print(datetime.now(),':',result)
      print(separator*20)
      return result
    return wrapper
  return decorator

class DataProcessing:
    """
    A class to analyze financial data from CSV files and prepare it for dashboard visualization.
    """

    def __init__(self, csv_file_path: str):
        """
        Initialize the analyzer with a CSV file path.

        Args:
            csv_file_path: Path to the CSV file containing financial data
        """
        self.csv_file_path = csv_file_path
        self.data = None
        self.income_statement = None
        self.balance_sheet = None
        self.cash_flow = None
        self.metrics = {}
        self.dates = None

    @logger(separator='*')
    def load_data(self) -> None:
        """
        Load financial data from CSV file and separate into different statements.
        """
        try:
            # Read the CSV file
            self.data = pd.read_csv(self.csv_file_path)

            # Extract years (columns except the first one)
            self.dates = [col for col in self.data.columns[1:] if col != 'TTM' and col != 'Last Quarter']

            # # Ectract financial statement
            # self.prepare_financial_statement()

            # print(f"Data loaded successfully from {self.csv_file_path}")
            return f"Data loaded successfully from {self.csv_file_path}"

        except Exception as e:
            return f"Error loading data: {e}"

    @staticmethod
    def transform_data(df) -> pd.DataFrame:
      df.set_index(df.columns[0], inplace=True)
      
      return df.astype(np.float64)

    @logger(separator='*')
    def prepare_financial_statement(self) -> pd.DataFrame:
          
        try:
            # Separate different financial statements
            income_stmt_start = 0
            balance_sheet_start = self.data[self.data.iloc[:, 0].str.contains('Balance Sheet', na=False)].index[0]
            cash_flow_start = self.data[self.data.iloc[:, 0].str.contains('Cash Flow', na=False)].index[0]

            # Extract income statement
            self.income_statement = self.data.iloc[income_stmt_start:balance_sheet_start].copy()
            self.income_statement = self.transform_data(self.income_statement)

            # Extract balance sheet
            self.balance_sheet = self.data.iloc[balance_sheet_start+1:cash_flow_start].copy()
            self.balance_sheet = self.transform_data(self.balance_sheet)

            # Extract cash flow statement
            self.cash_flow = self.data.iloc[cash_flow_start+1:].copy()
            self.cash_flow = self.transform_data(self.cash_flow)

            return "Financial Statements prepared successfully"

        except Exception as e:
            return f"Error preparing financial statement: {e}"

    def _get_financial_metric(self, statement, metric_name, column):
        try:
            return statement.loc[metric_name, column]
        except KeyError:
            return 0  # or np.nan, or log a warning

    def _safe_div(self, numerator, denominator):
        return numerator / denominator if denominator != 0 else 0

    def _get_raw_metrics(self, col):
          inc = self.income_statement.get(col, {})
          bal = self.balance_sheet.get(col, {})
          cf = self.cash_flow.get(col, {})

          return {
              'revenue': inc.get('Revenue', 0),
              'gross_profit': inc.get('Gross Profit', 0),
              'operating_income': inc.get('Operating Income', 0),
              'ebit': inc.get('EBIT', 0),
              'ebitda': inc.get('EBITDA', 0),
              'net_income': inc.get('Net Income', 0),
              'gross_profit_margin': inc.get('Gross Profit Margin %', 0),
              'operating_margin': inc.get('Operating Margin %', 0),
              'ebit_margin': inc.get('EBIT Margin %', 0),
              'ebitda_margin': inc.get('EBITDA Margin %', 0),
              'net_profit_margin': inc.get('Net Profit Margin %', 0),
              'revenue_growth': inc.get('Revenue Growth %', 0),
              'diluted_eps': inc.get('Diluted EPS', 0),
              'dividend': inc.get('Dividend Per Share', 0),
              'total_assets': bal.get('Total Assets', 0),
              'total_liabilities': bal.get('Total Liabilities', 0),
              'total_equity': bal.get('Total Equity', 0),
              'total_debt': bal.get('Total Debt', 0),
              'cash': bal.get('Cash And Cash Equivalents', 0),
              'working_capital': bal.get('Working Capital', 0),
              'shares_outstanding': bal.get('Shares Outstanding Capital', 0),
              'bps': bal.get('Book Value per Share', 0),
              'cfo': cf.get('Cash From Operating Activities', 0),
              'cfi': cf.get('Cash From Investing Activities', 0),
              'cff': cf.get('Cash From Financing Activities', 0),
              'fcf': cf.get('Free Cash Flow', 0),
              'capex': cf.get('Capital Expenditure', 0),
          }

    def _compute_derived_metrics(self, base):
        """Compute ratios and derived financial metrics"""
        net_income = base['net_income']
        total_equity = base['total_equity']
        total_assets = base['total_assets']
        total_liabilities = base['total_liabilities']
        operating_income = base['operating_income']
        revenue = base['revenue']
        fcf = base['fcf']
        total_debt = base['total_debt']

        # profitability ratio
        roe = self._safe_div(net_income, total_equity) * 100
        roa = self._safe_div(net_income, total_assets) * 100
        nopat = operating_income * 0.75  # assuming 25% tax
        invested_capital = total_liabilities  # simplified assumption
        roic = self._safe_div(nopat, invested_capital) * 100

        # solvency ratio
        de_ratio = self._safe_div(total_debt, total_equity)
        da_ratio = self._safe_div(total_debt, total_assets)
        equity_ratio = self._safe_div(total_equity, total_assets) * 100

        #efficiency and financial health assessments
        fcf_ratio = self._safe_div(fcf, revenue) * 100
        fcf_ni_ratio = self._safe_div(fcf, net_income)
        short_term_debt = total_liabilities - total_debt
        asset_turnover_ratio = self._safe_div(revenue, total_assets)  # asset turnover ratio

        return {
            'roe': roe,
            'roa': roa,
            'roic': roic,
            'de_ratio': de_ratio,
            'da_ratio': da_ratio,
            'equity_ratio': equity_ratio,
            'fcf_ratio': fcf_ratio,
            'fcf_ni_ratio': fcf_ni_ratio,
            'short_term_debt': short_term_debt,
            'long_term_debt': total_debt,
            'equity': total_equity,
            'asset_turnover_ratio': asset_turnover_ratio
        }

    def _compute_growth_metrics(self, metrics_dict, col, idx):
        """Compute growth metrics like EPS growth"""
        if idx > 0:
            prev_col = self.dates[idx - 1]
            prev_eps = metrics_dict[prev_col].get('diluted_eps', 0)
            curr_eps = metrics_dict[col]['diluted_eps']
            eps_growth = self._safe_div((curr_eps - prev_eps), prev_eps) * 100
        else:
            eps_growth = 0
        return {'eps_growth': eps_growth}

    def _calculate_financial_metrics(self):
        """Main method to calculate all financial metrics per column"""

        for idx, col in enumerate(self.dates):
            base = self._get_raw_metrics(col)
            derived = self._compute_derived_metrics(base)
            self.metrics[col] = {**base, **derived}
            growth = self._compute_growth_metrics(self.metrics, col, idx)
            self.metrics[col].update(growth)

    @staticmethod
    def transpose_metrics_for_plotting(metrics: dict, data_map: dict, dates: list) -> dict:
        """
        Transpose column-wise metrics into a structure suitable for plotting.
        """
        plot_data = {display_name: [] for display_name in data_map}

        for col in dates:
            raw_metrics = metrics.get(col, {})
            for display_name, internal_key in data_map.items():
                plot_data[display_name].append(raw_metrics.get(internal_key, 0))

        return plot_data

    @staticmethod
    def slice_dictionary_with_map_key(data_dict, map_key):
        """
        Slice a dictionary using values in map_key as internal keys.
        
        """
        result = {}
        internal_keys_to_keep = set(map_key.values())
      
        for year, internal_dict in data_dict.items():
            result[year] = {k: v for k, v in internal_dict.items() if k in internal_keys_to_keep}
        
        return result

    def prepare_data(self):
        """
        Helper function to load data and preprare financial statements for analysis
        """

        self.load_data()
        self.prepare_financial_statement()
        self._calculate_financial_metrics()

# def process_financial_data(csv_file_path: str, output_dir: str = None) -> Dict:
#     """
#     Main function to process financial data and generate analysis results.

#     Args:
#         csv_file_path: Path to the CSV file containing financial data
#         output_dir: Directory to save output files (optional)

#     Returns:
#         Dictionary containing analysis results
#     """
#     # Initialize analyzer and process data
#     analyzer = FinancialDataAnalyzer(csv_file_path)
#     analyzer.clean_data()

#     # Generate analysis results
#     ratios = analyzer.calculate_financial_ratios()
#     revenue_trends = analyzer.analyze_revenue_trends()
#     profitability = analyzer.analyze_profitability()
#     financial_health = analyzer.analyze_financial_health()
#     cash_flow = analyzer.analyze_cash_flow()

#     # Create visualization data
#     viz_data = analyzer.create_visualization_data()

#     # Export data for dashboard if output_dir is provided
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#         analyzer.export_data_for_dashboard(f"{output_dir}/financial_dashboard_data.json")

#         # Generate the comprehensive plot instead of multiple plots
#         analyzer.generate_comprehensive_plot(f"{output_dir}/financial_analysis_plot.png")

#     # Return analysis results
#     return {
#         'financial_ratios': ratios,
#         'revenue_trends': revenue_trends,
#         'profitability': profitability,
#         'financial_health': financial_health,
#         'cash_flow': cash_flow,
#         'visualization_data': viz_data
#     }

# def create_dashboard_components(data_file_path: str) -> Dict[str, Dict]:
#     """
#     Create component configurations for a React dashboard.

#     Args:
#         data_file_path: Path to the JSON data file

#     Returns:
#         Dictionary with dashboard component configurations
#     """
#     # Load the data
#     with open(data_file_path, 'r') as f:
#         data = json.load(f)

#     # Create dashboard components configuration
#     dashboard = {
#         'overview': {
#             'type': 'summary',
#             'title': 'Company Overview',
#             'data': data['company_overview']
#         },
#         'revenue_chart': {
#             'type': 'line_chart',
#             'title': 'Revenue and Growth',
#             'data': data['time_series_data']['revenue_growth'],
#             'config': {
#                 'x_axis': 'years',
#                 'y_axis': [
#                     {'key': 'revenue', 'name': 'Revenue', 'type': 'bar'},
#                     {'key': 'growth_rate', 'name': 'Growth Rate (%)', 'type': 'line', 'y_axis_id': 'right'}
#                 ]
#             }
#         },
#         'margins_chart': {
#             'type': 'line_chart',
#             'title': 'Profitability Margins',
#             'data': data['time_series_data']['margins'],
#             'config': {
#                 'x_axis': 'years',
#                 'y_axis': [
#                     {'key': 'gross_margin', 'name': 'Gross Margin (%)'},
#                     {'key': 'operating_margin', 'name': 'Operating Margin (%)'},
#                     {'key': 'net_margin', 'name': 'Net Margin (%)'}
#                 ]
#             }
#         },
#         'balance_sheet_chart': {
#             'type': 'stacked_bar',
#             'title': 'Balance Sheet Components',
#             'data': data['time_series_data']['balance_sheet'],
#             'config': {
#                 'x_axis': 'years',
#                 'y_axis': [
#                     {'key': 'assets', 'name': 'Total Assets'},
#                     {'key': 'liabilities', 'name': 'Total Liabilities'},
#                     {'key': 'equity', 'name': 'Total Equity'}
#                 ]
#             }
#         },
#         'cash_flow_chart': {
#             'type': 'line_chart',
#             'title': 'Cash Flow Components',
#             'data': data['time_series_data']['cash_flow'],
#             'config': {
#                 'x_axis': 'years',
#                 'y_axis': [
#                     {'key': 'operating_cf', 'name': 'Operating CF'},
#                     {'key': 'investing_cf', 'name': 'Investing CF'},
#                     {'key': 'financing_cf', 'name': 'Financing CF'},
#                     {'key': 'free_cf', 'name': 'Free CF'}
#                 ]
#             }
#         },
#         'key_ratios': {
#             'type': 'multi_chart',
#             'title': 'Key Financial Ratios',
#             'charts': [
#                 {
#                     'type': 'line_chart',
#                     'title': 'Return Ratios',
#                     'data': data['time_series_data']['key_ratios'].get('ROA', {}),
#                     'config': {
#                         'x_axis': 'years',
#                         'y_axis': [
#                             {'key': 'values', 'name': 'Return on Assets (%)'}
#                         ]
#                     }
#                 },
#                 {
#                     'type': 'line_chart',
#                     'title': 'Debt to Equity',
#                     'data': data['time_series_data']['key_ratios'].get('Debt_to_Equity', {}),
#                     'config': {
#                         'x_axis': 'years',
#                         'y_axis': [
#                             {'key': 'values', 'name': 'Debt to Equity'}
#                         ]
#                     }
#                 },
#                 {
#                     'type': 'line_chart',
#                     'title': 'Earnings Per Share',
#                     'data': data['time_series_data']['key_ratios'].get('EPS', {}),
#                     'config': {
#                         'x_axis': 'years',
#                         'y_axis': [
#                             {'key': 'values', 'name': 'EPS'}
#                         ]
#                     }
#                 }
#             ]
#         }
#     }

#     return dashboard


