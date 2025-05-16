
from datetime import datetime
from pathlib import Path

from custom_utils import DataProcessing
from visualization import DataVisualization


def create_financial_dashboard(csv_file_path, title=None, save=False, dir='./results/', output_file=f"dashboard.png"):
    """
    Create a financial dashboard from CSV data

    Args:
        csv_data: CSV data as string or file path
        title: Title for the dashboard
        output_file: Output file path for saved dashboard

    Returns:
        Path to saved dashboard image
    """

    # Create the dashboard
    data_processor = DataProcessing(csv_file_path)
    data_processor.prepare_data()
    data_visualization = DataVisualization(data_processor, title=title)

    # Save the dashboard
    if save:
        folder_name = datetime.now().strftime("%Y-%m-%d")
        output_dir = Path(dir) / folder_name

        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

    if title:
      output_file = f"{output_dir}/dashboard_{title}_{datetime.now().strftime('%Y%m%d_%H_%M_%S')}.png"
    else:
      output_file = f"{output_dir}/dashboard_{datetime.now().strftime('%Y%m%d_%H_%M_%S')}.png"

    return data_visualization.show_dashboard(save=save, filename=output_file)

if __name__ == "__main__":
    csv_file_path = "./data/summary-31-03-2025.csv"
    title = 'APPL'

    try:
        # Create and save the dashboard
        create_financial_dashboard(csv_file_path, title, save=True)

    except FileNotFoundError:
        print(f"CSV file not found: {csv_file_path}")
    except Exception as e:
        print(f"Error creating dashboard: {e}")
