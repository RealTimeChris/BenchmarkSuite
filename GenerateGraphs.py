import argparse
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re

# Set up command-line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description='Process benchmark results from a JSON file.')
    parser.add_argument('input_file', type=str, help='Path to the input JSON file')
    return parser.parse_args()

def main():
    args = parse_args()
    input_file = args.input_file
    
    # Load data from the specified JSON file
    with open(input_file, 'r') as file:
        data = json.load(file)

    output_folder = 'Graphs'
    os.makedirs(output_folder, exist_ok=True)

    df = pd.json_normalize(data, 'results')

    def sanitize_filename(filename):
        return re.sub(r'[<>:"/\\|?*]', ' ', filename)

    def increment_filename_if_exists(output_folder, filename):
        base, ext = os.path.splitext(filename)
        counter = 1
        new_filename = filename
        while os.path.exists(os.path.join(output_folder, new_filename)):
            new_filename = f"{base}_{counter}{ext}"
            counter += 1
        return new_filename

    def plot_and_save_benchmarks(df):
        tests = df["benchmarkName"].unique()

        for test in tests:
            sns.set_style("dark")
            sns.set_style(rc={'axes.facecolor': '#0d1117'})
            plt.figure(figsize=(10, 6))
            ax = plt.gca()

            test_data = df[df["benchmarkName"] == test]

            for i, library in enumerate(test_data["libraryName"].unique()):
                library_data = test_data[test_data["libraryName"] == library]
                if not library_data.empty:
                    color = library_data["benchmarkColor"].iloc[0]
                    time = library_data["resultTime"].iloc[0]
                    ax.bar(i, time, color=color, label=library)

            ax.set_xticks(range(len(test_data["libraryName"].unique())))
            ax.set_xticklabels(test_data["libraryName"].unique(), color='lightgray')
            ax.set_title(f'{test} Results', color='white')
            ax.set_xlabel('Library Name', color='white')
            ax.set_ylabel('Result Time (ms)', color='white')

            sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "#0d1117", "xtick.color": "#0d1117", "ytick.color": "#0d1117"})
            plt.gcf().set_facecolor("#0d1117")

            for text in ax.get_xticklabels() + ax.get_yticklabels():
                text.set_color('lightgray')

            ax.legend(title='Library', loc='best')
            sanitized_test_name = sanitize_filename(test)
            filename = f'{sanitized_test_name}_results.jpg'

            # Check if the file already exists and increment the filename if it does
            filename = increment_filename_if_exists(output_folder, filename)

            # Save the plot as a .jpg file
            plt.savefig(os.path.join(output_folder, filename), format='jpg')
            plt.close()

    # Call the function with the DataFrame
    plot_and_save_benchmarks(df)

if __name__ == "__main__":
    main()
