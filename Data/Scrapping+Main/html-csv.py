import os
import pandas as pd
from bs4 import BeautifulSoup
import csv

def met_data(month, year, base_path):
    """Extract meteorological data for a specific month and year."""
    file_path = os.path.join(base_path, str(year), f"{month}.html")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []

    with open(file_path, 'rb') as file_html:
        plain_text = file_html.read()

    tempD = []
    finalD = []

    soup = BeautifulSoup(plain_text, "lxml")
    for table in soup.findAll('table', {'class': 'medias mensuales numspan'}):
        for tbody in table:
            for tr in tbody:
                tempD.append(tr.get_text())

    rows = len(tempD) / 15

    for _ in range(round(rows)):
        newtempD = []
        for i in range(15):
            newtempD.append(tempD.pop(0))
        finalD.append(newtempD)

    # Remove unnecessary rows and columns
    if finalD:
        finalD.pop(0)  # Remove header row
        if len(finalD) > 0:
            finalD.pop(-1)  # Remove last row (if exists)

    for a in range(len(finalD)):
        finalD[a] = [finalD[a][1], finalD[a][2], finalD[a][3], finalD[a][4], finalD[a][5], 
                     finalD[a][7], finalD[a][8], finalD[a][9]]  # Keep only relevant columns

    return finalD


def combine_csv(year, output_path, base_path):
    """Combine data for a specific year into a single CSV file."""
    final_data = []

    # Output file path
    output_file = os.path.join(output_path, f"real_{year}.csv")

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM'])  # Write headers

        for month in range(1, 13):
            temp_data = met_data(month, year, base_path)
            if temp_data:
                final_data.extend(temp_data)

        # Write all data rows
        writer.writerows(final_data)
    print(f"Data for {year} saved to {output_file}")


def combine_all_years(output_path):
    """Combine data from all years into a single CSV file."""
    combined_data = []

    for year in range(2013, 2019):  # Years from 2013 to 2018
        file_path = os.path.join(output_path, f"real_{year}.csv")
        if os.path.exists(file_path):
            yearly_data = pd.read_csv(file_path).values.tolist()
            combined_data.extend(yearly_data)

    combined_file = os.path.join(output_path, "Real_Combine.csv")
    with open(combined_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['T', 'TM', 'Tm', 'SLP', 'H', 'VV', 'V', 'VM'])  # Write headers
        writer.writerows(combined_data)
    print(f"All years combined into {combined_file}")


if __name__ == "__main__":
    # Base directory containing HTML files
    base_path = "/Users/admin/Desktop/Data/Html_Data"

    # Output directory for processed data
    output_path = "/Users/admin/Desktop/Data/Real-Data"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Process data for each year
    for year in range(2013, 2019):  # From 2013 to 2018
        combine_csv(year, output_path, base_path)

    # Combine all years into a single CSV
    combine_all_years(output_path)
