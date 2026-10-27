# Smart Mobility Analytics Studio

A beginner-friendly Python web application for uploading, profiling, cleaning, visualizing, and exporting transportation CSV data.

## Features in Version 1

- Upload CSV files
- Preview data
- Profile data types, missing values, unique values, and duplicates
- Select columns for missing-value handling
- Drop rows with missing values in selected columns
- Fill missing numeric values using mean or median
- Fill missing values with a constant
- Remove exact duplicate rows
- Convert one column to text, numeric, or date/time
- Create histograms, scatter plots, line charts, bar charts, and correlation heatmaps
- Download cleaned CSV and a Markdown data-quality report

## Run locally (Windows)

1. Install Python 3.11 or newer from https://www.python.org/downloads/
   - During installation, select **Add Python to PATH**.
2. Install VS Code from https://code.visualstudio.com/
3. Create/open this project folder in VS Code.
4. Open **Terminal → New Terminal**.
5. Create a virtual environment:

   ```powershell
   py -m venv .venv
   ```

6. Activate it in PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, use Command Prompt terminal and run:

   ```bat
   .venv\Scripts\activate.bat
   ```

7. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

8. Run the app:

   ```powershell
   streamlit run app.py
   ```

9. Open the local URL printed in the terminal (usually http://localhost:8501).

## First test

On the app's welcome screen, download the sample CSV. Then upload it and test:
- inspect the profile,
- select `average_speed_kmh` for mean/median filling,
- convert `timestamp` to Date/time,
- create a line chart,
- download the cleaned CSV and report.

## Important limitations

- Version 1 accepts CSV files only.
- Mean/median filling is intended for numeric data and can distort results if used without considering the data context.
- Converting a column to numeric or date/time turns unparseable values into missing values. Review the report.
- Correlation does not prove causation.
- The app processes uploaded files in the running Python process. Do not upload confidential or personally identifiable data to an untrusted deployment.
- This is a data analysis tool, not a validated traffic forecasting system yet.

## Suggested next steps

1. Add a cleaning history and undo/reset button.
2. Add data validation rules and column-role mapping for transport data.
3. Add map visualizations when valid latitude/longitude columns are present.
4. Add baseline and machine-learning traffic prediction with time-aware evaluation.
5. Add tests and deployment configuration before public use.
