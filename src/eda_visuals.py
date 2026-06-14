import os
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("⏳ Loading dataset from SQLite database...")
    
    # 1. Establish database connection
    project_root = os.path.abspath(os.getcwd())
    database_file_path = os.path.join(project_root, 'atm_analytics.db')
    engine = create_engine(f"sqlite:///{database_file_path}")
    
    # 2. Extract data into Pandas
    df = pd.read_sql_query("SELECT * FROM aggregated_atm_features;", con=engine)
    
    if len(df) == 0:
        print("⚠️ The database table is empty! Please run your aggregation script first.")
        return

    # 3. Explicitly parse datetime formats
    df['Transaction_Date'] = pd.to_datetime(df['Transaction_Date'])
    
    # Re-extract calendar features for grouping stability
    df['Is_Weekend'] = df['Transaction_Date'].dt.dayofweek.isin([5, 6]).astype(int)

    print("📊 Preparing visualization aggregates...")
    # Aggregate total daily network volume
    daily_trend = df.groupby('Transaction_Date').agg(
        Daily_Total_Cash=('Total_Cash_Out', 'sum'),
        Is_Weekend=('Is_Weekend', 'max')
    ).reset_index()

    # 4. Build the Matplotlib/Seaborn Figure
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(15, 6))

    # Plot baseline demand trend line
    sns.lineplot(data=daily_trend, x='Transaction_Date', y='Daily_Total_Cash', 
                 color='#bcbd22', linewidth=2, label='Daily Network Cash Out')

    # Highlight weekend surges with distinct markers
    weekends = daily_trend[daily_trend['Is_Weekend'] == 1]
    plt.scatter(weekends['Transaction_Date'], weekends['Daily_Total_Cash'], 
                color='#1f77b4', s=50, label='Weekend Demand Peak', zorder=5)

    plt.title('ATM Network Cash Demand Fluctuations & Cyclic Surges (2018)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Timeline', fontsize=12)
    plt.ylabel('Total Cash Out Volume', fontsize=12)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    print("🚀 Launching interactive visualization window...")
    # In a python script, plt.show() halts execution and opens a separate window
    plt.show() 
    print("👋 Window closed. Script finished successfully.")

if __name__ == "__main__":
    main()