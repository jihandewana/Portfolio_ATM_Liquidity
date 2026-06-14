import os
import pandas as pd
from sqlalchemy import create_engine

def import_csv_to_sqlite():
    # 1. Initialize connection to local SQLite database
    db_path = 'sqlite:///atm_analytics.db'
    engine = create_engine(db_path)
    print("⏳ Connecting to local SQLite database...")

    data_folder = 'data'
    
    # Check if the data folder exists
    if not os.path.exists(data_folder):
        print(f"❌ Error: The folder '{data_folder}' does not exist. Please create it first.")
        return

    # List all files inside the data folder
    all_files = os.listdir(data_folder)
    print(f"🔍 Found files in data folder: {all_files}")

    # 2. Dynamic file finder based on keywords
    file_mapping = {}
    for file in all_files:
        file_lower = file.lower()
        full_path = os.path.join(data_folder, file)
        
        if 'manage' in file_lower or 'master' in file_lower:
            file_mapping[full_path] = 'atm_master'
        elif 'trans' in file_lower:
            file_mapping[full_path] = 'atm_transactions'
        elif 'risk' in file_lower or 'assess' in file_lower:
            file_mapping[full_path] = 'corporate_risk'

    if not file_mapping:
        print("❌ Error: No matching CSV files found. Please make sure your Kaggle files are inside the 'data/' folder.")
        return

    # 3. Load detected files into SQLite
    for csv_file, table_name in file_mapping.items():
        print(f"📖 Reading file: {csv_file}...")
        
        # Read CSV into a Pandas DataFrame
        df = pd.read_csv(csv_file)
        
        print(f"🚀 Loading data into SQL table: '{table_name}' ({len(df)} rows)...")
        # Using chunksize to safely load large transaction tables without system crash
        df.to_sql(table_name, con=engine, if_exists='replace', index=False, chunksize=10000)
        
        print(f"✅ Table '{table_name}' successfully created!\n")

    print("🏁 ETL Process completed successfully!")

if __name__ == "__main__":
    import_csv_to_sqlite()