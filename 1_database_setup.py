import duckdb
import time
import os

def setup_industry_database():
    print("Starting enterprise data ingestion and optimization...")
    start_time = time.time()

    # 1. Connect to DuckDB
    con = duckdb.connect('instacart_analytics.duckdb')

    # ⚡ OPTIMIZATION 1: Hardware Pragmas
    # Explicitly tell DuckDB to use your laptop's full power
    con.execute("PRAGMA threads=8;") # Adjust based on your CPU cores
    con.execute("PRAGMA memory_limit='10GB';") # Leaves 6GB for Windows/LLM

    # 2. Map the tables to your CSV files
    csv_files = {
        "aisles": "aisles.csv",
        "departments": "departments.csv",
        "products": "products.csv",
        "orders": "orders.csv",
        "order_products": "order_products__prior.csv"
    }

    # 3. Fast Ingestion Loop
    for table_name, file_name in csv_files.items():
        if os.path.exists(file_name):
            print(f"Ingesting {file_name} into table '{table_name}'...")
            try:
                con.execute(f"DROP TABLE IF EXISTS {table_name}")
                con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{file_name}');")
                print(f"Successfully loaded {table_name}")
            except Exception as e:
                print(f"Error loading {table_name}: {e}")
        else:
            print(f"Warning: {file_name} not found in the current directory. Skipping.")

    # ⚡ OPTIMIZATION 2: Build B-Tree Indexes on Foreign Keys
    print("\nBuilding B-Tree Indexes for lighting-fast LLM Joins...")
    index_queries = [
        "CREATE INDEX IF NOT EXISTS idx_op_product_id ON order_products(product_id);",
        "CREATE INDEX IF NOT EXISTS idx_p_product_id ON products(product_id);",
        "CREATE INDEX IF NOT EXISTS idx_p_aisle_id ON products(aisle_id);",
        "CREATE INDEX IF NOT EXISTS idx_p_dept_id ON products(department_id);"
    ]
    
    for idx_query in index_queries:
        print(f"Executing: {idx_query.split('ON')[0].strip()}...")
        con.execute(idx_query)

    # 4. Verification Check
    try:
        count = con.execute("SELECT COUNT(*) FROM order_products").fetchone()[0]
        print(f"\n✅ Verification: Total rows in order_products: {count:,}")
    except Exception as e:
        print("\nVerification skipped for order_products.")
        
    end_time = time.time()
    print(f"Database setup & optimization complete in {round(end_time - start_time, 2)} seconds.")
    
    # ⚡ OPTIMIZATION 3: Vacuum Database (Compresses and finalizes storage)
    con.execute("VACUUM;")
    con.close()

if __name__ == "__main__":
    setup_industry_database()