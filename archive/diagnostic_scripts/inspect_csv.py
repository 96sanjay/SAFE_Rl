import pandas as pd
import os

def inspect(path):
    print(f"\n--- INSPECTING: {path} ---")
    if not os.path.exists(path):
        print("❌ File NOT found.")
        return

    try:
        df = pd.read_csv(path)
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        # Check Step 4000 (Mid-year)
        if len(df) > 4000:
            row = df.iloc[4000]
            print(f"  [Row 4000 Data]")
            # Use .get() to avoid crashing if a column is missing
            print(f"    Step:            {row.get('step', 'N/A')}")
            print(f"    Grid Import:     {row.get('grid_import_kwh', 'N/A')}")
            print(f"    SoC Mean:        {row.get('soc_mean', 'N/A')}")
            print(f"    Cost:            {row.get('step_cost', 'N/A')}")
            print(f"    Discomfort:      {row.get('thermal_discomfort', 'N/A')}")
            print(f"    EV Cost:         {row.get('cost_ev_departure', 'N/A')}")
        else:
            print("  ❌ Data is too short (less than 4000 steps).")
            
    except Exception as e:
        print(f"  ❌ Error reading file: {e}")

if __name__ == "__main__":
    print("="*60)
    print(" CSV DATA FORENSICS")
    print("="*60)

    # Check all 3 files
    inspect("runs/kpi_no_control_kpis.csv")
    inspect("runs/kpi_citylearn_default_kpis.csv")
    inspect("runs/kpi_rbc_advanced_kpis.csv")
