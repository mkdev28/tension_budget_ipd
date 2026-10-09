"""
scratch_probe_schema.py — inspect raw column names from Supabase view
"""
import os, sys, requests
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}"}

response = requests.get(f"{SUPABASE_URL}/rest/v1/v_ml_training_pairs?select=*&limit=5", headers=headers)
response.raise_for_status()
df = pd.DataFrame(response.json())
print("Columns:", df.columns.tolist())
print("Total cols:", len(df.columns))
print("\nDtypes:\n", df.dtypes.to_string())
print("\nFirst row sample:")
print(df.iloc[0].to_string())
