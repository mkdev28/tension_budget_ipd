import os
import glob
import pandas as pd
import json

output_logs_path = 'output_logs'
sessions_summary = []

for user in os.listdir(output_logs_path):
    user_dir = os.path.join(output_logs_path, user)
    if not os.path.isdir(user_dir):
        continue
    
    for item in os.listdir(user_dir):
        item_path = os.path.join(user_dir, item)
        is_post_aug6 = False
        for date_str in ['20260806', '20260807', '20260808', '20260809', '20260810', '20260811', '20260812']:
            if date_str in item:
                is_post_aug6 = True
                break
        
        if is_post_aug6:
            if os.path.isdir(item_path):
                sub_files = os.listdir(item_path)
                feat_csv = [f for f in sub_files if f.endswith('_features.csv') or f == 'epoch_features.csv']
                feat_rows = 0
                labeled_rows = 0
                cols = []
                if feat_csv:
                    df = pd.read_csv(os.path.join(item_path, feat_csv[0]))
                    feat_rows = len(df)
                    cols = list(df.columns)
                    if 'subjective_strain_cr10' in df.columns:
                        labeled_rows = df['subjective_strain_cr10'].notna().sum()
                
                sessions_summary.append({
                    'user': user,
                    'session': item,
                    'type': 'folder',
                    'sub_files': sub_files,
                    'feat_rows': feat_rows,
                    'labeled_rows': labeled_rows,
                    'cols': cols
                })
            elif item.endswith('_features.csv'):
                df = pd.read_csv(item_path)
                feat_rows = len(df)
                cols = list(df.columns)
                labeled_rows = df['subjective_strain_cr10'].notna().sum() if 'subjective_strain_cr10' in df.columns else 0
                sessions_summary.append({
                    'user': user,
                    'session': item,
                    'type': 'csv',
                    'feat_rows': feat_rows,
                    'labeled_rows': labeled_rows,
                    'cols': cols
                })

print("=== SESSIONS RECORDED ON/AFTER AUGUST 6TH, 2026 ===")
total_rows = 0
total_labeled = 0
for s in sessions_summary:
    print(f"User: {s['user']} | Session: {s['session']} | Type: {s['type']} | Feature Rows: {s['feat_rows']} | Labeled: {s['labeled_rows']}")
    if s['sub_files']:
        print(f"   Sub-files: {s['sub_files']}")
    total_rows += s['feat_rows']
    total_labeled += s['labeled_rows']

print(f"\nTotal Post-Aug 6 Feature Rows: {total_rows}")
print(f"Total Post-Aug 6 Labeled Rows: {total_labeled}")

# Also check pre-Aug 6 for comparison
pre_rows = 0
pre_labeled = 0
for user in os.listdir(output_logs_path):
    user_dir = os.path.join(output_logs_path, user)
    if not os.path.isdir(user_dir): continue
    for item in os.listdir(user_dir):
        item_path = os.path.join(user_dir, item)
        is_pre_aug6 = False
        for date_str in ['20260801', '20260802', '20260803', '20260804', '20260805']:
            if date_str in item:
                is_pre_aug6 = True
                break
        if is_pre_aug6:
            if os.path.isdir(item_path):
                sub_files = os.listdir(item_path)
                feat_csv = [f for f in sub_files if f.endswith('_features.csv') or f == 'epoch_features.csv']
                if feat_csv:
                    df = pd.read_csv(os.path.join(item_path, feat_csv[0]))
                    pre_rows += len(df)
                    if 'subjective_strain_cr10' in df.columns:
                        pre_labeled += df['subjective_strain_cr10'].notna().sum()
            elif item.endswith('_features.csv'):
                df = pd.read_csv(item_path)
                pre_rows += len(df)
                if 'subjective_strain_cr10' in df.columns:
                    pre_labeled += df['subjective_strain_cr10'].notna().sum()

print(f"\nTotal Pre-Aug 6 Feature Rows: {pre_rows}")
print(f"Total Pre-Aug 6 Labeled Rows: {pre_labeled}")
