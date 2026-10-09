with open('train_pymc_hierarchical_model_v2.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

main_start = -1
for i, line in enumerate(lines):
    if 'if __name__ == "__main__":' in line:
        main_start = i
        break

if main_start != -1:
    clean_lines = lines[:main_start-3] 
    
    with open('train_pymc_hierarchical_model_v2_part2.py', 'r', encoding='utf-8') as f:
        part2_lines = f.readlines()
        
    part2_start = 0
    for i, line in enumerate(part2_lines):
        if 'STEP 5' in line:
            part2_start = i - 2
            break
            
    part2_clean = part2_lines[part2_start:]
    
    main_block = [
        '\n# ─────────────────────────────────────────────────────────────────────────────\n',
        '# MAIN\n',
        '# ─────────────────────────────────────────────────────────────────────────────\n',
        'if __name__ == "__main__":\n',
        '    df_raw = fetch_raw()\n',
        '    df_full = step1_audit(df_raw)\n',
        '    df_pairs = step2a_construct_targets(df_full)\n',
        '    SET_A, SET_B, SET_C = step2b_define_feature_sets()\n',
        '    flagged_session_ids, bounds_rationale = step2c_qc_gate(df_full)\n',
        '    step2d_outlier_policy()\n',
        '    step2e_eligibility(df_pairs)\n',
        '    \n',
        '    print("\\n--- Running Step 3 on each feature set (full dataset, no LOSO split yet) ---")\n',
        '    for set_name, fset in [("SET_A", SET_A), ("SET_B", SET_B), ("SET_C", SET_C)]:\n',
        '        df_scaled, step3_report = step3_feature_engineering(df_pairs, fset, set_name, train_mask=None)\n',
        '    \n',
        '    append_session_log(bounds_rationale, step3_report)\n',
        '    \n',
        '    run_all_models(df_pairs)\n',
        '    \n',
        '    print("\\n" + "=" * 60)\n',
        '    print("STEP 4-7 COMPLETE. Models evaluated.")\n',
        '    print("=" * 60)\n'
    ]
    
    with open('train_pymc_hierarchical_model_v2.py', 'w', encoding='utf-8') as f:
        f.writelines(clean_lines + part2_clean + main_block)
        
    print('Successfully re-assembled v2 script.')
else:
    print('Could not find __main__')
