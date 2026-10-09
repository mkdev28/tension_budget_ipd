with open('train_pymc_hierarchical_model_v2.py', 'rb') as f:
    raw = f.read()

# Remove null bytes
clean = raw.replace(b'\x00', b'')

text = clean.decode('utf-8', errors='ignore')
idx = text.find('\" \" \"')
if idx != -1:
    text = text[:idx]

with open('train_pymc_hierarchical_model_v2.py', 'w', encoding='utf-8') as f:
    f.write(text.strip() + '\n')
