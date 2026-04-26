import h5py, json, os

models = [
    r'models\emotion_model.h5',
    r'models\emotion_model_tl.h5',
]
for m in models:
    print(f'\n=== {os.path.basename(m)} ===')
    print(f'  File size: {os.path.getsize(m) / 1e6:.2f} MB')
    try:
        with h5py.File(m, 'r') as f:
            keys = list(f.keys())
            print(f'  Top-level keys: {keys}')
            if 'model_config' in f.attrs:
                cfg = json.loads(f.attrs['model_config'])
                print(f'  Model class: {cfg.get("class_name", "N/A")}')
            if 'training_config' in f.attrs:
                tc = json.loads(f.attrs['training_config'])
                print(f'  Training config keys: {list(tc.keys())}')
            if 'optimizer_weights' in f:
                print('  Optimizer weights: PRESENT (model was trained)')
    except Exception as e:
        print(f'  Error: {e}')
