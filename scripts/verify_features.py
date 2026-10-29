"""Run the notebook's Sobel implementation against an independent NumPy reference."""
import ast
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
nb = json.loads((root / 'notebooks/handwritten-symbols.ipynb').read_text(encoding='utf8'))
namespace = {'np': np}
for cell in nb['cells']:
    source = ''.join(cell['source'])
    if cell['cell_type'] != 'code' or 'def extract_sobel_features' not in source:
        continue
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('H_filter', 'V_filter') for t in node.targets))]
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<notebook-feature-functions>', 'exec'), namespace)
if 'extract_sobel_features' not in namespace:
    raise AssertionError('Sobel feature implementation was not found')
image = np.arange(45 * 45, dtype=np.float32).reshape(45, 45) % 251
windows = np.lib.stride_tricks.sliding_window_view(image, (3, 3))
h = np.einsum('ijkl,kl->ij', windows, namespace['H_filter'])
v = np.einsum('ijkl,kl->ij', windows, namespace['V_filter'])
reference = np.abs(h) + np.abs(v)
reference /= reference.max()
actual = namespace['extract_sobel_features'](image)
np.testing.assert_allclose(actual, reference.ravel(), atol=1e-6)
zero = namespace['extract_sobel_features'](np.zeros((45, 45), dtype=np.uint8))
assert zero.shape == (43 * 43,) and np.isfinite(zero).all() and not zero.any()
print(json.dumps({'independent_sobel_reference': 'passed', 'constant_image_boundary': 'passed', 'feature_length': len(actual)}))
