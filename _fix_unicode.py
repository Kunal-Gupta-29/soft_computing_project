"""
_fix_unicode.py
Replaces ALL non-ASCII characters in Python source files with ASCII equivalents.
Also adds  # -*- coding: utf-8 -*-  header if needed.
Run: python _fix_unicode.py
"""
import os
import re

FILES = [
    'train.py', 'preprocess.py', 'model.py', 'config.py', 'gradcam.py',
    'autism_detector.py', 'realtime.py', 'app.py', 'ga_optimizer.py', 'train_autism.py',
    'evaluate.py', 'emotion_tracker.py', 'emotion.py',
]

# Complete replacement map
CHARS_MAP = {
    # Arrows
    '\u2190': '<-',  '\u2192': '->',  '\u2191': '^',  '\u2193': 'v',
    '\u21d2': '=>',  '\u21d0': '<=',
    # Box drawing (single)
    '\u2500': '-',   '\u2502': '|',   '\u250c': '+',  '\u2510': '+',
    '\u2514': '+',   '\u2518': '+',   '\u251c': '+',  '\u2524': '+',
    '\u252c': '+',   '\u2534': '+',   '\u253c': '+',
    # Box drawing (double)
    '\u2550': '=',   '\u2551': '|',   '\u2554': '+',  '\u2557': '+',
    '\u255a': '+',   '\u255d': '+',   '\u2560': '+',  '\u2563': '+',
    '\u2566': '+',   '\u2569': '+',   '\u256c': '+',
    # Box drawing (mixed)
    '\u2552': '+',   '\u2555': '+',   '\u2558': '+',  '\u255b': '+',
    # Dashes / quotes
    '\u2014': '--',  '\u2013': '-',   '\u2012': '-',
    '\u2018': "'",   '\u2019': "'",   '\u201c': '"',  '\u201d': '"',
    '\u2039': '<',   '\u203a': '>',
    # Math symbols
    '\u00d7': 'x',   '\u00f7': '/',   '\u00b1': '+/-', '\u2248': '~=',
    '\u2264': '<=',  '\u2265': '>=',  '\u2260': '!=',  '\u221e': 'inf',
    '\u03b1': 'alpha', '\u03b2': 'beta', '\u03c3': 'sigma',
    # Bullets / stars
    '\u2022': '*',   '\u2023': '>',   '\u25cf': '*',  '\u25cb': 'o',
    '\u2605': '*',   '\u2606': '*',
    # Check marks (used in print statements)
    '\u2714': 'OK',  '\u2716': 'X',   '\u2718': 'X',
    '\u2713': 'OK',
    # Emoji-adjacent
    '\u00e9': 'e',   '\u00e0': 'a',   '\u00fc': 'u',  '\u00f6': 'o',
    # Misc
    '\u00a0': ' ',   '\u200b': '',    '\u200c': '',   '\u200d': '',
    '\ufeff': '',    # BOM
}

# Generic fallback: replace any remaining non-ASCII with '?'
def clean(txt):
    for bad, good in CHARS_MAP.items():
        txt = txt.replace(bad, good)
    # Any remaining > 127 are replaced with '?'
    result = []
    for ch in txt:
        if ord(ch) > 127:
            result.append('?')
        else:
            result.append(ch)
    return ''.join(result)

fixed_files = 0
for fname in FILES:
    if not os.path.exists(fname):
        continue
    txt = open(fname, encoding='utf-8').read()
    new = clean(txt)
    if new != txt:
        open(fname, 'w', encoding='utf-8').write(new)
        bad_count = sum(1 for c in txt if ord(c) > 127)
        print(f'Fixed {bad_count:>4} non-ASCII chars in {fname}')
        fixed_files += 1
    else:
        print(f'  OK (no changes):  {fname}')

print(f'\nDone. Fixed {fixed_files} files.')

# Verify
remaining = 0
for fname in FILES:
    if not os.path.exists(fname):
        continue
    txt = open(fname, encoding='utf-8').read()
    bad = [c for c in txt if ord(c) > 127]
    if bad:
        print(f'WARNING: {fname} still has {len(bad)} non-ASCII chars!')
        remaining += len(bad)
if remaining == 0:
    print('Verification passed: all files are clean ASCII.')
