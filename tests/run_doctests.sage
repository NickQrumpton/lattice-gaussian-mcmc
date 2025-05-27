#!/usr/bin/env sage
"""
Run doctests for NTRU lattice module.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sage.all import *
import doctest

# Run doctests for NTRU module
ntru_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'lattices', 'ntru.py')

print("Running doctests for NTRU lattice module...")
print(f"Testing: {ntru_path}\n")

# Create a doctest runner
runner = doctest.DocTestRunner(verbose=True)

# Load the module and run doctests
with open(ntru_path, 'r') as f:
    module_text = f.read()

# Parse doctests
parser = doctest.DocTestParser()
tests = parser.get_doctest(module_text, {}, "ntru.py", ntru_path, 0)

# Run the tests
runner.run(tests)

# Print summary
if runner.failures == 0:
    print(f"\n✅ All {runner.tries} doctests passed!")
else:
    print(f"\n❌ {runner.failures} out of {runner.tries} doctests failed!")
    sys.exit(1)