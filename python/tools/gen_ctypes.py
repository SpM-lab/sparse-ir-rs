#!/usr/bin/env python3
"""
Generate Python ctypes bindings from C-API header using libclang.

This script parses sparseir.h and generates ctypes_autogen.py with function
prototypes that can be automatically applied to the loaded library.
"""

import sys
from pathlib import Path

try:
    from clang import cindex
    from clang.cindex import Index, CursorKind
except ImportError:
    print("ERROR: clang (libclang) is required. Install with: pip install clang")
    sys.exit(1)

# Map C types to ctypes
# Note: On macOS, int64_t is typically long, not long long
CTYPE_MAP = {
    'int': 'c_int',
    'double': 'c_double',
    'bool': 'c_bool',
    'int64_t': 'c_int64',
    'long long': 'c_int64',  # int64_t is often typedef'd to long long on Linux
    'long long int': 'c_int64',
    'long': 'c_int64',  # On macOS, int64_t is typically long
    'size_t': 'c_size_t',
    'void': 'None',
    'c_complex': 'c_double_complex',
}

# Map opaque types
OPAQUE_TYPES = {
    'spir_kernel': 'spir_kernel',
    'spir_funcs': 'spir_funcs',
    'spir_basis': 'spir_basis',
    'spir_sampling': 'spir_sampling',
    'spir_sve_result': 'spir_sve_result',
}


def get_cursor_type_str(cursor_type):
    """Convert clang Type to Python ctypes string."""
    spelling = cursor_type.spelling
    
    # Handle pointers
    if cursor_type.kind == cindex.TypeKind.POINTER:
        pointee = cursor_type.get_pointee()
        pointee_spelling = pointee.spelling
        pointee_canonical = pointee.get_canonical()
        pointee_canonical_spelling = pointee_canonical.spelling if pointee_canonical else pointee_spelling
        
        # Special handling for int64_t: check if the original spelling or canonical contains int64_t
        # even if canonical resolves to long/long long
        if 'int64_t' in pointee_spelling or 'int64_t' in pointee_canonical_spelling:
            pointee_str = 'c_int64'
        # Special handling for c_complex: check if it's c_complex type
        elif 'c_complex' in pointee_spelling or 'c_complex' in pointee_canonical_spelling:
            pointee_str = 'c_double_complex'
        # Check canonical type first for other typedefs
        elif pointee_canonical_spelling in CTYPE_MAP:
            pointee_str = CTYPE_MAP[pointee_canonical_spelling]
        elif pointee_spelling in CTYPE_MAP:
            pointee_str = CTYPE_MAP[pointee_spelling]
        else:
            pointee_str = get_cursor_type_str(pointee)
        
        # Check if it's an opaque type
        if pointee_str in OPAQUE_TYPES:
            return OPAQUE_TYPES[pointee_str]
        
        # Handle char* -> c_char_p
        if pointee_str == 'char':
            return 'c_char_p'
        
        # Generic pointer -> POINTER(type)
        return f'POINTER({pointee_str})'
    
    # Handle const
    if 'const' in spelling:
        spelling = spelling.replace('const', '').strip()
    
    # Handle typedefs - check canonical type for int64_t, size_t, c_complex, etc.
    # On some platforms, int64_t might be typedef'd to long long
    canonical = cursor_type.get_canonical()
    canonical_spelling = canonical.spelling if canonical else spelling
    
    # Special handling for c_complex
    if 'c_complex' in spelling or 'c_complex' in canonical_spelling:
        return 'c_double_complex'
    
    # Check canonical type first (handles typedefs)
    if canonical_spelling in CTYPE_MAP:
        return CTYPE_MAP[canonical_spelling]
    
    # Also check if canonical spelling contains known types
    for ctype, pytype in CTYPE_MAP.items():
        if canonical_spelling.endswith(ctype) or canonical_spelling == ctype:
            return pytype
    
    # Map basic types (exact match on original spelling)
    if spelling in CTYPE_MAP:
        return CTYPE_MAP[spelling]
    
    # Handle opaque types directly
    if spelling in OPAQUE_TYPES:
        return OPAQUE_TYPES[spelling]
    
    # Default: return as-is (will need manual mapping)
    return spelling


def parse_function(cursor):
    """Parse a function cursor and return (name, restype, argtypes)."""
    if cursor.kind != CursorKind.FUNCTION_DECL:
        return None
    
    name = cursor.spelling
    
    # Only process spir_* functions
    if not name.startswith('spir_'):
        return None
    
    # Get return type
    result_type = cursor.result_type
    restype = get_cursor_type_str(result_type)
    
    # Get arguments
    argtypes = []
    for arg in cursor.get_arguments():
        arg_type = get_cursor_type_str(arg.type)
        # Debug output for c_complex parameters
        if 'complex' in arg.type.spelling.lower() or 'complex' in str(arg.type.get_canonical().spelling if arg.type.get_canonical() else '').lower():
            pointee = arg.type.get_pointee() if arg.type.kind == cindex.TypeKind.POINTER else None
            if pointee:
                print(f"DEBUG {name} arg '{arg.spelling}': spelling='{pointee.spelling}', canonical='{pointee.get_canonical().spelling if pointee.get_canonical() else 'N/A'}' -> {arg_type}")
        argtypes.append(arg_type)
    
    return (name, restype, argtypes)


def parse_header(header_path, include_dirs):
    """Parse the C header and extract function signatures."""
    index = Index.create()
    
    # Build include arguments
    # Include standard headers for int64_t, size_t, etc.
    # Use C99 standard to ensure int64_t is properly defined
    args = ['-x', 'c-header', '-std=c99', '-D__STDC_LIMIT_MACROS', '-D__STDC_CONSTANT_MACROS']
    for inc_dir in include_dirs:
        args.extend(['-I', inc_dir])
    
    # Add system include paths for stdint.h
    import platform
    if platform.system() == 'Darwin':
        # macOS: try common paths
        import subprocess
        try:
            result = subprocess.run(['xcrun', '--show-sdk-path'], capture_output=True, text=True)
            if result.returncode == 0:
                sdk_path = result.stdout.strip()
                args.extend(['-isysroot', sdk_path])
        except:
            pass
    
    # Parse the translation unit
    try:
        tu = index.parse(header_path, args=args)
    except Exception as e:
        print(f"ERROR: Failed to parse {header_path}: {e}")
        sys.exit(1)
    
    functions = {}
    
    def visit_cursor(cursor):
        """Recursively visit cursors to find function declarations."""
        if cursor.location.file and cursor.location.file.name == header_path:
            func_info = parse_function(cursor)
            if func_info:
                name, restype, argtypes = func_info
                functions[name] = (restype, argtypes)
        
        # Recurse into children
        for child in cursor.get_children():
            visit_cursor(child)
    
    visit_cursor(tu.cursor)
    
    return functions


def generate_python_file(functions, output_path):
    """Generate the Python file with function prototypes."""
    lines = [
        '"""',
        'Auto-generated ctypes bindings from C-API header.',
        'DO NOT EDIT THIS FILE MANUALLY.',
        'Generated by tools/gen_ctypes.py',
        '"""',
        '',
        'import ctypes',
        'from ctypes import (',
        '    c_int, c_double, c_int64, c_size_t, c_bool,',
        '    POINTER, c_char_p, Structure,',
        ')',
        '',
        'from .ctypes_wrapper import (',
        '    spir_kernel, spir_funcs, spir_basis,',
        '    spir_sampling, spir_sve_result,',
        ')',
        '',
        '# Custom complex type (matches core.py definition)',
        'class c_double_complex(Structure):',
        '    """Complex number as ctypes Structure."""',
        '    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]',
        '',
        '    @property',
        '    def value(self):',
        '        return self.real + 1j * self.imag',
        '',
        '',
        '# Function prototypes: {name: (restype, [argtypes])}',
        'FUNCTIONS = {',
    ]
    
    # Sort functions by name for consistent output
    for name in sorted(functions.keys()):
        restype, argtypes = functions[name]
        # Format argtypes as a list of strings (will be evaluated in core.py)
        argtypes_list = ', '.join([f"'{at}'" for at in argtypes])
        lines.append(f"    '{name}': ('{restype}', [{argtypes_list}]),")
    
    lines.extend([
        '}',
        '',
    ])
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Generated {output_path} with {len(functions)} functions")


def main():
    """Main entry point."""
    # Determine paths
    script_dir = Path(__file__).parent
    python_dir = script_dir.parent
    project_root = python_dir.parent
    
    # Try to find header in copied location first (after setup_build.py)
    # Then fall back to original location
    header_candidates = [
        python_dir / 'include' / 'sparseir' / 'sparseir.h',  # Copied location
        project_root / 'backend' / 'cxx' / 'include' / 'sparseir' / 'sparseir.h',  # Original
    ]
    
    header_path = None
    for candidate in header_candidates:
        if candidate.exists():
            header_path = candidate
            break
    
    if not header_path:
        print(f"ERROR: Header not found. Tried:")
        for candidate in header_candidates:
            print(f"  {candidate}")
        sys.exit(1)
    
    output_path = python_dir / 'pylibsparseir' / 'ctypes_autogen.py'
    
    # Include directories
    include_dirs = [
        str(python_dir / 'include'),  # Copied location
        str(python_dir / 'include' / 'sparseir'),
        str(project_root / 'backend' / 'cxx' / 'include'),  # Original
        str(project_root / 'backend' / 'cxx' / 'include' / 'sparseir'),
    ]
    
    if not header_path.exists():
        print(f"ERROR: Header not found: {header_path}")
        sys.exit(1)
    
    print(f"Parsing {header_path}...")
    functions = parse_header(str(header_path), include_dirs)
    
    if not functions:
        print("WARNING: No functions found!")
        sys.exit(1)
    
    print(f"Found {len(functions)} functions")
    generate_python_file(functions, output_path)


if __name__ == '__main__':
    main()

