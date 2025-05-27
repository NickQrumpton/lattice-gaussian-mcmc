#!/usr/bin/env python3
"""
Verify all publication materials are present and complete.
"""

from pathlib import Path
import json

def verify_materials():
    """Verify all expected files are present."""
    print("Publication Materials Verification")
    print("="*50)
    
    base_dir = Path('results')
    
    # Expected files
    expected_tables = [
        'table_1_algorithm_comparison.tex',
        'table_1_algorithm_comparison.csv',
        'table_2_cryptographic_parameters.tex', 
        'table_2_cryptographic_parameters.csv',
        'table_3_performance_benchmark.tex',
        'table_3_performance_benchmark.csv',
        'table_4_convergence_summary.tex',
        'table_4_convergence_summary.csv',
        'table_5_scaling_analysis.tex',
        'table_5_scaling_analysis.csv'
    ]
    
    expected_figures = [
        'figure_1_performance_basic.png',
        'figure_2_quality_basic.png', 
        'figure_3_convergence_basic.png'
    ]
    
    expected_data = [
        'identity_n16_hard.npz',
        'identity_n16_near.npz',
        'identity_n16_smooth.npz',
        'identity_n64_hard.npz',
        'identity_n64_near.npz',
        'identity_n64_smooth.npz',
        'identity_n256_hard.npz',
        'identity_n256_near.npz',
        'identity_n256_smooth.npz'
    ]
    
    # Check tables
    print("\n📊 TABLES:")
    tables_dir = base_dir / 'tables'
    missing_tables = []
    
    for table in expected_tables:
        if (tables_dir / table).exists():
            print(f"  ✅ {table}")
        else:
            print(f"  ❌ {table}")
            missing_tables.append(table)
    
    # Check figures  
    print("\n📈 FIGURES:")
    figures_dir = base_dir / 'figures'
    missing_figures = []
    
    for figure in expected_figures:
        if (figures_dir / figure).exists():
            print(f"  ✅ {figure}")
        else:
            print(f"  ❌ {figure}")
            missing_figures.append(figure)
    
    # Check additional figure formats
    additional_figs = list(figures_dir.glob('*.pdf')) + list(figures_dir.glob('*.png'))
    print(f"  📎 Additional formats: {len(additional_figs)} files")
    
    # Check sample data
    print("\n💾 SAMPLE DATA:")
    samples_dir = base_dir / 'samples'
    missing_data = []
    
    for data_file in expected_data:
        if (samples_dir / data_file).exists():
            print(f"  ✅ {data_file}")
        else:
            print(f"  ❌ {data_file}")
            missing_data.append(data_file)
    
    # Summary statistics
    print("\n📋 SUMMARY:")
    print(f"  Tables: {len(expected_tables) - len(missing_tables)}/{len(expected_tables)} complete")
    print(f"  Figures: {len(expected_figures) - len(missing_figures)}/{len(expected_figures)} complete")
    print(f"  Data files: {len(expected_data) - len(missing_data)}/{len(expected_data)} complete")
    
    # File sizes
    print("\n📦 FILE STATISTICS:")
    
    all_files = list(base_dir.rglob('*'))
    file_count = len([f for f in all_files if f.is_file()])
    
    # Calculate total size
    total_size = sum(f.stat().st_size for f in all_files if f.is_file())
    total_mb = total_size / (1024 * 1024)
    
    print(f"  Total files: {file_count}")
    print(f"  Total size: {total_mb:.1f} MB")
    
    # Breakdown by type
    file_types = {}
    for f in all_files:
        if f.is_file():
            ext = f.suffix.lower()
            file_types[ext] = file_types.get(ext, 0) + 1
    
    print("  File types:")
    for ext, count in sorted(file_types.items()):
        print(f"    {ext or '(no ext)'}: {count} files")
    
    # Overall status
    print("\n🎯 OVERALL STATUS:")
    
    missing_critical = missing_tables + missing_figures + missing_data
    if not missing_critical:
        print("  ✅ ALL MATERIALS COMPLETE AND READY FOR PUBLICATION")
        print("  📝 Ready for manuscript integration")
        print("  📊 Ready for conference/journal submission")
    else:
        print(f"  ⚠️  {len(missing_critical)} files missing")
        print("  🔧 Additional generation needed")
    
    return len(missing_critical) == 0

if __name__ == "__main__":
    success = verify_materials()
    
    if success:
        print("\n🎉 Publication materials verification PASSED!")
    else:
        print("\n⚠️  Some materials need attention")
        
    print(f"\nResults directory: {Path('results').absolute()}")