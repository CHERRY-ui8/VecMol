"""
Generate quality analysis report table for denoiser-generated molecules
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_analysis_results():
    """
    Load all analysis results
    """
    base_dir = project_root / "funcmol" / "analysis"
    
    # Extract data from previous analysis results
    results = {
        # Basic statistics
        'total_molecules': 537,
        'avg_atoms': 15.25,
        'atom_count_range': '7 - 22',
        
        # Validity metrics
        'validity': 18.81,  # %
        'uniqueness': 8.91,  # %
        'novelty': 0.00,  # %
        
        # Connectivity metrics
        'connected_ratio': 0.19,  # %
        'avg_components': 8.42,
        'max_components': 15.00,
        
        # Stability metrics
        'mol_stability': 0.00,  # %
        'atom_stability': 25.94,  # %
        
        # Distance statistics
        'avg_min_distance': 1.14,  # Å
        'median_min_distance': 1.08,  # Å
        'min_distance_range': '0.04 - 7.04',  # Å
        'large_distance_ratio': 0.16,  # %
        
        # Bond length statistics
        'total_bonds': 4144,
        'avg_bond_length': 1.09,  # Å
        'bond_length_range': '0.04 - 1.64',  # Å
        'normal_bond_ratio': 97.22,  # %
        
        # Spatial distribution
        'avg_coord_range': '4.55, 4.41, 4.52',  # Å
        'max_coord_range': 8.80,  # Å
        'avg_mol_span': 5.82,  # Å
        'max_mol_span': 13.01,  # Å
        'large_pair_ratio': 54.50,  # %
        
        # Error statistics
        'atomvalence_errors': 436,
        'valid_smiles': 101,
    }
    
    return results


def create_quality_table(results, output_dir):
    """
    Create quality analysis tables
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create categorized table data
    tables = {}
    
    # 1. Basic statistics
    tables['Basic Statistics'] = pd.DataFrame([
        ['Total Molecules', f"{results['total_molecules']}", 'molecules'],
        ['Average Atoms', f"{results['avg_atoms']:.2f}", 'atoms'],
        ['Atom Count Range', results['atom_count_range'], 'atoms'],
    ], columns=['Metric', 'Value', 'Unit'])
    
    # 2. Validity metrics
    tables['Validity Metrics'] = pd.DataFrame([
        ['Validity', f"{results['validity']:.2f}%", 'Proportion of molecules with valid SMILES'],
        ['Uniqueness', f"{results['uniqueness']:.2f}%", 'Proportion of unique valid molecules'],
        ['Novelty', f"{results['novelty']:.2f}%", 'Proportion not in training set'],
    ], columns=['Metric', 'Value', 'Description'])
    
    # 3. Connectivity metrics
    tables['Connectivity Metrics'] = pd.DataFrame([
        ['Connected Ratio', f"{results['connected_ratio']:.2f}%", 'Proportion of molecules forming complete graph'],
        ['Avg Components', f"{results['avg_components']:.2f}", 'Average number of fragments per molecule'],
        ['Max Components', f"{int(results['max_components'])}", 'Maximum number of fragments in a molecule'],
    ], columns=['Metric', 'Value', 'Description'])
    
    # 4. Stability metrics
    tables['Stability Metrics'] = pd.DataFrame([
        ['Molecular Stability', f"{results['mol_stability']:.2f}%", 'Proportion of fully stable molecules'],
        ['Atomic Stability', f"{results['atom_stability']:.2f}%", 'Proportion of atoms following valence rules'],
    ], columns=['Metric', 'Value', 'Description'])
    
    # 5. Distance statistics
    tables['Distance Statistics'] = pd.DataFrame([
        ['Avg Min Distance', f"{results['avg_min_distance']:.2f} Å", 'Average distance to nearest neighbor'],
        ['Median Min Distance', f"{results['median_min_distance']:.2f} Å", 'Median of minimum distances'],
        ['Min Distance Range', results['min_distance_range'] + ' Å', 'Range of minimum distances'],
        ['Large Distance Ratio (>3Å)', f"{results['large_distance_ratio']:.2f}%", 'Proportion of atoms with large distances'],
    ], columns=['Metric', 'Value', 'Description'])
    
    # 6. Bond length statistics
    tables['Bond Length Statistics'] = pd.DataFrame([
        ['Total Bonds', f"{results['total_bonds']}", 'Total number of detected bonds'],
        ['Avg Bond Length', f"{results['avg_bond_length']:.2f} Å", 'Average length of all bonds'],
        ['Bond Length Range', results['bond_length_range'] + ' Å', 'Range of bond lengths'],
        ['Normal Bond Ratio (0.7-2.0Å)', f"{results['normal_bond_ratio']:.2f}%", 'Proportion of bonds in reasonable range'],
    ], columns=['Metric', 'Value', 'Description'])
    
    # 7. Spatial distribution
    tables['Spatial Distribution'] = pd.DataFrame([
        ['Avg Coord Range', f"({results['avg_coord_range']}) Å", 'Average range along X, Y, Z axes'],
        ['Max Coord Range', f"{results['max_coord_range']:.2f} Å", 'Maximum coordinate range in a molecule'],
        ['Avg Molecular Span', f"{results['avg_mol_span']:.2f} Å", 'Average maximum atom-pair distance'],
        ['Max Molecular Span', f"{results['max_mol_span']:.2f} Å", 'Maximum span in a molecule'],
        ['Large Pair Ratio (>3Å)', f"{results['large_pair_ratio']:.2f}%", 'Proportion of atom pairs too far to bond'],
    ], columns=['Metric', 'Value', 'Description'])
    
    # 8. Error statistics
    tables['Error Statistics'] = pd.DataFrame([
        ['AtomValence Errors', f"{results['atomvalence_errors']}", 'Number of molecules with valence errors'],
        ['Valid SMILES', f"{results['valid_smiles']}", 'Number of molecules with valid SMILES'],
        ['Error Rate', f"{results['atomvalence_errors']/results['total_molecules']*100:.2f}%", 'Proportion of AtomValence errors'],
    ], columns=['Metric', 'Value', 'Description'])
    
    # Save all tables
    output_file = output_dir / "molecule_quality_report.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Denoiser-Generated Molecule Quality Analysis Report\n\n")
        f.write(f"**Analysis Time**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Molecules**: {results['total_molecules']}\n\n")
        
        for table_name, df in tables.items():
            f.write(f"## {table_name}\n\n")
            # Manually generate markdown table
            headers = '| ' + ' | '.join(df.columns) + ' |'
            f.write(headers + '\n')
            f.write('|' + '|'.join(['---'] * len(df.columns)) + '|\n')
            for _, row in df.iterrows():
                f.write('| ' + ' | '.join([str(v) for v in row.values]) + ' |\n')
            f.write("\n\n")
    
    # Save as CSV format (summary table)
    summary_data = []
    for table_name, df in tables.items():
        for _, row in df.iterrows():
            summary_data.append({
                'Category': table_name,
                'Metric': row['Metric'],
                'Value': row['Value'],
                'Description': row.get('Description', '')
            })
    
    summary_df = pd.DataFrame(summary_data)
    csv_file = output_dir / "molecule_quality_summary.csv"
    summary_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    
    # Create visualization tables
    create_visualization_table(tables, output_dir)
    
    print(f"\nReport saved to:")
    print(f"  - Markdown: {output_file}")
    print(f"  - CSV: {csv_file}")
    
    return tables, summary_df


def create_visualization_table(tables, output_dir):
    """
    Create visualization tables
    """
    # Create key metrics comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Denoiser-Generated Molecule Quality Analysis - Key Metrics', fontsize=16, fontweight='bold')
    
    # 1. Validity metrics
    ax1 = axes[0, 0]
    validity_data = tables['Validity Metrics']
    metrics = validity_data['Metric'].values
    values = [float(v.replace('%', '')) for v in validity_data['Value'].values]
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    bars = ax1.barh(metrics, values, color=colors)
    ax1.set_xlabel('Percentage (%)', fontsize=10)
    ax1.set_title('Validity Metrics', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, max(values) * 1.2)
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax1.text(val + max(values)*0.02, i, f'{val:.2f}%', 
                va='center', fontsize=9)
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Connectivity metrics
    ax2 = axes[0, 1]
    connectivity_data = tables['Connectivity Metrics']
    metrics = connectivity_data['Metric'].values
    values = [float(v.replace('%', '')) if '%' in v else float(v) 
              for v in connectivity_data['Value'].values]
    colors = ['#FF6B6B', '#FFA07A', '#FFD700']
    bars = ax2.barh(metrics, values, color=colors)
    ax2.set_xlabel('Value', fontsize=10)
    ax2.set_title('Connectivity Metrics', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, max(values) * 1.2)
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax2.text(val + max(values)*0.02, i, f'{val:.2f}', 
                va='center', fontsize=9)
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Stability metrics
    ax3 = axes[1, 0]
    stability_data = tables['Stability Metrics']
    metrics = stability_data['Metric'].values
    values = [float(v.replace('%', '')) for v in stability_data['Value'].values]
    colors = ['#FF6B6B', '#4ECDC4']
    bars = ax3.barh(metrics, values, color=colors)
    ax3.set_xlabel('Percentage (%)', fontsize=10)
    ax3.set_title('Stability Metrics', fontsize=12, fontweight='bold')
    ax3.set_xlim(0, max(values) * 1.2 if max(values) > 0 else 30)
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax3.text(val + (max(values)*0.02 if max(values) > 0 else 1), i, f'{val:.2f}%', 
                va='center', fontsize=9)
    ax3.grid(axis='x', alpha=0.3)
    
    # 4. Spatial distribution metrics
    ax4 = axes[1, 1]
    spatial_data = tables['Spatial Distribution']
    # Select key metrics
    key_metrics = ['Avg Molecular Span', 'Max Molecular Span', 'Large Pair Ratio (>3Å)']
    key_values = []
    for metric in key_metrics:
        row = spatial_data[spatial_data['Metric'] == metric]
        if not row.empty:
            val_str = row['Value'].values[0]
            if 'Å' in val_str:
                val = float(val_str.replace(' Å', ''))
            elif '%' in val_str:
                val = float(val_str.replace('%', ''))
            else:
                val = float(val_str)
            key_values.append(val)
    
    colors = ['#FF6B6B', '#FFA07A', '#FFD700']
    bars = ax4.barh(key_metrics, key_values, color=colors)
    ax4.set_xlabel('Value', fontsize=10)
    ax4.set_title('Spatial Distribution Key Metrics', fontsize=12, fontweight='bold')
    ax4.set_xlim(0, max(key_values) * 1.2)
    for i, (bar, val) in enumerate(zip(bars, key_values)):
        unit = 'Å' if i < 2 else '%'
        ax4.text(val + max(key_values)*0.02, i, f'{val:.2f}{unit}', 
                va='center', fontsize=9)
    ax4.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / "molecule_quality_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  - Visualization: {output_file}")
    plt.close()
    
    # Create comprehensive comparison table
    create_comprehensive_table(tables, output_dir)


def create_comprehensive_table(tables, output_dir):
    """
    Create comprehensive comparison table
    """
    # Extract key metrics
    key_metrics = {
        'Validity': ['Validity', 'Uniqueness', 'Novelty'],
        'Connectivity': ['Connected Ratio', 'Avg Components'],
        'Stability': ['Molecular Stability', 'Atomic Stability'],
        'Spatial': ['Avg Molecular Span', 'Large Pair Ratio (>3Å)'],
    }
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Category', 'Metric', 'Value', 'Assessment']
    
    for category, metrics in key_metrics.items():
        for metric in metrics:
            # Find corresponding data
            value = None
            for table_name, df in tables.items():
                row = df[df['Metric'] == metric]
                if not row.empty:
                    value = row['Value'].values[0]
                    break
            
            if value:
                # Assessment status
                if 'Validity' in category or 'Stability' in category:
                    val_num = float(value.replace('%', ''))
                    if val_num > 50:
                        status = 'Good'
                    elif val_num > 20:
                        status = 'Fair'
                    else:
                        status = 'Poor'
                elif 'Connectivity' in category:
                    if 'Ratio' in metric:
                        val_num = float(value.replace('%', ''))
                        if val_num > 50:
                            status = 'Good'
                        elif val_num > 20:
                            status = 'Fair'
                        else:
                            status = 'Poor'
                    else:
                        val_num = float(value)
                        if val_num < 2:
                            status = 'Good'
                        elif val_num < 5:
                            status = 'Fair'
                        else:
                            status = 'Poor'
                else:  # Spatial
                    val_num = float(value.replace('Å', '').replace('%', ''))
                    if 'Span' in metric:
                        if val_num < 3:
                            status = 'Good'
                        elif val_num < 6:
                            status = 'Fair'
                        else:
                            status = 'Poor'
                    else:
                        if val_num < 20:
                            status = 'Good'
                        elif val_num < 40:
                            status = 'Fair'
                        else:
                            status = 'Poor'
                
                table_data.append([category, metric, value, status])
    
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.35, 0.2, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Set header style
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Set assessment column color
    for i, row in enumerate(table_data, 1):
        status = row[3]
        if 'Good' in status:
            table[(i, 3)].set_facecolor('#C8E6C9')
        elif 'Fair' in status:
            table[(i, 3)].set_facecolor('#FFF9C4')
        else:
            table[(i, 3)].set_facecolor('#FFCDD2')
    
    plt.title('Denoiser-Generated Molecule Quality Analysis - Comprehensive Assessment', 
             fontsize=14, fontweight='bold', pad=20)
    
    output_file = output_dir / "molecule_quality_comprehensive_table.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  - Comprehensive Table: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate molecule quality analysis report')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (optional)'
    )
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        output_dir = project_root / "funcmol" / "analysis" / "quality_report"
    else:
        output_dir = Path(args.output_dir)
    
    print("="*60)
    print("Generating Denoiser Molecule Quality Analysis Report")
    print("="*60)
    
    # Load analysis results
    results = load_analysis_results()
    
    # Create tables
    tables, summary_df = create_quality_table(results, output_dir)
    
    # Print key metrics
    print("\n" + "="*60)
    print("Key Metrics Summary")
    print("="*60)
    print(f"Validity: {results['validity']:.2f}%")
    print(f"Connected Ratio: {results['connected_ratio']:.2f}%")
    print(f"Avg Components: {results['avg_components']:.2f}")
    print(f"Molecular Stability: {results['mol_stability']:.2f}%")
    print(f"Avg Molecular Span: {results['avg_mol_span']:.2f} Å")
    print(f"Large Pair Ratio: {results['large_pair_ratio']:.2f}%")
    
    print("\n" + "="*60)
    print("Report Generation Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
