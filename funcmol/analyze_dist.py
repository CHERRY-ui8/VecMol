#!/usr/bin/env python3
"""
Distance Distribution Analysis Tool
For analyzing dist distribution in encoder and providing cutoff suggestions
"""

import torch
import numpy as np
from models.encoder import analyze_distribution, CrossGraphEncoder
from torch_geometric.data import Data, Batch

def create_sample_data(batch_size=2, n_atoms_per_mol=50):
    """Create sample data for testing"""
    batch_data = []
    
    for b in range(batch_size):
        # Randomly generate atom coordinates (within reasonable range)
        pos = torch.randn(n_atoms_per_mol, 3) * 5.0  # Coordinate range ~±5Å
        # Randomly generate atom types (assuming 5 atom types)
        x = torch.randint(0, 5, (n_atoms_per_mol,))
        # Create batch index
        batch = torch.full((n_atoms_per_mol,), b, dtype=torch.long)
        
        data = Data(pos=pos, x=x, batch=batch)
        batch_data.append(data)
    
    return Batch.from_data_list(batch_data)

def analyze_encoder_distribution():
    """Analyze distance distribution in encoder"""
    print("Creating sample data...")
    data = create_sample_data(batch_size=2, n_atoms_per_mol=100)
    
    print("Initializing encoder...")
    encoder = CrossGraphEncoder(
        n_atom_types=5,
        grid_size=8,
        code_dim=64,
        hidden_dim=128,
        num_layers=2,
        k_neighbors=16,
        atom_k_neighbors=8,
        dist_version='new',
        cutoff=5.0
    )
    
    # Enable debug mode to get dist statistics
    for layer in encoder.layers:
        layer.debug_dist_stats = True
    
    print("Running encoder and collecting dist data...")
    with torch.no_grad():
        # Run encoder
        output = encoder(data)
        
        # Manually calculate dist distribution (for verification)
        print("\nManually calculating dist distribution:")
        atom_coords = data.pos
        grid_coords = encoder.grid_coords.repeat(data.num_graphs, 1)
        
        # Calculate distances from atoms to grid
        from torch_geometric.nn import knn
        grid_to_atom_edges = knn(
            x=atom_coords,
            y=grid_coords,
            k=encoder.k_neighbors,
            batch_x=data.batch,
            batch_y=torch.arange(data.num_graphs).repeat_interleave(encoder.grid_size**3)
        )
        
        # Calculate distances
        row, col = grid_to_atom_edges
        rel = atom_coords[col] - grid_coords[row]
        dist = torch.norm(rel, dim=-1)
        
        # Analyze distribution
        print("Analyzing distance distribution...")
        stats = analyze_distribution(dist, save_plot=True, plot_path="dist_analysis.png")
        
        return stats

if __name__ == "__main__":
    stats = analyze_encoder_distribution()
    print(f"\nAnalysis complete! Results saved to dist_analysis.png") 