"""
Clustering history: save and load clustering history.
"""
import numpy as np
from pathlib import Path
from typing import List, Optional, Union
from vecmol.utils.gnf_converter_modules.dataclasses import ClusteringHistory
from vecmol.utils.constants import ELEMENTS_HASH_INV


class ClusteringHistorySaver:
    """Saver for clustering history."""

    def __init__(self, n_atom_types: int = 5):
        """
        Args:
            n_atom_types: Number of atom types.
        """
        self.n_atom_types = n_atom_types

    def save_clustering_history(
        self,
        histories: List[ClusteringHistory],
        output_dir: str,
        batch_idx: Union[int, str] = 0,
        elements: Optional[List[str]] = None
    ) -> None:
        """
        Save clustering history as SDF (one molecule per round) and a detailed text file.

        Args:
            histories: List of clustering histories
            output_dir: Output directory
            batch_idx: Batch index
            elements: Atom type symbols, e.g. ["C", "H", "O", "N", "F"]
        """
        if elements is None:
            elements = [ELEMENTS_HASH_INV.get(i, f"Type{i}") for i in range(self.n_atom_types)]

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if len(histories) > 0:
            if isinstance(batch_idx, int):
                sdf_path = output_path / f"sample_{batch_idx:04d}_clustering_history.sdf"
                txt_path = output_path / f"sample_{batch_idx:04d}_clustering_history.txt"
            else:
                sdf_path = output_path / f"{batch_idx}_clustering_history.sdf"
                txt_path = output_path / f"{batch_idx}_clustering_history.txt"

            self._save_clustering_history_sdf(histories, sdf_path, elements)
            self._save_clustering_history_txt(histories, txt_path, elements)

    def _save_clustering_history_sdf(
        self,
        histories: List[ClusteringHistory],
        output_path: Path,
        elements: List[str]
    ) -> None:
        """Save clustering history as SDF, one molecule per iteration (all types merged)."""
        try:
            from vecmol.sample_diffusion import xyz_to_sdf

            sdf_strings = []
            max_iterations = max(len(h.iterations) for h in histories) if histories else 0

            for iter_idx in range(max_iterations):
                all_coords_this_iter = []
                all_types_this_iter = []
                type_info = []
                
                for history in histories:
                    if iter_idx < len(history.iterations):
                        record = history.iterations[iter_idx]
                        if len(record.new_atoms_coords) > 0:
                            all_coords_this_iter.append(record.new_atoms_coords)
                            all_types_this_iter.append(record.new_atoms_types)
                            
                            atom_symbol = elements[history.atom_type] if history.atom_type < len(elements) else f"Type{history.atom_type}"
                            type_info.append(f"{atom_symbol}:{record.n_atoms_clustered}")

                if len(all_coords_this_iter) > 0:
                    combined_coords = np.vstack(all_coords_this_iter)
                    combined_types = np.concatenate(all_types_this_iter)
                    
                    sdf_str = xyz_to_sdf(combined_coords, combined_types, elements)
                    lines = sdf_str.split('\n')
                    type_info_str = ", ".join(type_info)
                    first_record = next((h.iterations[iter_idx] for h in histories if iter_idx < len(h.iterations)), None)
                    if first_record:
                        title = f"Clustering Iter {iter_idx}, eps={first_record.eps:.4f}, min_samples={first_record.min_samples}, atoms={len(combined_coords)} ({type_info_str})"
                        title = title[:80].ljust(80)
                        lines[0] = title
                    else:
                        title = f"Clustering Iter {iter_idx}, atoms={len(combined_coords)} ({type_info_str})"
                        title = title[:80].ljust(80)
                        lines[0] = title
                    sdf_strings.append('\n'.join(lines))

            all_final_coords = []
            all_final_types = []
            final_type_info = []

            for history in histories:
                type_coords = []
                type_types = []
                for record in history.iterations:
                    if len(record.new_atoms_coords) > 0:
                        type_coords.append(record.new_atoms_coords)
                        type_types.append(record.new_atoms_types)
                
                if len(type_coords) > 0:
                    combined_type_coords = np.vstack(type_coords)
                    combined_type_types = np.concatenate(type_types)
                    all_final_coords.append(combined_type_coords)
                    all_final_types.append(combined_type_types)
                    
                    atom_symbol = elements[history.atom_type] if history.atom_type < len(elements) else f"Type{history.atom_type}"
                    final_type_info.append(f"{atom_symbol}:{len(combined_type_coords)}")

            if len(all_final_coords) > 0:
                final_combined_coords = np.vstack(all_final_coords)
                final_combined_types = np.concatenate(all_final_types)
                final_sdf_str = xyz_to_sdf(final_combined_coords, final_combined_types, elements)
                lines = final_sdf_str.split('\n')
                final_type_info_str = ", ".join(final_type_info)
                title = f"Final Result, Total atoms={len(final_combined_coords)} ({final_type_info_str})"
                title = title[:80].ljust(80)
                lines[0] = title
                sdf_strings.append('\n'.join(lines))

            if sdf_strings:
                with open(output_path, 'w') as f:
                    f.write(''.join(sdf_strings))
        except Exception as e:
            print(f"Warning: Failed to save clustering history SDF: {e}")
    
    def _save_clustering_history_txt(
        self,
        histories: List[ClusteringHistory],
        output_path: Path,
        elements: List[str]
    ) -> None:
        """Save clustering history as text file with details (all types merged)."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("Autoregressive clustering log (all atom types)\n")
                f.write("=" * 80 + "\n\n")

                total_atoms = sum(h.total_atoms for h in histories)
                max_iterations = max(len(h.iterations) for h in histories) if histories else 0
                f.write(f"Total iterations: {max_iterations}\n")
                f.write(f"Final total atoms: {total_atoms}\n")
                f.write("-" * 80 + "\n\n")

                for iter_idx in range(max_iterations):
                    f.write(f"Iteration: {iter_idx}\n")

                    type_records = []
                    for history in histories:
                        if iter_idx < len(history.iterations):
                            record = history.iterations[iter_idx]
                            atom_symbol = elements[history.atom_type] if history.atom_type < len(elements) else f"Type{history.atom_type}"
                            type_records.append((atom_symbol, history.atom_type, record))

                    if type_records:
                        first_record = type_records[0][2]
                        f.write(f"  Threshold: eps={first_record.eps:.4f}, min_samples={first_record.min_samples}\n")

                        total_clusters = sum(r[2].n_clusters_found for r in type_records)
                        total_atoms_clustered = sum(r[2].n_atoms_clustered for r in type_records)
                        total_noise = sum(r[2].n_noise_points for r in type_records)

                        f.write(f"  Total clusters: {total_clusters}\n")
                        f.write(f"  Total clustered atoms: {total_atoms_clustered}\n")
                        f.write(f"  Total noise points: {total_noise}\n")
                        f.write(f"  Bond validation passed: {all(r[2].bond_validation_passed for r in type_records)}\n")
                        f.write("\n")

                        for atom_symbol, atom_type_idx, record in type_records:
                            f.write(f"  Type {atom_symbol}:\n")
                            f.write(f"    Clusters found: {record.n_clusters_found}\n")
                            f.write(f"    Clustered atoms: {record.n_atoms_clustered}\n")
                            f.write(f"    Noise points: {record.n_noise_points}\n")

                            if len(record.new_atoms_coords) > 0:
                                f.write(f"    New clustered atom coords:\n")
                                for i, (coord, atom_type) in enumerate(zip(record.new_atoms_coords, record.new_atoms_types)):
                                    atom_sym = elements[atom_type] if atom_type < len(elements) else f"Type{atom_type}"
                                    f.write(f"      {i+1}. {atom_sym}: ({coord[0]:.4f}, {coord[1]:.4f}, {coord[2]:.4f})\n")
                            else:
                                f.write(f"    No new atoms clustered this round\n")
                            f.write("\n")
                    else:
                        f.write("  No types produced new atoms this round\n")
                    
                    f.write("\n")
                
                f.write("=" * 80 + "\n\n")
        except Exception as e:
            print(f"Warning: Failed to save clustering history text: {e}")

