import torch
from rdkit import Chem, RDLogger
from rdkit.Geometry import Point3D

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

allowed_bonds = {'H': {0: 1, 1: 0, -1: 0},
                 'C': {0: [3, 4], 1: 3, -1: 3},
                 'N': {0: [2, 3], 1: [2, 3, 4], -1: 2},    # In QM9, N+ seems to be present in the form NH+ and NH2+
                 'O': {0: 2, 1: 3, -1: 1},
                 'F': {0: 1, -1: 0},
                 'B': 3, 'Al': 3, 'Si': 4,
                 'P': {0: [3, 5], 1: 4},
                 'S': {0: [2, 6], 1: [2, 3], 2: 4, 3: 5, -1: 3},
                 'Cl': 1, 'As': 3,
                 'Br': {0: 1, 1: 2}, 'I': 1, 'Hg': [1, 2], 'Bi': [3, 5], 'Se': [2, 4, 6]}
bond_dict = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


class Molecule:
    def __init__(self, atom_types, bond_types, positions, charges, atom_decoder):
        """ atom_types: n      LongTensor
            charges: n         LongTensor
            bond_types: n x n  LongTensor
            positions: n x 3   FloatTensor
            atom_decoder: extracted from dataset_infos. """
        assert atom_types.dim() == 1 and atom_types.dtype == torch.long, f"shape of atoms {atom_types.shape} " \
                                                                         f"and dtype {atom_types.dtype}"
        assert bond_types.dim() == 2 and bond_types.dtype == torch.long, f"shape of bonds {bond_types.shape} --" \
                                                                         f" {bond_types.dtype}"
        assert len(atom_types.shape) == 1
        assert len(bond_types.shape) == 2
        assert len(positions.shape) == 2

        self.atom_types = atom_types.long()
        self.bond_types = bond_types.long()
        self.positions = positions
        self.charges = charges
        self.rdkit_mol = self.build_molecule(atom_decoder)
        self.num_nodes = len(atom_types)
        self.num_atom_types = len(atom_decoder)

    def build_molecule(self, atom_decoder, verbose=False):
        """ If positions is None,
        """
        if verbose:
            print("building new molecule")

        mol = Chem.RWMol()
        for atom, charge in zip(self.atom_types, self.charges):
            if atom == -1:
                continue
            a = Chem.Atom(atom_decoder[int(atom.item())])
            if charge.item() != 0:
                a.SetFormalCharge(charge.item())
            mol.AddAtom(a)
            if verbose:
                print("Atom added: ", atom.item(), atom_decoder[atom.item()])

        edge_types = torch.triu(self.bond_types, diagonal=1)
        edge_types[edge_types == -1] = 0
        all_bonds = torch.nonzero(edge_types)
        for i, bond in enumerate(all_bonds):
            if bond[0].item() != bond[1].item():
                mol.AddBond(bond[0].item(), bond[1].item(), bond_dict[edge_types[bond[0], bond[1]].item()])
                if verbose:
                    print("bond added:", bond[0].item(), bond[1].item(), edge_types[bond[0], bond[1]].item(),
                          bond_dict[edge_types[bond[0], bond[1]].item()])

        try:
            mol = mol.GetMol()
        except Chem.KekulizeException:
            print("Can't kekulize molecule")
            return None

        # Set coordinates
        positions = self.positions.double()
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, Point3D(positions[i][0].item(), positions[i][1].item(), positions[i][2].item()))
        mol.AddConformer(conf)

        return mol


def check_stability(molecule, dataset_info, debug=False, atom_decoder=None, smiles=None, bond_types=None):
    """ 
    molecule: Molecule object.
    bond_types: Optional bond types matrix. If provided, use this instead of molecule.bond_types.
                This allows using different margin values to build bonds and then check stability.
    """
    device = molecule.atom_types.device
    if atom_decoder is None:
        atom_decoder = dataset_info.atom_decoder

    atom_types = molecule.atom_types
    if bond_types is not None:
        edge_types = bond_types.clone()
    else:
        edge_types = molecule.bond_types.clone()

    edge_types[edge_types == 4] = 1.5
    edge_types[edge_types < 0] = 0

    valencies = torch.sum(edge_types, dim=-1).long()

    n_stable_bonds = 0
    mol_stable = True
    # #region agent log
    import json; log_data = {"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "rdkit_functions.py:111", "message": "check_stability entry", "data": {"num_atoms": len(atom_types), "mol_stable_initial": mol_stable}, "timestamp": int(__import__("time").time() * 1000)}; open("/datapool/data2/home/pxg/data/hyc/funcmol-main-neuralfield/.cursor/debug.log", "a").write(json.dumps(log_data) + "\n")
    # #endregion
    for i, (atom_type, valency, charge) in enumerate(zip(atom_types, valencies, molecule.charges)):
        atom_type = atom_type.item()
        valency = valency.item()
        charge = charge.item()
        possible_bonds = allowed_bonds[atom_decoder[atom_type]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == valency
        elif type(possible_bonds) == dict:
            expected_bonds = possible_bonds[charge] if charge in possible_bonds.keys() else possible_bonds[0]
            is_stable = expected_bonds == valency if type(expected_bonds) == int else valency in expected_bonds
        else:
            is_stable = valency in possible_bonds
        # #region agent log
        log_data = {"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "rdkit_functions.py:124", "message": "atom stability check", "data": {"atom_idx": i, "atom_type": atom_decoder[atom_type], "valency": valency, "charge": charge, "is_stable": is_stable, "mol_stable_before": mol_stable}, "timestamp": int(__import__("time").time() * 1000)}; open("/datapool/data2/home/pxg/data/hyc/funcmol-main-neuralfield/.cursor/debug.log", "a").write(json.dumps(log_data) + "\n")
        # #endregion
        if not is_stable:
            mol_stable = False
            # #region agent log
            log_data = {"sessionId": "debug-session", "runId": "run1", "hypothesisId": "B", "location": "rdkit_functions.py:126", "message": "mol_stable set to False", "data": {"atom_idx": i, "mol_stable_after": mol_stable}, "timestamp": int(__import__("time").time() * 1000)}; open("/datapool/data2/home/pxg/data/hyc/funcmol-main-neuralfield/.cursor/debug.log", "a").write(json.dumps(log_data) + "\n")
            # #endregion
        if not is_stable and debug:
            if smiles is not None:
                print(smiles)
            print(f"Invalid atom {atom_decoder[atom_type]}: valency={valency}, charge={charge}")
            print()
        n_stable_bonds += int(is_stable)
    # #region agent log
    log_data = {"sessionId": "debug-session", "runId": "run1", "hypothesisId": "A", "location": "rdkit_functions.py:133", "message": "check_stability before return", "data": {"mol_stable": mol_stable, "mol_stable_type": str(type(mol_stable)), "n_stable_bonds": n_stable_bonds}, "timestamp": int(__import__("time").time() * 1000)}; open("/datapool/data2/home/pxg/data/hyc/funcmol-main-neuralfield/.cursor/debug.log", "a").write(json.dumps(log_data) + "\n")
    # #endregion
    result_tensor = torch.tensor([mol_stable], dtype=torch.float, device=device)
    # #region agent log
    log_data = {"sessionId": "debug-session", "runId": "run1", "hypothesisId": "E", "location": "rdkit_functions.py:136", "message": "tensor created", "data": {"mol_stable_bool": mol_stable, "tensor_value": result_tensor.item(), "tensor_dtype": str(result_tensor.dtype)}, "timestamp": int(__import__("time").time() * 1000)}; open("/datapool/data2/home/pxg/data/hyc/funcmol-main-neuralfield/.cursor/debug.log", "a").write(json.dumps(log_data) + "\n")
    # #endregion
    return result_tensor,\
           torch.tensor([n_stable_bonds], dtype=torch.float, device=device),\
           len(atom_types)

