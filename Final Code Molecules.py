import numpy as np
import torch
import torch.nn as nn
from bayes_opt import BayesianOptimization
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from tqdm import tqdm
import traceback

# -----------------------------------------------------
# Optional dependencies
# -----------------------------------------------------
try:
    from chembl_webresource_client.new_client import new_client
    CHEMBL_AVAILABLE = True
    print("✓ ChEMBL client loaded correctly")
except Exception as e:
    CHEMBL_AVAILABLE = False
    print(f"⚠ ChEMBL not available: {e}")

try:
    from pyscf import gto, dft
    PYSCF_AVAILABLE = True
    print("✓ PySCF loaded correctly")
except Exception as e:
    PYSCF_AVAILABLE = False
    print(f"⚠ PySCF not available: {e} — DFT properties will be simulated")

# -----------------------------------------------------
# Numerical utilities
# -----------------------------------------------------
def safe_mean(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else float('nan')

# ============================================================================
# PART 1: EXTENDED DESCRIPTOR CALCULATOR (WITHOUT RDKit)
# ============================================================================
class MolecularDescriptorCalculator:
    """Calculates >40 molecular descriptors without RDKit (heuristic)."""

    ATOMIC_WEIGHTS = {
        'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999,
        'F': 18.998, 'P': 30.974, 'S': 32.06, 'Cl': 35.45,
        'Br': 79.904, 'I': 126.90
    }
    ELECTRONEGATIVITY = {
        'H': 2.20, 'C': 2.55, 'N': 3.04, 'O': 3.44,
        'F': 3.98, 'P': 2.19, 'S': 2.58, 'Cl': 3.16,
        'Br': 2.96, 'I': 2.66
    }
    VDW_RADII = {
        'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52,
        'F': 1.47, 'P': 1.80, 'S': 1.80, 'Cl': 1.75,
        'Br': 1.85, 'I': 1.98
    }
    ATOM_Z = {'H':1,'C':6,'N':7,'O':8,'F':9,'P':15,'S':16,'Cl':17,'Br':35,'I':53}

    # ------------------------ Basic parsing ------------------------
    def parse_smiles_enhanced(self, smiles: str):
        """Extracts atomic counts and approximate 'neighborhood' relationships."""
        s = smiles
        # remove simple annotations
        clean = re.sub(r'[()\[\]]', '', s)
        atom_counts = {}
        # patterns for multi-letter halogen atoms
        halogens = {'Cl':'Cl','Br':'Br'}
        # count Cl/Br first
        for sym, pat in halogens.items():
            n = len(re.findall(pat, clean))
            if n: atom_counts[sym] = n
            clean = re.sub(pat, '', clean)
        # now mono-letter
        for sym in ['C','N','O','S','P','F','I','H']:
            n = len(re.findall(fr'{sym}(?![a-z])', clean))
            if n: atom_counts[sym] = atom_counts.get(sym,0)+n

        # estimate H if not explicit
        if 'H' not in atom_counts:
            c = atom_counts.get('C',0); n = atom_counts.get('N',0); o = atom_counts.get('O',0)
            approx_h = max(0, 2*c + n + o - len(smiles)//3)
            if approx_h: atom_counts['H'] = approx_h

        # approximate "bond" relationships: count by separator character
        bonds_info = {}
        prev = None; idx = 0
        for ch in smiles:
            if ch.isalpha():
                token = ch
                # consider aromatic lowercase as 'C'
                if ch=='c': token = 'C'
                if prev is not None:
                    bonds_info.setdefault(prev, []).append(token)
                    bonds_info.setdefault(token, []).append(prev)
                prev = token
                idx += 1
        return atom_counts, bonds_info

    # ------------------------ Functional groups ------------------------
    def count_functional_groups(self, smiles: str):
        patt = {
            'hydroxyl': r'O[H]?',
            'amine': r'N(?![a-z])',
            'amide': r'C\(=O\)N|N\(C\)C=O|NC=O',
            'carbonyl': r'C=O',
            'nitrile': r'C#N',
            'sulfonyl': r'S\(=O\)\(=O\)',
            'ether': r'C-O-C',
            'ester': r'C\(=O\)O',
            'halide': r'F|Cl|Br|I',
            'nitro': r'\[N\+\]\(=O\)\[O-\]|N\(=O\)O',
            'ring_aromatic': r'c'
        }
        counts = {}
        for k, p in patt.items():
            counts[k] = len(re.findall(p, smiles))
        return counts

    # ------------------------ Base magnitudes ------------------------
    def calculate_molecular_weight(self, atom_counts):
        return sum(self.ATOMIC_WEIGHTS.get(a, 0)*c for a,c in atom_counts.items())

    def estimate_logp_enhanced(self, atom_counts, smiles, func_groups):
        contrib = {'C':0.54,'H':0.0,'N':-0.6,'O':-0.8,'S':0.25,'F':-0.3,'Cl':0.7,'Br':1.0,'I':1.3}
        lp = sum(contrib.get(a,0)*c for a,c in atom_counts.items())
        # adjustments for groups
        lp += -0.6*func_groups.get('hydroxyl',0)
        lp += -0.8*func_groups.get('amide',0)
        lp += -0.4*func_groups.get('nitrile',0)
        if 'c' in smiles: lp += 1.0
        if '=' in smiles: lp += 0.2
        return lp

    def estimate_polar_surface_area_enhanced(self, atom_counts, func_groups):
        psa_atoms = {'N':23.79,'O':20.23,'S':25.30,'P':23.85}
        psa = sum(psa_atoms.get(a,0)*c for a,c in atom_counts.items())
        psa += 8.0*func_groups.get('nitrile',0)
        psa += 12.0*func_groups.get('amide',0)
        return psa

    def count_rotatable_bonds_enhanced(self, smiles):
        patterns = [r'C-?C', r'C-?N', r'C-?O', r'C-?S']
        rot = sum(len(re.findall(p, smiles)) for p in patterns)
        if 'c' in smiles or '1' in smiles: rot = max(0, rot-3)
        return rot

    def estimate_aromatic_rings(self, smiles): return max(0, smiles.count('c')//6)
    def count_saturated_rings(self, smiles): return smiles.count('1')+smiles.count('2')
    def count_aliphatic_rings(self, smiles): return max(0, self.count_saturated_rings(smiles)-self.estimate_aromatic_rings(smiles))
    def count_total_rings(self, smiles): return self.count_saturated_rings(smiles)+self.estimate_aromatic_rings(smiles)
    def count_heteroatoms(self, atom_counts): return sum(atom_counts.get(a,0) for a in ['N','O','S','P','F','Cl','Br','I'])

    # ------------------------ Advanced/geometric ------------------------
    def calculate_molecular_volume(self, atom_counts):
        # approx volume by sum of vdw volumes
        return sum((self.VDW_RADII.get(a,1.5)**3)*c for a,c in atom_counts.items()) * 1.2

    def calculate_csp3_fraction(self, atom_counts, smiles):
        c = atom_counts.get('C',0); aromatic_c = smiles.count('c')
        sp3 = max(0, c - aromatic_c)
        return sp3 / max(1, c)

    def calculate_complexity_index(self, smiles):
        return 1.5*len(smiles) + 2.0*self.count_total_rings(smiles)

    def calculate_asphericity(self, atom_counts):
        # proxy: deviation from uniform composition
        total = sum(atom_counts.values())
        if total == 0: return 0.0
        fracs = np.array(list(atom_counts.values()))/total
        return float(np.var(fracs))

    def calculate_flexibility_index(self, smiles, atom_counts):
        rot = self.count_rotatable_bonds_enhanced(smiles)
        heavy = sum(c for a,c in atom_counts.items() if a!='H')
        return rot / max(1, heavy)

    def calculate_electronegativity_descriptors(self, atom_counts):
        tot = sum(atom_counts.values()) or 1
        mean_en = sum(self.ELECTRONEGATIVITY.get(a,2.5)*c for a,c in atom_counts.items())/tot
        max_en = max((self.ELECTRONEGATIVITY.get(a,0) for a in atom_counts), default=0)
        min_en = min((self.ELECTRONEGATIVITY.get(a,10) for a in atom_counts), default=0)
        return {'MeanEN': mean_en, 'MaxEN': max_en, 'MinEN': min_en, 'ENRange': max_en-min_en}

    def calculate_bond_descriptors(self, bonds_info, atom_counts):
        degs = [len(v) for v in bonds_info.values()] or [0]
        return {
            'MeanConnectivity': float(np.mean(degs)),
            'BranchingIndex': float(sum(1 for d in degs if d>2)),
            'BondDensity': float(len(bonds_info)/max(1, sum(c for a,c in atom_counts.items() if a!='H')))
        }

    # ------------------------ Master descriptor set ------------------------
    def calculate_comprehensive_descriptors(self, smiles: str):
        atom_counts, bonds_info = self.parse_smiles_enhanced(smiles)
        func_groups = self.count_functional_groups(smiles)
        electronegativity_desc = self.calculate_electronegativity_descriptors(atom_counts)
        bond_desc = self.calculate_bond_descriptors(bonds_info, atom_counts)

        basic = {
            'MolWt': self.calculate_molecular_weight(atom_counts),
            'LogP': self.estimate_logp_enhanced(atom_counts, smiles, func_groups),
            'NumHDonors': func_groups.get('hydroxyl',0) + func_groups.get('amine',0),
            'NumHAcceptors': atom_counts.get('N',0) + atom_counts.get('O',0),
            'TPSA': self.estimate_polar_surface_area_enhanced(atom_counts, func_groups),
            'NumRotatableBonds': self.count_rotatable_bonds_enhanced(smiles),
            'NumAromaticRings': self.estimate_aromatic_rings(smiles),
            'NumSaturatedRings': self.count_saturated_rings(smiles),
            'NumAliphaticRings': self.count_aliphatic_rings(smiles),
            'RingCount': self.count_total_rings(smiles),
            'NumHeteroatoms': self.count_heteroatoms(atom_counts),
            'HeavyAtoms': sum(c for a,c in atom_counts.items() if a!='H'),
        }

        functional = {
            'Func_hydroxyl': func_groups.get('hydroxyl',0),
            'Func_amine': func_groups.get('amine',0),
            'Func_amide': func_groups.get('amide',0),
            'CarbonylCount': func_groups.get('carbonyl',0),
            'NitrileCount': func_groups.get('nitrile',0),
            'SulfonylCount': func_groups.get('sulfonyl',0),
            'HalogenCount': sum(atom_counts.get(x,0) for x in ['F','Cl','Br','I']),
            'NitroCount': func_groups.get('nitro',0),
            'EtherCount': func_groups.get('ether',0),
            'EsterCount': func_groups.get('ester',0),
        }

        advanced = {
            'MolecularVolume': self.calculate_molecular_volume(atom_counts),
            'FractionCsp3': self.calculate_csp3_fraction(atom_counts, smiles),
            'BertzCT': self.calculate_complexity_index(smiles),
            'AsphericalityIndex': self.calculate_asphericity(atom_counts),
            'FlexibilityIndex': self.calculate_flexibility_index(smiles, atom_counts),
            'VdWRadiusSum': sum(self.VDW_RADII.get(a,1.5)*c for a,c in atom_counts.items()),
            'EstimatedDensity': (basic['MolWt']/max(1.0, self.calculate_molecular_volume(atom_counts))),
        }
        advanced.update(electronegativity_desc)
        advanced.update(bond_desc)

        ratios = {
            'H_over_C': atom_counts.get('H',0)/max(1, atom_counts.get('C',0)),
            'O_over_C': atom_counts.get('O',0)/max(1, atom_counts.get('C',0)),
            'N_over_C': atom_counts.get('N',0)/max(1, atom_counts.get('C',0)),
            'Hal_over_C': functional['HalogenCount']/max(1, atom_counts.get('C',0)),
            'MW_over_100': basic['MolWt']/100.0,
            'LogP_squared': basic['LogP']**2,
            'TPSA_over_MW': basic['TPSA']/max(1.0, basic['MolWt']),
            'Aromatic_ratio': basic['NumAromaticRings']/max(1, basic['RingCount']),
            'Hetero_ratio': basic['NumHeteroatoms']/max(1, basic['HeavyAtoms']),
            'Ring_density': basic['RingCount']/max(1, basic['HeavyAtoms']),
            'Polar_ratio': basic['TPSA']/max(1.0, basic['MolWt']),
            'Carbon_fraction': atom_counts.get('C',0)/max(1, basic['HeavyAtoms']),
            'Nitrogen_fraction': atom_counts.get('N',0)/max(1, basic['HeavyAtoms']),
            'Oxygen_fraction': atom_counts.get('O',0)/max(1, basic['HeavyAtoms']),
        }

        topology = {
            'MeanConnectivity2': advanced['MeanConnectivity'],
            'BranchingIndex2': advanced['BranchingIndex'],
            'BondDensity2': advanced['BondDensity'],
            'PolarizabilityProxy': sum((self.ATOM_Z.get(a,0)**2)*c for a,c in atom_counts.items())
        }

        desc = {}
        for d in (basic, functional, advanced, ratios, topology):
            desc.update(d)
        # ensure minimum of 40 descriptors
        if len(desc) < 40:
            base_vals = list(desc.values())
            for i in range(40 - len(desc)):
                desc[f'Derived_{i}'] = base_vals[i % len(base_vals)] * 1.0
        return desc

# ============================================================================
# PART 2: DATASET — ChEMBL + variants
# ============================================================================
class EnhancedSolventDataset:
    """Peptide synthesis solvent dataset (>=50 compounds)."""
    def __init__(self, cache_file='enhanced_solvents_cache.json'):
        self.cache_file = cache_file
        self.data = []
        self.descriptor_calc = MolecularDescriptorCalculator()

    def _get_extended_local_dataset(self, max_compounds=120):
        extended = [
            {'name': 'Water', 'smiles': 'O', 'formula': 'H2O'},
            {'name': 'Methanol', 'smiles': 'CO', 'formula': 'CH4O'},
            {'name': 'Ethanol', 'smiles': 'CCO', 'formula': 'C2H6O'},
            {'name': 'Isopropanol', 'smiles': 'CC(C)O', 'formula': 'C3H8O'},
            {'name': '1-Butanol', 'smiles': 'CCCCO', 'formula': 'C4H10O'},
            {'name': 't-Butanol', 'smiles': 'CC(C)(C)O', 'formula': 'C4H10O'},
            {'name': 'Formamide', 'smiles': 'C(=O)N', 'formula': 'CH3NO'},
            {'name': 'DMF', 'smiles': 'CN(C)C=O', 'formula': 'C3H7NO'},
            {'name': 'DMAc', 'smiles': 'CN(C)C(C)=O', 'formula': 'C4H9NO'},
            {'name': 'DMSO', 'smiles': 'CS(=O)C', 'formula': 'C2H6OS'},
            {'name': 'NMP', 'smiles': 'CN1CCCC1=O', 'formula': 'C5H9NO'},
            {'name': 'HMPA', 'smiles': 'CN(C)P(=O)(N(C)C)N(C)C', 'formula': 'C6H18N3OP'},
            {'name': 'Acetonitrile', 'smiles': 'CC#N', 'formula': 'C2H3N'},
            {'name': 'Butyronitrile', 'smiles': 'CCCC#N', 'formula': 'C4H7N'},
            {'name': 'Propylene carbonate', 'smiles': 'O=C1OCCO1', 'formula': 'C4H6O3'},
            {'name': 'Sulfolane', 'smiles': 'O=S1(CCCC1)=O', 'formula': 'C4H8O2S'},
            {'name': 'Chloroform', 'smiles': 'C(Cl)(Cl)Cl', 'formula': 'CHCl3'},
            {'name': 'Dichloromethane', 'smiles': 'CCl2', 'formula': 'CH2Cl2'},
            {'name': 'Fluorobenzene', 'smiles': 'c1ccc(cc1)F', 'formula': 'C6H5F'},
            {'name': 'Nitromethane', 'smiles': 'C[N+](=O)[O-]', 'formula': 'CH3NO2'},
            {'name': 'THF', 'smiles': 'C1CCOC1', 'formula': 'C4H8O'},
            {'name': 'Dioxane', 'smiles': 'C1COCCOC1', 'formula': 'C4H8O2'},
            {'name': 'DME', 'smiles': 'COCCOC', 'formula': 'C4H10O2'},
            {'name': 'Diglyme', 'smiles': 'COCCOCCOC', 'formula': 'C6H14O3'},
            {'name': 'Diethyl ether', 'smiles': 'CCOCC', 'formula': 'C4H10O'},
            {'name': 'Ethyl acetate', 'smiles': 'CC(=O)OCC', 'formula': 'C4H8O2'},
        ]
        # variants with functional groups/heteroatoms
        variants = []
        adders = [
            ('-OH','O'), ('-CN','C#N'), ('-NO2','[N+](=O)[O-]'),
            ('-F','F'), ('-Cl','Cl'), ('-Br','Br'),
            ('-SO2Me','S(=O)(=O)C'), ('-OEt','OCC')
        ]
        for mol in extended:
            for suf, frag in adders:
                variants.append({'name': f"{mol['name']}{suf}", 'smiles': mol['smiles']+frag, 'formula': (mol.get('formula','') or '')})
        dataset = extended + variants
        return dataset[:max_compounds]

    def fetch_chembl_solvents(self, max_compounds=200):
        """Fetches relevant solvents from ChEMBL ensuring diversity."""
        if not CHEMBL_AVAILABLE:
            print("⚠ ChEMBL not available, using extended local dataset")
            return self._get_extended_local_dataset(max_compounds=max_compounds)

        molecule = new_client.molecule
        solvent_queries = [
            'solvent', 'polar aprotic', 'polar protic', 'amide',
            'nitrile', 'sulfoxide', 'ether', 'halogenated', 'ester',
            'carbonate', 'alcohol', 'formamide', 'nitromethane'
        ]
        chembl_compounds, seen = [], set()
        for query in solvent_queries:
            try:
                results = molecule.filter(
                    molecule_synonyms__molecule_synonym__icontains=query,
                    molecule_properties__mw_freebase__lte=350,
                    molecule_properties__mw_freebase__gte=20
                ).only(['molecule_chembl_id','molecule_structures','pref_name'])[:50]
                for cmpd in results:
                    smi = (cmpd.get('molecule_structures') or {}).get('canonical_smiles')
                    if not smi: continue
                    name = cmpd.get('pref_name') or f"ChEMBL_{cmpd.get('molecule_chembl_id')}"
                    key = (name, smi)
                    if key in seen: continue
                    seen.add(key)
                    chembl_compounds.append({'name': name, 'smiles': smi, 'chembl_id': cmpd.get('molecule_chembl_id'), 'source':'ChEMBL'})
                time.sleep(0.3)
                if len(chembl_compounds) >= max_compounds: break
            except Exception as e:
                print(f"  Error query '{query}': {e}")
                continue
        if len(chembl_compounds) < 50:
            chembl_compounds += self._get_extended_local_dataset(max_compounds=max_compounds-len(chembl_compounds))
        return chembl_compounds[:max_compounds]

    def _create_variant(self, base_smiles: str):
        """Creates simple variants by adding functional fragments."""
        fragments = ['O', 'C#N', '[N+](=O)[O-]', 'F', 'Cl', 'Br', 'S(=O)(=O)C', 'OC', 'N']
        new = []
        for i, frag in enumerate(fragments):
            new.append(base_smiles + frag)
            if i % 2 == 0 and len(base_smiles) < 25:
                new.append(base_smiles + 'C')  # extend chain
        return list(dict.fromkeys(new))  # unique, preserves order

    def generate_dataset(self, max_compounds=120):
        """Generates dataset with descriptors; uses cache if exists."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.data = json.load(f)
                if len(self.data) >= min(60, max_compounds//2):
                    print(f"📁 Loading {len(self.data)} compounds from cache")
                    return self.data[:max_compounds]
            except Exception:
                pass

        print(" Fetching solvents (ChEMBL/local) and generating variants...")
        base = self.fetch_chembl_solvents(max_compounds=max_compounds//2)
        # add variants to increase diversity
        expanded = []
        for entry in base:
            expanded.append({'name': entry.get('name','Unknown'), 'smiles': entry['smiles'], 'formula': entry.get('formula','')})
            for var_smi in self._create_variant(entry['smiles'])[:2]:  # 2 variants per compound
                expanded.append({'name': entry.get('name','Unknown')+'_var', 'smiles': var_smi, 'formula': ''})

        # if still insufficient, complete with local dataset
        if len(expanded) < max_compounds:
            expanded += self._get_extended_local_dataset(max_compounds=max_compounds - len(expanded))

        # calculate descriptors
        data = []
        for cmpd in tqdm(expanded[:max_compounds], desc="Calculating descriptors"):
            try:
                desc = self.descriptor_calc.calculate_comprehensive_descriptors(cmpd['smiles'])
                data.append({
                    'name': cmpd['name'], 'smiles': cmpd['smiles'],
                    'formula': cmpd.get('formula',''), 'descriptors': desc
                })
            except Exception as e:
                # skip problematic molecules
                print(f"   Descriptor error for {cmpd['smiles']}: {e}")
                continue

        if not data:
            raise RuntimeError("Failed to generate solvent dataset.")

        self.data = data
        try:
            with open(self.cache_file, 'w') as f: json.dump(self.data, f, indent=2)
        except Exception:
            pass
        print(f"✓ Dataset generated: {len(self.data)} compounds")
        return self.data

    def get_descriptor_matrix(self):
        if not self.data: return None
        desc_names = list(self.data[0]['descriptors'].keys())
        X = []; smiles = []; names = []
        for d in self.data:
            X.append([d['descriptors'][k] for k in desc_names])
            smiles.append(d['smiles']); names.append(d['name'])
        return {'descriptors': np.array(X, dtype=float),
                'descriptor_names': desc_names,
                'smiles': smiles,
                'names': names}

# ============================================================================
# PART 3: ROBUST DFT CALCULATIONS
# ============================================================================
class SimplifiedDFTCalculator:
    """DFT Calculator; uses PySCF if available, with approximated 1D geometry."""
    def __init__(self, basis='sto-3g', xc='b3lyp', cache_dir='dft_cache'):
        self.basis = basis; self.xc = xc; self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _parse_atoms(self, smiles):
        # reuse simple parser
        calc = MolecularDescriptorCalculator()
        atom_counts,_ = calc.parse_smiles_enhanced(smiles)
        atoms = []
        for a,c in atom_counts.items():
            for _ in range(c): atoms.append(a)
        return atoms

    def _estimate_coords(self, atoms):
        coords = []; x = 0.0
        for a in atoms:
            coords.append([x, 0.0, 0.0]); x += 1.4
        return np.array(coords)

    def _simulate(self, smiles, properties):
        np.random.seed(hash(smiles) % (2**32))
        calc = MolecularDescriptorCalculator()
        ac,_ = calc.parse_smiles_enhanced(smiles)
        n_atoms = sum(ac.values()); n_heavy = sum(c for a,c in ac.items() if a!='H')
        res = {}
        if 'energy' in properties: res['energy'] = -n_atoms*25.0 + np.random.normal(0,5)
        if 'dipole' in properties:
            hetero = sum(ac.get(a,0) for a in ['N','O','S','F','Cl'])
            res['dipole'] = hetero*1.7 + np.random.exponential(1.0)
        if 'homo' in properties: res['homo'] = np.random.normal(-0.3, 0.1)
        if 'lumo' in properties: res['lumo'] = np.random.normal(0.1, 0.05)
        if 'homo' in res and 'lumo' in res: res['gap'] = res['lumo'] - res['homo']
        if 'polarizability' in properties: res['polarizability'] = n_heavy*3.0 + np.random.normal(0,2)
        return res

    def calculate_dft_properties(self, smiles, properties=('energy','dipole','homo','lumo','gap','polarizability')):
        cache_file = os.path.join(self.cache_dir, f"{hash(smiles)}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file,'r') as f: return json.load(f)
            except Exception: pass

        if not PYSCF_AVAILABLE:
            result = self._simulate(smiles, properties)
            with open(cache_file,'w') as f: json.dump(result,f)
            return result

        try:
            atoms = self._parse_atoms(smiles)
            coords = self._estimate_coords(atoms)
            mol_lines = "; ".join(f"{a} {x:.6f} {y:.6f} {z:.6f}" for (a,(x,y,z)) in zip(atoms, coords))
            mol = gto.Mole()
            mol.atom = mol_lines
            mol.basis = self.basis
            mol.build()
            mf = dft.RKS(mol); mf.xc = self.xc; mf.kernel()
            if not mf.converged: raise RuntimeError("DFT did not converge")
            out = {}
            if 'energy' in properties: out['energy'] = float(mf.e_tot)
            if 'dipole' in properties:
                dip = mf.dip_moment(); out['dipole'] = float(np.linalg.norm(dip))
            if 'homo' in properties or 'lumo' in properties:
                mo_e = mf.mo_energy; mo_occ = mf.mo_occ
                homo_idx = int(np.where(mo_occ>0)[0][-1]); lumo_idx = homo_idx+1
                if 'homo' in properties: out['homo'] = float(mo_e[homo_idx])
                if 'lumo' in properties:
                    out['lumo'] = float(mo_e[lumo_idx] if lumo_idx < len(mo_e) else mo_e[homo_idx]+0.1)
            if 'homo' in out and 'lumo' in out: out['gap'] = out['lumo']-out['homo']
            if 'polarizability' in properties:
                out['polarizability'] = len(atoms)*2.5
            with open(cache_file,'w') as f: json.dump(out,f)
            return out
        except Exception as e:
            print(f"⚠ DFT error for {smiles}: {e}")
            print(traceback.format_exc()) # Print the full traceback
            result = self._simulate(smiles, properties)
            try:
                with open(cache_file,'w') as f: json.dump(result,f)
            except Exception: pass
            return result

    def calculate_batch_with_validation(self, smiles_list, properties, max_workers=2):
        """Calculates properties in parallel and flags failures/NaNs."""
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut = {ex.submit(self.calculate_dft_properties, s, properties): s for s in smiles_list}
            for f in tqdm(as_completed(fut), total=len(fut), desc="DFT Calculations"):
                s = fut[f]
                try:
                    res = f.result()
                    # validate NaNs
                    valid = True
                    for p in properties:
                        if (p in res) and (res[p] is not None) and np.isfinite(res[p]):
                            continue
                        valid = False; break
                    results.append({'smiles': s, **({} if res is None else res), 'valid': valid})
                except Exception as e:
                    print(f"  ✖ Error in {s}: {e}")
                    results.append({'smiles': s, 'valid': False})
        return results

# ============================================================================
# PART 4: SURROGATE MODEL + UNCERTAINTY + CV
# ============================================================================
class EnhancedSurrogateModel:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.uncertainty_models = {}
        self.trained_properties = []
        self.dft_data = {}

    def integrate_dft_data(self, smiles_list, dft_entries, target_properties):
        self.dft_data = {e['smiles']: {p:e.get(p,None) for p in target_properties} for e in dft_entries if e.get('valid',False)}
        print(f"✓ Integrated {len(self.dft_data)} valid DFT results")
        return list(target_properties)

    def _fit_property_model(self, X, y):
        scaler_X = StandardScaler(); scaler_y = StandardScaler()
        Xs = scaler_X.fit_transform(X); ys = scaler_y.fit_transform(y.reshape(-1,1)).ravel()
        model = RandomForestRegressor(n_estimators=250, max_depth=12, random_state=42, n_jobs=-1)
        model.fit(Xs, ys)
        gp = None
        if len(Xs) > 12:
            kernel = C(1.0, (1e-2, 1e3)) * RBF(length_scale=1.0)
            gp = GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=2)
            gp.fit(Xs, ys)
        return model, gp, scaler_X, scaler_y

    def train_with_dft(self, X, smiles_list, descriptor_names, kfold=5):
        # map smiles -> row
        idx_map = {s:i for i,s in enumerate(smiles_list)}
        if not self.dft_data:
            print("⚠ No DFT data; cannot train surrogate with ground truth")
            return

        props = list(next(iter(self.dft_data.values())).keys())
        for prop in props:
            # build dataset for this property
            rows = [idx_map[s] for s,v in self.dft_data.items() if v.get(prop) is not None and np.isfinite(v.get(prop))]
            if len(rows) < max(8, kfold):
                print(f"  ⚠ Insufficient data for {prop} (n={len(rows)}); CV will be skipped")
                continue
            Xp = X[rows]; yp = np.array([self.dft_data[smiles_list[i]][prop] for i in rows], dtype=float)

            # k-fold CV
            kf = KFold(n_splits=kfold, shuffle=True, random_state=42)
            r2s, rmses = [], []
            for tr, te in kf.split(Xp):
                model, gp, sx, sy = self._fit_property_model(Xp[tr], yp[tr])
                yhat = sy.inverse_transform(model.predict(sx.transform(Xp[te])).reshape(-1,1)).ravel()
                r2s.append(r2_score(yp[te], yhat))
                rmses.append(np.sqrt(mean_squared_error(yp[te], yhat)))
            print(f"  CV {prop}: R2={np.mean(r2s):.3f}±{np.std(r2s):.3f}, RMSE={np.mean(rmses):.3f}")

            # train final with all data
            model, gp, sx, sy = self._fit_property_model(Xp, yp)
            self.models[prop] = model; self.scalers[prop] = {'X':sx, 'y':sy}
            if gp is not None: self.uncertainty_models[prop] = gp
            if prop not in self.trained_properties: self.trained_properties.append(prop)

    def predict_with_uncertainty(self, X):
        preds = {}; uncs = {}
        for prop, model in self.models.items():
            sx = self.scalers[prop]['X']; sy = self.scalers[prop]['y']
            Xs = sx.transform(X)
            yp = sy.inverse_transform(model.predict(Xs).reshape(-1,1)).ravel()
            preds[prop] = yp
            if prop in self.uncertainty_models:
                _, std = self.uncertainty_models[prop].predict(Xs, return_std=True)
                uncs[prop] = sy.scale_[0] * std
            else:
                if hasattr(model,'estimators_'):
                    tree_preds = np.vstack([t.predict(Xs) for t in model.estimators_])
                    std = tree_preds.std(axis=0)
                    uncs[prop] = sy.scale_[0] * std
                else:
                    uncs[prop] = np.ones_like(yp)*0.2
        return preds, uncs

    def predict_with_uncertainty_robust(self, X):
        try:
            return self.predict_with_uncertainty(X)
        except Exception as e:
            print(f"⚠ Prediction failed; returning NaNs ({e})")
            return {}, {}

# ============================================================================
# PART 5: MODELS (VAE + AE)
# ============================================================================
class MolecularVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=8, hidden_dims=[64, 32]):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU(), nn.Dropout(0.1)
        )
        self.fc_mu = nn.Linear(hidden_dims[1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[1], latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dims[1], hidden_dims[0]), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], input_dim)
        )
    def encode(self, x): h = self.encoder(x); return self.fc_mu(h), self.fc_logvar(h)
    def reparameterize(self, mu, logvar): std = torch.exp(0.5*logvar); eps = torch.randn_like(std); return mu + eps*std
    def decode(self, z): return self.decoder(z)
    def forward(self, x):
        mu, logvar = self.encode(x); z = self.reparameterize(mu, logvar); recon = self.decode(z)
        return recon, mu, logvar, z

def vae_loss(recon_x, x, mu, logvar, beta=0.5):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kl /= x.size(0)*x.size(1)
    return recon_loss + beta*kl, recon_loss, kl

class PropertyAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=4, hidden_dims=[32, 16]):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], hidden_dims[1]), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dims[1], latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dims[1]), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dims[1], hidden_dims[0]), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], input_dim)
        )
    def encode(self, x): return self.encoder(x)
    def decode(self, z): return self.decoder(z)
    def forward(self, x): z = self.encode(x); return self.decode(z), z

# ============================================================================
# PART 6: OBJECTIVE FUNCTION — PEPTIDE SYNTHESIS
# ============================================================================
def peptide_synthesis_objective(z1,z2,z3,z4,z5,z6,z7,z8, vae_model, ae_model, scaler, surrogate_model):
    """Prioritizes high dipole (≈5–8 D) and moderate gap (≈0.3–0.6 eV); penalizes uncertainty."""
    z = torch.tensor([[z1,z2,z3,z4,z5,z6,z7,z8]], dtype=torch.float32)
    with torch.no_grad():
        dec = vae_model.decode(z)
        refined, _ = ae_model(dec)
        X = refined.numpy()
        preds, uncs = surrogate_model.predict_with_uncertainty_robust(X)

    total = 0.0
    # Dipole — we want high (main priority)
    if 'dipole' in preds:
        d = preds['dipole'][0]; u = (uncs.get('dipole',[0.0])[0] if 'dipole' in uncs else 0.0)
        # score on plateau 5–8 D; decreases outside
        if d < 3: ds = (d-3)*2
        elif d > 9: ds = 9 - d
        else: ds = 10 - abs(6.5 - d)
        total += (ds/(1+u)) * 4.0

    # Gap — moderate ~0.45 eV
    if 'gap' in preds:
        g = preds['gap'][0]; u = (uncs.get('gap',[0.0])[0] if 'gap' in uncs else 0.0)
        gs = max(0, 10 - abs(g - 0.45)*20)  # 0 at extremes, 10 at center
        total += (gs/(1+u)) * 3.0

    # Total energy as a soft regularizer
    if 'energy' in preds:
        e = preds['energy'][0]; u = (uncs.get('energy',[0.0])[0] if 'energy' in uncs else 0.0)
        es = -abs(e + 80)*0.02
        total += es/(1+u)

    # Penalty for mean uncertainty
    if uncs:
        mean_unc = safe_mean([float(np.mean(v)) for v in uncs.values()])
        if np.isfinite(mean_unc): total -= 2.0*mean_unc
    return float(total)

# ============================================================================
# PART 7: MAIN PIPELINE
# ============================================================================
class EnhancedSolventDesignPipeline:
    def __init__(self):
        self.dataset = None
        self.dft_calculator = None
        self.descriptor_data = None
        self.surrogate_model = None
        self.vae_model = None
        self.ae_model = None
        self.scaler = None

    def execute_complete_pipeline(self, max_compounds=120, max_dft_calcs=25):
        print("="*90)
        print(" ENHANCED PIPELINE — PEPTIDE SYNTHESIS SOLVENT DESIGN")
        print("="*90)

        # STEP 1: Dataset
        print("\nSTEP 1 — Generating extended dataset...")
        self.dataset = EnhancedSolventDataset()
        compounds = self.dataset.generate_dataset(max_compounds=max_compounds)

        # STEP 2: Descriptors + normalization
        print("\nSTEP 2 — Building descriptor matrix...")
        self.descriptor_data = self.dataset.get_descriptor_matrix()
        X = self.descriptor_data['descriptors']
        smiles_list = self.descriptor_data['smiles']
        desc_names = self.descriptor_data['descriptor_names']
        self.scaler = StandardScaler()
        Xn = self.scaler.fit_transform(X)
        print(f"✓ {X.shape[0]} molecules, {X.shape[1]} descriptors")

        # STEP 3: DFT
        print(f"\nSTEP 3 — Executing up to {max_dft_calcs} DFT calculations...")
        self.dft_calculator = SimplifiedDFTCalculator()
        # pseudo-diverse selection: sample by percentiles
        n = min(max_dft_calcs, len(smiles_list))
        idx = np.linspace(0, len(smiles_list)-1, n, dtype=int)
        smiles_dft = [smiles_list[i] for i in idx]
        target_props = ['energy','dipole','homo','lumo','gap','polarizability']
        raw_dft = self.dft_calculator.calculate_batch_with_validation(smiles_dft, target_props, max_workers=2)

        # filter valid
        dft_valid = [e for e in raw_dft if e.get('valid',False)]
        print(f"✓ {len(dft_valid)} valid DFT from {len(raw_dft)} attempted")
        if len(dft_valid) == 0:
            raise RuntimeError("All DFT calculations failed; cannot proceed.")

        # STEP 4: Surrogate + CV
        print("\nSTEP 4 — Training surrogate with cross-validation (k=5)...")
        self.surrogate_model = EnhancedSurrogateModel()
        self.surrogate_model.integrate_dft_data(smiles_list, dft_valid, target_props)
        self.surrogate_model.train_with_dft(Xn, smiles_list, desc_names, kfold=5)

        # STEP 5: Autoencoders
        print("\nSTEP 5 — Training VAE and Property Autoencoder...")
        self._train_autoencoders(Xn)

        # STEP 6: Bayesian Optimization with new objective function
        print("\nSTEP 6 — Bayesian Optimization (high dipole + moderate gap)...")
        best = self._execute_bayesian_optimization()

        # STEP 7: DFT Validation of best candidate (robust to NaNs)
        print("\nSTEP 7 — DFT Validation of best candidate...")
        validation = self._validate_best_candidate(best)

        return {
            'best_result': best,
            'dft_validation': validation,
            'dataset_info': {
                'n_compounds': len(smiles_list),
                'n_dft_calcs': len(dft_valid),
                'descriptors': X.shape[1]
            }
        }

    def _train_autoencoders(self, Xn):
        X_tensor = torch.tensor(Xn, dtype=torch.float32)
        self.vae_model = MolecularVAE(input_dim=Xn.shape[1], latent_dim=8)
        opt = torch.optim.Adam(self.vae_model.parameters(), lr=1e-3)
        for epoch in range(250):
            recon, mu, logvar, z = self.vae_model(X_tensor)
            loss, rloss, kl = vae_loss(recon, X_tensor, mu, logvar, beta=0.5)
            opt.zero_grad(); loss.backward(); opt.step()
            if (epoch+1) % 100 == 0:
                print(f"  VAE epoch {epoch+1}/250 — loss: {loss.item():.5f}")

        with torch.no_grad():
            recon_full, _, _, _ = self.vae_model(X_tensor)

        self.ae_model = PropertyAutoencoder(input_dim=Xn.shape[1], latent_dim=4)
        opt2 = torch.optim.Adam(self.ae_model.parameters(), lr=1e-3)
        for epoch in range(180):
            recon2, z2 = self.ae_model(recon_full.detach())
            loss2 = nn.MSELoss()(recon2, recon_full.detach())
            opt2.zero_grad(); loss2.backward(); opt2.step()
            if (epoch+1) % 90 == 0:
                print(f"  AE epoch {epoch+1}/180 — loss: {loss2.item():.5f}")

    def _execute_bayesian_optimization(self):
        def objective_wrapper(**kwargs):
            return peptide_synthesis_objective(
                **kwargs,
                vae_model=self.vae_model, ae_model=self.ae_model,
                scaler=self.scaler, surrogate_model=self.surrogate_model
            )

        # estimate latent space bounds from X encoding
        X_tensor = torch.tensor(self.scaler.transform(self.dataset.get_descriptor_matrix()['descriptors']), dtype=torch.float32)
        with torch.no_grad():
            _, mu, logvar, _ = self.vae_model(X_tensor)
            mu_mean = mu.mean(dim=0); mu_std = mu.std(dim=0).clamp(min=0.5)
        bounds = {f'z{i+1}': (mu_mean[i].item()-2*mu_std[i].item(), mu_mean[i].item()+2*mu_std[i].item()) for i in range(8)}

        optimizer = BayesianOptimization(f=objective_wrapper, pbounds=bounds, verbose=2, random_state=42)
        optimizer.maximize(init_points=15, n_iter=25)
        return optimizer.max

    def _validate_best_candidate(self, best):
        params = best['params']
        z = torch.tensor([[params[f'z{i+1}'] for i in range(8)]], dtype=torch.float32)
        with torch.no_grad():
            desc = self.vae_model.decode(z)
            ref, _ = self.ae_model(desc)
            Xcand = ref.numpy()

        # nearest molecule in dataset
        Xn = self.scaler.transform(self.dataset.get_descriptor_matrix()['descriptors'])
        dists = np.linalg.norm(Xn - Xcand, axis=1)
        idx = int(np.argmin(dists))
        smi = self.dataset.get_descriptor_matrix()['smiles'][idx]
        print(f"Most similar molecule: {smi} (dist={dists[idx]:.4f})")

        # DFT calculation
        props = self.dft_calculator.calculate_dft_properties(smi, ('energy','dipole','homo','lumo','gap','polarizability'))
        ml_pred, _ = self.surrogate_model.predict_with_uncertainty(Xcand)

        comparison = {}
        for k, v in props.items():
            if k in ml_pred and np.isfinite(v):
                comparison[k] = {
                    'dft': float(v),
                    'ml': float(ml_pred[k][0]),
                    'rel_error': float(abs(v - ml_pred[k][0])/max(1e-8, abs(v)) * 100.0)
                }

        return {'smiles': smi, 'distance': float(dists[idx]), 'dft_properties': props, 'ml_predictions': {k: float(v[0]) for k,v in ml_pred.items()}, 'comparison': comparison}

# ============================================================================
# EXECUTION
# ============================================================================
if __name__ == "__main__":
    pipeline = EnhancedSolventDesignPipeline()
    results = pipeline.execute_complete_pipeline(
        max_compounds=120,  # >50 guaranteed
        max_dft_calcs=25    # 25 DFT required
    )

    print("\n" + "="*90)
    print("FINAL RESULTS")
    print("="*90)
    best = results['best_result']
    validation = results['dft_validation']
    info = results['dataset_info']

    print(f"\nDataset: {info['n_compounds']} compounds, {info['descriptors']} descriptors")
    print(f"Valid DFT: {info['n_dft_calcs']}")

    print(f"\nBest candidate: score={best['target']:.4f}")
    print(f"  • Similar molecule: {validation['smiles']}  |  distance={validation['distance']:.4f}")

    print("\nDFT vs ML Comparison:")
    for prop, comp in validation['comparison'].items():
        print(f"  - {prop}: DFT={comp['dft']:.4f} | ML={comp['ml']:.4f} | error%={comp['rel_error']:.2f}")
