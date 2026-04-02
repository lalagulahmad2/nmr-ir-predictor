#!/usr/bin/env python3
"""Molecular NMR and IR Spectrum Predictor"""

from flask import Flask, render_template, request, jsonify
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
import numpy as np
import json
import re
import time
import threading

# Optional SDBS scraping dependencies — fail gracefully if missing
try:
    import requests as _requests
    from bs4 import BeautifulSoup as _BS
    _SDBS_AVAILABLE = True
except ImportError:
    _SDBS_AVAILABLE = False

app = Flask(__name__)

# ============================================================
# FUNCTIONAL GROUP DETECTION
# ============================================================

SMARTS_PATTERNS = {
    'alcohol':        '[OX2H1][CX4]',
    'phenol':         '[OX2H1]c',
    'carboxylic_acid':'[CX3](=O)[OX2H1]',
    'ester':          '[CX3](=O)[OX2][#6]',
    'ketone':         '[CX4,c][CX3](=O)[CX4,c]',
    'aldehyde':       '[CX3H1](=O)',
    'primary_amide':  '[CX3](=O)[NX3H2]',
    'secondary_amide':'[CX3](=O)[NX3H1]',
    'tertiary_amide': '[CX3](=O)[NX3H0]',
    'primary_amine':  '[NX3H2][CX4]',
    'secondary_amine':'[NX3H1]([CX4])[CX4]',
    'aromatic_amine': '[NX3H2]c',
    'aromatic':       'a',
    'benzene_ring':   'c1ccccc1',
    'alkene':         '[CX3]=[CX3]',
    'terminal_alkene':'[CX3H2]=[CX3]',
    'alkyne':         '[CX2]#[CX2]',
    'terminal_alkyne':'[CX2H]#[CX2]',
    'ether':          '[OX2]([CX4,c])[CX4,c]',
    'epoxide':        '[C;r3][O;r3][C;r3]',
    'nitrile':        '[NX1]#[CX2]',
    'isocyanate':     '[NX2]=[C]=[OX1]',
    'nitro':          '[$([NX3](=O)=O),$([NX3+](=O)[O-])]',
    'thiol':          '[SX2H1]',
    'thioether':      '[SX2]([CX4])[CX4]',
    'fluoride':       '[F][CX4]',
    'chloride':       '[Cl][CX4]',
    'bromide':        '[Br][CX4]',
    'iodide':         '[I][CX4]',
    'aryl_chloride':  '[Cl]c',
    'aryl_bromide':   '[Br]c',
    'aryl_fluoride':  '[F]c',
    'anhydride':      '[CX3](=O)[OX2][CX3](=O)',
    'lactone':        '[CX3](=O)[OX2][CX4;R]',
    'lactam':         '[CX3](=O)[NX3;R]',
    'acyl_halide':    '[CX3](=O)[F,Cl,Br,I]',
    'sulfoxide':      '[#16X3](=O)',
    'sulfone':        '[#16X4](=O)(=O)',
    'imine':          '[CX3]=[NX2]',
    'sulfonamide':    '[SX4](=O)(=O)[NX3H]',
    'phosphate':      '[PX4](=O)',
}

def get_functional_groups(mol):
    results = {}
    for name, smarts in SMARTS_PATTERNS.items():
        try:
            pat = Chem.MolFromSmarts(smarts)
            if pat:
                matches = mol.GetSubstructMatches(pat)
                if matches:
                    results[name] = len(matches)
        except Exception:
            pass
    return results


def count_ch_groups(mol):
    """Return (n_CH3, n_CH2, n_CH, n_Cq) for sp3 carbons."""
    from rdkit.Chem import rdchem
    mol_h = Chem.AddHs(mol)
    Chem.SanitizeMol(mol_h)
    ch3 = ch2 = ch = cq = 0
    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() == 6 and atom.GetHybridization() == rdchem.HybridizationType.SP3:
            n_h = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
            if n_h == 3:   ch3 += 1
            elif n_h == 2: ch2 += 1
            elif n_h == 1: ch  += 1
            else:          cq  += 1
    return ch3, ch2, ch, cq


# ============================================================
# IR SPECTRUM
# ============================================================

def _lor(x, x0, w, I):
    return I / (1.0 + ((x - x0) / w) ** 2)

def _gau(x, x0, w, I):
    return I * np.exp(-np.log(2) * ((x - x0) / w) ** 2)

def _mix(x, x0, w, I, eta=0.5):
    return eta * _lor(x, x0, w, I) + (1 - eta) * _gau(x, x0, w, I)


def generate_ir_spectrum(mol):
    fg = get_functional_groups(mol)
    ch3, ch2, ch, cq = count_ch_groups(mol)

    wn = np.linspace(400, 4000, 3601)
    A  = np.zeros(len(wn))

    def p(c, w, I, shape='L'):
        nonlocal A
        if   shape == 'L': A += _lor(wn, c, w, I)
        elif shape == 'G': A += _gau(wn, c, w, I)
        else:              A += _mix(wn, c, w, I)

    # ---- C-H STRETCHES (sp3) ----
    if ch3 > 0:
        s = min(ch3 / 2.0, 2.0)
        p(2962, 13, 0.55*s); p(2872, 10, 0.28*s)
    if ch2 > 0:
        s = min(ch2 / 2.5, 2.0)
        p(2926, 13, 0.65*s); p(2853, 10, 0.38*s)
    if ch  > 0: p(2890, 10, 0.15)

    # ---- C-H BENDS (sp3) ----
    if ch3 > 0:
        p(1460, 12, 0.30); p(1375,  9, 0.38)
    if ch2 > 0:
        p(1465, 12, 0.28)
        if ch2 >= 4:
            p(722, 10, 0.28); p(730, 10, 0.18)

    # ---- AROMATIC ----
    if 'aromatic' in fg:
        p(3030,  9, 0.17); p(3060,  9, 0.13); p(3090,  7, 0.09)
        p(1600, 14, 0.35, 'M'); p(1580, 10, 0.22, 'M')
        p(1500, 12, 0.40, 'M'); p(1450, 10, 0.28, 'M')
        p( 750, 20, 0.62, 'M'); p( 700, 15, 0.52, 'M')

    # ---- ALCOHOL ----
    if 'alcohol' in fg:
        p(3340, 190, 0.90, 'G'); p(3200, 200, 0.35, 'G')
        p(1050,  25, 0.50, 'M'); p(1100,  20, 0.32, 'M')
        p(1380,  12, 0.20); p(650, 65, 0.28, 'G')

    # ---- PHENOL ----
    if 'phenol' in fg:
        p(3350, 160, 0.82, 'G')
        p(1230,  22, 0.55, 'M'); p(1180, 18, 0.40, 'M'); p(1150, 14, 0.28, 'M')

    # ---- CARBOXYLIC ACID ----
    if 'carboxylic_acid' in fg:
        for c in range(2500, 3301, 60):
            p(c, 75, 0.13, 'G')
        p(1710, 18, 0.96); p(1250, 22, 0.50, 'M')
        p(1210, 18, 0.40, 'M'); p(1395, 14, 0.22, 'M')
        p(935, 50, 0.40, 'G')

    # ---- ESTER ----
    if 'ester' in fg and 'carboxylic_acid' not in fg:
        p(1740, 16, 0.96)
        p(1200, 28, 0.72, 'M'); p(1170, 22, 0.55, 'M'); p(1050, 22, 0.38, 'M')
    elif 'ester' in fg:  # acid + ester (e.g. aspirin)
        p(1760, 16, 0.80)
        p(1200, 25, 0.60, 'M')

    # ---- LACTONE (5-membered) ----
    if 'lactone' in fg and 'ester' not in fg:
        p(1770, 16, 0.95)

    # ---- KETONE ----
    if 'ketone' in fg and 'carboxylic_acid' not in fg and 'ester' not in fg:
        p(1715, 17, 0.95)

    # ---- ALDEHYDE ----
    if 'aldehyde' in fg:
        p(1725, 17, 0.95)
        p(2820, 14, 0.30); p(2720, 14, 0.32)   # Fermi doublet

    # ---- AMIDE ----
    if 'primary_amide' in fg:
        p(3350, 27, 0.48, 'M'); p(3180, 27, 0.42, 'M')
        p(1680, 18, 0.90); p(1620, 15, 0.55, 'M'); p(1400, 12, 0.22, 'M')
    if 'secondary_amide' in fg:
        p(3310, 24, 0.62, 'M'); p(3080, 24, 0.20, 'M')
        p(1660, 18, 0.90); p(1540, 15, 0.70, 'M'); p(1270, 15, 0.32, 'M')
    if 'tertiary_amide' in fg:
        p(1650, 20, 0.90)

    # ---- AMINES ----
    if 'primary_amine' in fg:
        p(3380, 22, 0.42, 'M'); p(3300, 22, 0.36, 'M')
        p(1610, 14, 0.32, 'M'); p(1080, 18, 0.24, 'M')
    if 'secondary_amine' in fg:
        p(3330, 25, 0.48, 'M')
    if 'aromatic_amine' in fg:
        p(3450, 22, 0.40, 'M'); p(3360, 22, 0.35, 'M')
        p(1600, 14, 0.35, 'M')

    # ---- ALKENE ----
    if 'alkene' in fg:
        p(3080, 14, 0.22); p(1640, 18, 0.38, 'M')
        if 'terminal_alkene' in fg:
            p(990, 14, 0.58); p(910, 18, 0.65)
        else:
            p(970, 14, 0.55); p(700, 18, 0.42)

    # ---- ALKYNE ----
    if 'terminal_alkyne' in fg:
        p(3300, 12, 0.54); p(2120, 22, 0.54); p(630, 22, 0.48, 'M')
    elif 'alkyne' in fg:
        p(2190, 25, 0.18)   # internal C≡C (weak)

    # ---- ETHER ----
    if 'ether' in fg:
        p(1120, 28, 0.68, 'M'); p(1070, 22, 0.42, 'M')

    # ---- EPOXIDE ----
    if 'epoxide' in fg:
        p(1250, 18, 0.45, 'M'); p(950, 20, 0.40, 'M'); p(810, 18, 0.38, 'M')

    # ---- NITRILE ----
    if 'nitrile' in fg:
        p(2230, 18, 0.65)

    # ---- ISOCYANATE ----
    if 'isocyanate' in fg:
        p(2270, 28, 0.95)   # very strong, broad

    # ---- NITRO ----
    if 'nitro' in fg:
        p(1540, 18, 0.90); p(1370, 15, 0.85)

    # ---- THIOL ----
    if 'thiol' in fg:
        p(2550, 12, 0.22)   # weak but diagnostic

    # ---- THIOETHER ----
    if 'thioether' in fg:
        p(1050, 18, 0.28, 'M'); p(700, 18, 0.32, 'M')

    # ---- HALIDES ----
    if 'fluoride' in fg:
        p(1100, 28, 0.68, 'M'); p(1050, 22, 0.55, 'M')
    if 'chloride' in fg:
        p(760, 22, 0.55, 'M'); p(660, 20, 0.44, 'M')
    if 'bromide' in fg:
        p(600, 28, 0.55, 'M'); p(555, 22, 0.42, 'M')
    if 'iodide' in fg:
        p(600, 30, 0.40, 'M'); p(510, 28, 0.36, 'M')
    if 'aryl_chloride' in fg:
        p(1090, 20, 0.44, 'M'); p(1080, 18, 0.38, 'M')
    if 'aryl_bromide' in fg:
        p(1075, 20, 0.38, 'M'); p(1045, 18, 0.34, 'M')

    # ---- ANHYDRIDE ----
    if 'anhydride' in fg:
        p(1820, 16, 0.88); p(1760, 16, 0.92)
        p(1050, 28, 0.65, 'M')

    # ---- ACYL HALIDE ----
    if 'acyl_halide' in fg:
        p(1800, 15, 0.96)

    # ---- SULFOXIDE / SULFONE ----
    if 'sulfoxide' in fg:
        p(1050, 20, 0.62, 'M')
    if 'sulfone' in fg:
        p(1150, 22, 0.78, 'M'); p(1050, 20, 0.72, 'M')

    # ---- IMINE ----
    if 'imine' in fg:
        p(1640, 18, 0.55, 'M')

    # ---- SULFONAMIDE ----
    if 'sulfonamide' in fg:
        p(3300, 25, 0.50, 'M')
        p(1160, 22, 0.72, 'M'); p(1340, 18, 0.68, 'M')

    # ---- FINGERPRINT REGION COMPLEXITY ----
    formula = rdMolDescriptors.CalcMolFormula(mol)
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(formula))
    rng = np.random.default_rng(seed)

    # C-C skeletal stretches (one per bond)
    for bond in mol.GetBonds():
        if (bond.GetBeginAtom().GetAtomicNum() == 6 and
                bond.GetEndAtom().GetAtomicNum() == 6 and
                not bond.GetIsAromatic()):
            c  = rng.uniform(850, 1300)
            I  = rng.uniform(0.04, 0.22)
            w  = rng.uniform(8, 22)
            A += _gau(wn, c, w, I)

    # Additional skeletal vibrations
    n_extra = min(mol.GetNumAtoms(), 18)
    for _ in range(n_extra):
        c  = rng.uniform(450, 1500)
        I  = rng.uniform(0.02, 0.18)
        w  = rng.uniform(6, 28)
        A += _gau(wn, c, w, I)

    # Very subtle baseline noise
    A += rng.normal(0, 0.006, len(wn))
    A = np.clip(A, 0, 3.2)

    transmittance = 10.0 ** (-A) * 100.0
    return wn.tolist(), transmittance.tolist()


# ============================================================
# 1H NMR
# ============================================================

SHOOLERY = {
    'alkyl': 0.47, 'vinyl': 1.32, 'alkynyl': 1.44, 'aryl': 1.85,
    'F': 3.30, 'Cl': 2.53, 'Br': 2.33, 'I': 1.82,
    'OR': 2.36, 'OH': 2.56, 'OAr': 3.23, 'OCOR': 3.01,
    'COR': 1.20, 'CHO': 1.03, 'COOH': 1.55, 'COOR': 1.55,
    'CONR': 1.03, 'CN': 1.70, 'NO2': 3.36,
    'NH2': 1.57, 'NHR': 1.57, 'NR2': 1.57,
    'SR': 1.64, 'SH': 1.64,
}


def _shoolery(nbr, mol_h):
    """Return Shoolery constant for a heavy-atom neighbor."""
    from rdkit.Chem import rdchem
    an  = nbr.GetAtomicNum()
    hyb = nbr.GetHybridization()

    if an == 6:
        if nbr.GetIsAromatic():
            return SHOOLERY['aryl']
        if hyb == rdchem.HybridizationType.SP2:
            for b in nbr.GetBonds():
                o = b.GetOtherAtom(nbr)
                if o.GetAtomicNum() == 8 and b.GetBondTypeAsDouble() == 2:
                    # C=O: determine subtype
                    if nbr.GetTotalNumHs() == 1:
                        return SHOOLERY['CHO']
                    for b2 in nbr.GetBonds():
                        x = b2.GetOtherAtom(nbr)
                        if x.GetIdx() == o.GetIdx(): continue
                        if x.GetAtomicNum() == 8:
                            return SHOOLERY['COOH'] if x.GetTotalNumHs() else SHOOLERY['COOR']
                        if x.GetAtomicNum() == 7:
                            return SHOOLERY['CONR']
                    return SHOOLERY['COR']
            return SHOOLERY['vinyl']
        if hyb == rdchem.HybridizationType.SP:
            return SHOOLERY['alkynyl']
        return SHOOLERY['alkyl']

    if an == 8:
        for b in nbr.GetBonds():
            x = b.GetOtherAtom(nbr)
            if x.GetIsAromatic(): return SHOOLERY['OAr']
            if x.GetAtomicNum() == 6:
                for b2 in x.GetBonds():
                    o2 = b2.GetOtherAtom(x)
                    if o2.GetAtomicNum() == 8 and b2.GetBondTypeAsDouble() == 2 and o2.GetIdx() != nbr.GetIdx():
                        return SHOOLERY['OCOR']
        return SHOOLERY['OH'] if nbr.GetTotalNumHs() else SHOOLERY['OR']

    if an == 7:
        nh = nbr.GetTotalNumHs()
        return SHOOLERY['NH2'] if nh >= 2 else (SHOOLERY['NHR'] if nh == 1 else SHOOLERY['NR2'])
    if an == 9:  return SHOOLERY['F']
    if an == 17: return SHOOLERY['Cl']
    if an == 35: return SHOOLERY['Br']
    if an == 53: return SHOOLERY['I']
    if an == 16: return SHOOLERY['SH'] if nbr.GetTotalNumHs() else SHOOLERY['SR']
    if an == 7 and any(b.GetBondTypeAsDouble() == 2 for b in nbr.GetBonds()):
        return SHOOLERY['NO2']
    return 0.5


def predict_h_shift(h_atom, mol_h):
    from rdkit.Chem import rdchem
    parent  = h_atom.GetNeighbors()[0]
    an      = parent.GetAtomicNum()
    hyb     = parent.GetHybridization()

    # ---- H on O ----
    if an == 8:
        heavy = [n for n in parent.GetNeighbors() if n.GetAtomicNum() != 1]
        if heavy:
            c = heavy[0]
            if c.GetAtomicNum() == 6:
                if c.GetHybridization() == rdchem.HybridizationType.SP2:
                    for b in c.GetBonds():
                        if b.GetOtherAtom(c).GetAtomicNum() == 8 and b.GetBondTypeAsDouble() == 2:
                            return 11.5   # COOH
                    return 6.5  # enol
                if c.GetIsAromatic():
                    return 8.2   # phenol
        return 2.8  # alcohol

    # ---- H on N ----
    if an == 7:
        heavy = [n for n in parent.GetNeighbors() if n.GetAtomicNum() != 1]
        for c in heavy:
            if c.GetAtomicNum() == 6:
                for b in c.GetBonds():
                    if b.GetOtherAtom(c).GetAtomicNum() == 8 and b.GetBondTypeAsDouble() == 2:
                        return 7.8   # amide N-H
                if c.GetIsAromatic():
                    return 4.5   # Ar-NH2
        return 2.2

    # ---- H on S ----
    if an == 16:
        return 1.8

    # ---- H on C ----
    if an == 6:
        # Aldehyde H
        for b in parent.GetBonds():
            x = b.GetOtherAtom(parent)
            if x.GetAtomicNum() == 8 and b.GetBondTypeAsDouble() == 2:
                if parent.GetTotalNumHs() == 1:
                    return 9.75

        # Aromatic H
        if parent.GetIsAromatic():
            return _aromatic_shift(parent, mol_h)

        # Alkene H (sp2)
        if hyb == rdchem.HybridizationType.SP2:
            shift = 5.25
            for nbr in parent.GetNeighbors():
                if nbr.GetAtomicNum() == 1: continue
                an2 = nbr.GetAtomicNum()
                if an2 == 8: shift -= 0.8   # enol
                if an2 == 6 and nbr.GetIsAromatic(): shift += 0.55
            return shift

        # Alkyne H (sp)
        if hyb == rdchem.HybridizationType.SP:
            return 2.5

        # sp3 C-H: Shoolery
        base = 0.23
        for nbr in parent.GetNeighbors():
            if nbr.GetAtomicNum() == 1: continue
            base += _shoolery(nbr, mol_h)
        return base

    return 1.0


def _aromatic_shift(c_atom, mol):
    base = 7.27
    ring_info = mol.GetRingInfo()
    rings = [r for r in ring_info.AtomRings() if c_atom.GetIdx() in r]
    if not rings:
        return base
    ring = rings[0]
    pos  = ring.index(c_atom.GetIdx())
    n    = len(ring)
    for i, idx in enumerate(ring):
        atom = mol.GetAtomWithIdx(idx)
        for nbr in atom.GetNeighbors():
            if nbr.GetIdx() in ring:
                continue
            an   = nbr.GetAtomicNum()
            dist = min(abs(i - pos), n - abs(i - pos))
            eff  = 0.0
            if an == 6:
                eff = 0.2 if nbr.GetHybridization().name == 'SP2' else -0.15
            elif an == 8: eff = -0.45
            elif an == 7: eff = -0.65
            elif an == 9:  eff = -0.20
            elif an == 17: eff = 0.03
            elif an == 35: eff = 0.22
            elif an == 53:
                # Regular ArI: ortho +0.40, meta -0.26, para -0.65
                # Hypervalent I(III): acts as EWG, all positions slightly downfield
                n_bonds_i = len(nbr.GetBonds())
                if n_bonds_i > 1:  # hypervalent (PIDA, DMP, etc.)
                    eff = {1: 0.55, 2: 0.24, 3: 0.31}.get(dist, 0.0)
                else:              # regular monoiodide (PhI, alkyl-I)
                    eff = {1: 0.308, 2: -0.578, 3: -0.65}.get(dist, 0.0)
            if   dist == 1: base += eff * 1.3
            elif dist == 2: base += eff * 0.45
            elif dist == 3: base += eff * 1.0
    return base


# ============================================================
# J-COUPLING CONSTANTS
# ============================================================

def _mult_name(n):
    names = ['s', 'd', 't', 'q', 'quint', 'sext', 'sept']
    return names[n] if n < len(names) else 'm'


def _compute_3j(c1, c2):
    """Vicinal 3J H-H coupling (Hz) for H on c1 coupled to H on c2."""
    from rdkit.Chem import rdchem
    h1 = c1.GetHybridization()
    h2 = c2.GetHybridization()
    SP2 = rdchem.HybridizationType.SP2
    SP3 = rdchem.HybridizationType.SP3

    # Aromatic-aromatic ortho: 6-9 Hz
    if c1.GetIsAromatic() and c2.GetIsAromatic():
        return 8.0

    # sp2–sp2 alkene
    if h1 == SP2 and h2 == SP2 and not c1.GetIsAromatic() and not c2.GetIsAromatic():
        # Direct C=C bond → geminal coupling on =CH2
        for bond in c1.GetBonds():
            if bond.GetOtherAtom(c1).GetIdx() == c2.GetIdx():
                bd = bond.GetBondTypeAsDouble()
                if abs(bd - 2.0) < 0.1:
                    # terminal =CH2: use geminal
                    h_c1 = sum(1 for n in c1.GetNeighbors() if n.GetAtomicNum() == 1)
                    h_c2 = sum(1 for n in c2.GetNeighbors() if n.GetAtomicNum() == 1)
                    if h_c1 >= 2 or h_c2 >= 2:
                        return 2.0          # geminal vinyl
                    return 15.0             # internal: assume trans dominant
        return 10.0  # vicinal through two sp2 (non-directly bonded)

    # sp3–sp3 vicinal: Karplus-adjusted ~7 Hz
    if h1 == SP3 and h2 == SP3:
        j = 7.0
        ewg = {8, 7, 9, 17, 35, 53}
        for n in c1.GetNeighbors():
            if n.GetAtomicNum() in ewg: j -= 0.25
        for n in c2.GetNeighbors():
            if n.GetAtomicNum() in ewg: j -= 0.25
        return round(max(j, 5.5), 1)

    # sp3–sp2 allylic/benzylic: 4J ≈ 1.5 Hz (small, often unresolved)
    return 1.5


def _build_coupling(couplings):
    """
    couplings: list of (J_hz, n_protons) sorted largest-J first
    Returns {'mult': str, 'j': [float]}
    """
    if not couplings:
        return {'mult': 's', 'j': []}

    total = sum(c for _, c in couplings)
    distinct_j = sorted({round(j, 1) for j, _ in couplings}, reverse=True)

    if len(distinct_j) == 1:
        return {'mult': _mult_name(total), 'j': [distinct_j[0]]}

    # Multiple J values → dd / dt / td / ddd / m
    individual = []
    for j, c in sorted(couplings, key=lambda x: x[0], reverse=True):
        individual.extend([round(j, 1)] * c)

    n = len(individual)
    if n == 2:
        name = 'dd'
    elif n == 3:
        cnt = {}
        for v in individual:
            cnt[v] = cnt.get(v, 0) + 1
        mx = max(cnt.values())
        if mx == 2:
            pair_j = max((v for v, c in cnt.items() if c == 2))
            lone_j = [v for v, c in cnt.items() if c == 1][0]
            name = 'dt' if pair_j < lone_j else 'td'
        else:
            name = 'ddd'
    elif n == 4:
        name = 'ddd'
    else:
        name = 'm'
    return {'mult': name, 'j': distinct_j}


def get_coupling_and_j(h_atom, mol_h):
    """Returns {'mult': str, 'j': [float in Hz]}."""
    from rdkit.Chem import rdchem
    parent = h_atom.GetNeighbors()[0]
    h_idx  = h_atom.GetIdx()

    # Exchangeable protons → broad singlet, no J
    if parent.GetAtomicNum() in (7, 8, 16):
        return {'mult': 's (br)', 'j': []}

    hyb = parent.GetHybridization()

    # ── AROMATIC H ──
    if parent.GetIsAromatic():
        ring_info = mol_h.GetRingInfo()
        rings = [r for r in ring_info.AtomRings() if parent.GetIdx() in r]
        if not rings:
            return {'mult': 's', 'j': []}
        ring    = rings[0]
        pos     = list(ring).index(parent.GetIdx())
        n_ring  = len(ring)
        j_map   = {}   # j_val → n_protons
        for i, ridx in enumerate(ring):
            if ridx == parent.GetIdx(): continue
            ra = mol_h.GetAtomWithIdx(ridx)
            if not ra.GetIsAromatic(): continue
            h_on = sum(1 for x in ra.GetNeighbors() if x.GetAtomicNum() == 1)
            if h_on == 0: continue
            dist = min(abs(i - pos), n_ring - abs(i - pos))
            if dist == 1:   j = 8.0   # ortho
            elif dist == 2: j = 2.0   # meta 4J
            else: continue            # para: < 1 Hz, omit
            j = round(j, 1)
            j_map[j] = j_map.get(j, 0) + h_on
        couplings = sorted(j_map.items(), key=lambda x: x[0], reverse=True)
        return _build_coupling(couplings)

    # ── sp (alkyne terminal) – no vicinal coupling ──
    if hyb == rdchem.HybridizationType.SP:
        return {'mult': 's', 'j': []}

    # ── sp2 ALKENE H ──
    if hyb == rdchem.HybridizationType.SP2:
        j_map = {}
        for nbr in parent.GetNeighbors():
            if nbr.GetAtomicNum() != 6: continue
            h_on = sum(1 for x in nbr.GetNeighbors()
                       if x.GetAtomicNum() == 1 and x.GetIdx() != h_idx)
            if h_on == 0: continue
            bond  = mol_h.GetBondBetweenAtoms(parent.GetIdx(), nbr.GetIdx())
            is_db = bond and abs(bond.GetBondTypeAsDouble() - 2.0) < 0.1
            j = 2.0 if is_db else _compute_3j(parent, nbr)
            j = round(j, 1)
            j_map[j] = j_map.get(j, 0) + h_on
        # geminal vinyl partner (other H on same sp2 C)
        gem = [x for x in parent.GetNeighbors()
               if x.GetAtomicNum() == 1 and x.GetIdx() != h_idx]
        if gem:
            j_map[2.0] = j_map.get(2.0, 0) + len(gem)
        couplings = sorted(j_map.items(), key=lambda x: x[0], reverse=True)
        return _build_coupling(couplings)

    # ── sp3 H ──
    j_map = {}
    for nbr in parent.GetNeighbors():
        if nbr.GetAtomicNum() != 6: continue
        h_on = sum(1 for x in nbr.GetNeighbors()
                   if x.GetAtomicNum() == 1 and x.GetIdx() != h_idx)
        if h_on == 0: continue
        j = round(_compute_3j(parent, nbr), 1)
        j_map[j] = j_map.get(j, 0) + h_on
    couplings = sorted(j_map.items(), key=lambda x: x[0], reverse=True)
    return _build_coupling(couplings)


def predict_hnmr(mol):
    mol_h = Chem.AddHs(mol)
    Chem.SanitizeMol(mol_h)
    ranks = list(Chem.CanonicalRankAtoms(mol_h, breakTies=False))

    h_groups = {}
    for atom in mol_h.GetAtoms():
        if atom.GetAtomicNum() == 1:
            r = ranks[atom.GetIdx()]
            h_groups.setdefault(r, []).append(atom)

    peaks = []
    for h_list in h_groups.values():
        rep    = h_list[0]
        shift  = predict_h_shift(rep, mol_h)
        coup   = get_coupling_and_j(rep, mol_h)
        peaks.append({
            'shift': round(shift, 2),
            'mult':  coup['mult'],
            'j':     coup['j'],        # list of Hz values
            'n':     len(h_list)
        })

    peaks.sort(key=lambda x: x['shift'], reverse=True)

    # Build spectrum — use actual J values converted to ppm at 400 MHz
    FREQ = 400.0
    ppm = np.linspace(-1, 15, 32001)
    I   = np.zeros(len(ppm))

    for pk in peaks:
        s = pk['shift']; n = pk['n']; m = pk['mult']
        j_list = pk.get('j', [])
        # Convert first (largest) J to ppm; fall back to 7 Hz default
        J_ppm = (j_list[0] / FREQ) if j_list else (7.0 / FREQ)
        J2    = (j_list[1] / FREQ) if len(j_list) > 1 else J_ppm
        w = 0.003
        if 's' in m:
            sub = [(s, 1)]
        elif m == 'd':
            sub = [(s - J_ppm/2, 1), (s + J_ppm/2, 1)]
        elif m == 't':
            sub = [(s - J_ppm, 1), (s, 2), (s + J_ppm, 1)]
        elif m == 'q':
            sub = [(s - 1.5*J_ppm, 1), (s - 0.5*J_ppm, 3), (s + 0.5*J_ppm, 3), (s + 1.5*J_ppm, 1)]
        elif m == 'quint':
            sub = [(s + (k-2)*J_ppm, [1,4,6,4,1][k]) for k in range(5)]
        elif m == 'sext':
            sub = [(s + (k-2.5)*J_ppm, [1,5,10,10,5,1][k]) for k in range(6)]
        elif m == 'sept':
            sub = [(s + (k-3)*J_ppm, [1,6,15,20,15,6,1][k]) for k in range(7)]
        elif m == 'dd':
            sub = [(s - J_ppm/2 - J2/2, 1), (s + J_ppm/2 - J2/2, 1),
                   (s - J_ppm/2 + J2/2, 1), (s + J_ppm/2 + J2/2, 1)]
        elif m in ('dt', 'td'):
            sub = [(s - J_ppm/2 - J2, 1), (s - J_ppm/2, 2), (s - J_ppm/2 + J2, 1),
                   (s + J_ppm/2 - J2, 1), (s + J_ppm/2, 2), (s + J_ppm/2 + J2, 1)]
        elif m == 'ddd':
            J3 = (j_list[2] / FREQ) if len(j_list) > 2 else J2
            sub = [(s + e1*J_ppm/2 + e2*J2/2 + e3*J3/2, 1)
                   for e1 in (-1, 1) for e2 in (-1, 1) for e3 in (-1, 1)]
        else:
            sub = [(s, 1)]
        for sp, si in sub:
            I += _lor(ppm, sp, w, si * n * 0.12)

    if I.max() > 0:
        I /= I.max()
    return peaks, ppm.tolist(), I.tolist()


def _build_hnmr_spectrum(peaks):
    """Rebuild ¹H NMR (x, y) from a peak list (supports corrected shifts)."""
    FREQ = 400.0
    ppm  = np.linspace(-1, 15, 32001)
    I    = np.zeros(len(ppm))
    for pk in peaks:
        s  = pk['shift']; n = pk['n']; m = pk['mult']
        j_list = pk.get('j', [])
        J_ppm = (j_list[0] / FREQ) if j_list else (7.0 / FREQ)
        J2    = (j_list[1] / FREQ) if len(j_list) > 1 else J_ppm
        w = 0.003
        if 's' in m:
            sub = [(s, 1)]
        elif m == 'd':
            sub = [(s - J_ppm/2, 1), (s + J_ppm/2, 1)]
        elif m == 't':
            sub = [(s - J_ppm, 1), (s, 2), (s + J_ppm, 1)]
        elif m == 'q':
            sub = [(s-1.5*J_ppm,1),(s-.5*J_ppm,3),(s+.5*J_ppm,3),(s+1.5*J_ppm,1)]
        elif m == 'quint':
            sub = [(s+(k-2)*J_ppm, [1,4,6,4,1][k]) for k in range(5)]
        elif m == 'sext':
            sub = [(s+(k-2.5)*J_ppm, [1,5,10,10,5,1][k]) for k in range(6)]
        elif m == 'sept':
            sub = [(s+(k-3)*J_ppm, [1,6,15,20,15,6,1][k]) for k in range(7)]
        elif m == 'dd':
            sub = [(s-J_ppm/2-J2/2,1),(s+J_ppm/2-J2/2,1),
                   (s-J_ppm/2+J2/2,1),(s+J_ppm/2+J2/2,1)]
        elif m in ('dt', 'td'):
            sub = [(s-J_ppm/2-J2,1),(s-J_ppm/2,2),(s-J_ppm/2+J2,1),
                   (s+J_ppm/2-J2,1),(s+J_ppm/2,2),(s+J_ppm/2+J2,1)]
        elif m == 'ddd':
            J3 = (j_list[2]/FREQ) if len(j_list)>2 else J2
            sub = [(s+e1*J_ppm/2+e2*J2/2+e3*J3/2, 1)
                   for e1 in (-1,1) for e2 in (-1,1) for e3 in (-1,1)]
        else:
            sub = [(s, 1)]
        for sp, si in sub:
            I += _lor(ppm, sp, w, si * n * 0.12)
    if I.max() > 0:
        I /= I.max()
    return ppm.tolist(), I.tolist()


# ============================================================
# 13C NMR
# ============================================================

def predict_c_shift(c_atom, mol):
    from rdkit.Chem import rdchem
    hyb = c_atom.GetHybridization()

    # Aromatic
    if c_atom.GetIsAromatic():
        shift = 128.5
        for nbr in c_atom.GetNeighbors():
            if nbr.GetIsAromatic(): continue
            an = nbr.GetAtomicNum()
            if an == 8: shift += 28
            elif an == 7: shift -= 8
            elif an == 6 and nbr.GetHybridization() == rdchem.HybridizationType.SP2: shift += 8
            elif an == 9:  shift += 30   # ArF ipso ~159 ppm
            elif an == 17: shift += 5    # ArCl ipso ~134 ppm
            elif an == 35: shift -= 7    # ArBr ipso ~122 ppm
            elif an == 53:               # heavy-atom effect on ipso C
                n_bonds_i = len(nbr.GetBonds())
                shift += (5 if n_bonds_i > 1 else -38)  # hypervalent ~134, regular ~91 ppm
        return shift

    # sp2 (C=O or C=C)
    if hyb == rdchem.HybridizationType.SP2:
        for b in c_atom.GetBonds():
            x = b.GetOtherAtom(c_atom)
            if x.GetAtomicNum() == 8 and b.GetBondTypeAsDouble() == 2:
                # Carbonyl: determine type
                nh_nbrs = [n for n in c_atom.GetNeighbors() if n.GetIdx() != x.GetIdx()]
                for n2 in nh_nbrs:
                    if n2.GetAtomicNum() == 8:
                        return 178 if n2.GetTotalNumHs() else 171
                    if n2.GetAtomicNum() == 7:
                        return 168
                    if n2.GetAtomicNum() == 1:
                        return 202  # aldehyde
                return 210  # ketone
            if x.GetAtomicNum() == 7 and b.GetBondTypeAsDouble() == 2:
                return 164  # imine
        shift = 125  # alkene
        for nbr in c_atom.GetNeighbors():
            if nbr.GetAtomicNum() == 8: shift += 32
            elif nbr.GetAtomicNum() == 7: shift += 15
        return shift

    # sp (alkyne / nitrile)
    if hyb == rdchem.HybridizationType.SP:
        for b in c_atom.GetBonds():
            if b.GetBondTypeAsDouble() == 3:
                x = b.GetOtherAtom(c_atom)
                if x.GetAtomicNum() == 7: return 118
                if x.GetAtomicNum() == 6:
                    return 68 if c_atom.GetTotalNumHs() else 80
        return 75

    # sp3 – Grant-Paul inspired
    shift = 15.0
    for nbr in c_atom.GetNeighbors():
        if nbr.GetAtomicNum() == 1: continue
        an = nbr.GetAtomicNum()
        if an == 6:
            shift += 9.1
            if nbr.GetIsAromatic(): shift += 6
            elif nbr.GetHybridization() == rdchem.HybridizationType.SP2: shift += 2
        elif an == 8: shift += 50
        elif an == 7: shift += 28
        elif an == 9: shift += 65
        elif an == 17: shift += 30
        elif an == 35: shift += 22
        elif an == 53: shift -= 35   # heavy-atom upfield shift (CH₃I: ~−20 ppm)
        elif an == 16: shift += 18

    # Beta O/N/X (smaller effect)
    for nbr1 in c_atom.GetNeighbors():
        if nbr1.GetAtomicNum() == 1: continue
        for nbr2 in nbr1.GetNeighbors():
            if nbr2.GetIdx() == c_atom.GetIdx() or nbr2.GetAtomicNum() == 1: continue
            an2 = nbr2.GetAtomicNum()
            if an2 == 8:  shift += 7
            elif an2 == 7: shift += 4
            elif an2 in (9, 17): shift += 4

    return shift


def predict_cnmr(mol):
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    c_groups = {}
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 6:
            c_groups.setdefault(ranks[atom.GetIdx()], []).append(atom)

    peaks = []
    for c_list in c_groups.values():
        rep   = c_list[0]
        shift = predict_c_shift(rep, mol)
        nh    = rep.GetTotalNumHs()
        dept  = ['C', 'CH', 'CH₂', 'CH₃'][min(nh, 3)]
        peaks.append({'shift': round(shift, 1), 'dept': dept, 'count': len(c_list)})

    peaks.sort(key=lambda x: x['shift'], reverse=True)

    ppm = np.linspace(-10, 240, 50001)
    I   = np.zeros(len(ppm))
    for pk in peaks:
        I += _lor(ppm, pk['shift'], 0.5, pk['count'] * 0.6)
    if I.max() > 0:
        I /= I.max()
    return peaks, ppm.tolist(), I.tolist()


def _build_cnmr_spectrum(peaks):
    """Rebuild ¹³C NMR (x, y) from a peak list."""
    ppm = np.linspace(-10, 240, 50001)
    I   = np.zeros(len(ppm))
    for pk in peaks:
        I += _lor(ppm, pk['shift'], 0.5, pk['count'] * 0.6)
    if I.max() > 0:
        I /= I.max()
    return ppm.tolist(), I.tolist()


# ============================================================
# MOLECULE SVG
# ============================================================

def mol_to_svg(mol):
    try:
        AllChem.Compute2DCoords(mol)
        d = rdMolDraw2D.MolDraw2DSVG(420, 280)
        d.drawOptions().addStereoAnnotation = True
        d.DrawMolecule(mol)
        d.FinishDrawing()
        return d.GetDrawingText()
    except Exception as e:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="420" height="280"><text x="10" y="30" fill="red">Error: {e}</text></svg>'


# ============================================================
# KETCHER / INDIGO API  (RDKit-backed)
# ============================================================

def _ket_to_mol(ket_str):
    """Parse Ketcher KET JSON to RDKit mol."""
    try:
        from rdkit.Chem import RWMol
        from rdkit.Chem.rdchem import BondType as BT, Atom as RDAtom
        data = json.loads(ket_str)
        mol_data = next((v for v in data.values()
                         if isinstance(v, dict) and v.get('type') == 'molecule'), None)
        if mol_data is None:
            return None
        rwmol = RWMol()
        idx_map = {}
        for i, a in enumerate(mol_data.get('atoms', [])):
            lbl = a.get('label', 'C')
            try:
                atom = RDAtom(lbl)
            except Exception:
                atom = RDAtom(6)
            atom.SetFormalCharge(int(a.get('charge', 0)))
            iso = int(a.get('isotope', 0))
            if iso:
                atom.SetIsotope(iso)
            idx_map[i] = rwmol.AddAtom(atom)
        bt_map = {1: BT.SINGLE, 2: BT.DOUBLE, 3: BT.TRIPLE, 4: BT.AROMATIC}
        for b in mol_data.get('bonds', []):
            a1, a2 = int(b['atoms'][0]), int(b['atoms'][1])
            if a1 in idx_map and a2 in idx_map:
                rwmol.AddBond(idx_map[a1], idx_map[a2], bt_map.get(int(b.get('type', 1)), BT.SINGLE))
        Chem.SanitizeMol(rwmol)
        return rwmol.GetMol()
    except Exception:
        return None


def _mol_to_ket(mol):
    """Convert RDKit mol to Ketcher KET JSON string."""
    from rdkit.Chem.rdchem import BondType as BT
    if mol.GetNumConformers() == 0:
        AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    atoms = []
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        entry = {
            'label': atom.GetSymbol(),
            'location': [round(float(pos.x) / 1.5, 4), round(float(pos.y) / 1.5, 4), 0.0],
        }
        ch = atom.GetFormalCharge()
        if ch:
            entry['charge'] = ch
        iso = atom.GetIsotope()
        if iso:
            entry['isotope'] = iso
        atoms.append(entry)
    bt_map = {BT.SINGLE: 1, BT.DOUBLE: 2, BT.TRIPLE: 3, BT.AROMATIC: 4}
    bonds = [{'type': bt_map.get(b.GetBondType(), 1),
              'atoms': [b.GetBeginAtomIdx(), b.GetEndAtomIdx()]}
             for b in mol.GetBonds()]
    return json.dumps({
        'root': {'nodes': [{'$ref': 'mol0'}], 'connections': []},
        'mol0': {'type': 'molecule', 'atoms': atoms, 'bonds': bonds},
    })


def _parse_struct(s):
    """Parse KET / MOL / SMILES string to RDKit mol."""
    if not s or not s.strip():
        return None
    # KET JSON
    stripped = s.strip()
    if stripped.startswith('{'):
        mol = _ket_to_mol(stripped)
        if mol:
            return mol
    # MOL / SDF block — do NOT strip; V2000 requires exact line positions
    if '\n' in s and ('M  END' in s or 'V2000' in s or 'M  V30' in s):
        mol = Chem.MolFromMolBlock(s, sanitize=True, removeHs=True)
        if mol:
            return mol
        # Fallback: try with explicit \n normalisation
        mol = Chem.MolFromMolBlock(s.replace('\r\n', '\n'), sanitize=True, removeHs=True)
        if mol:
            return mol
    return Chem.MolFromSmiles(stripped)


def _indigo_info_response():
    return jsonify({'indigo_version': '1.0.0-rdkit', 'imago_versions': [], 'isAvailable': True})

@app.route('/v2/info', methods=['GET', 'POST', 'OPTIONS'])
def indigo_info_short():
    return _indigo_info_response()

@app.route('/v2/indigo/info', methods=['GET', 'POST', 'OPTIONS'])
def indigo_info():
    return _indigo_info_response()


@app.route('/v2/indigo/convert', methods=['POST', 'OPTIONS'])
def indigo_convert():
    if request.method == 'OPTIONS':
        return app.make_default_options_response()
    data = request.get_json(force=True) or {}
    struct_str = data.get('struct', '')
    out_fmt = data.get('output_format', 'chemical/x-daylight-smiles')
    if not struct_str or not struct_str.strip():
        return jsonify({'struct': '', 'format': out_fmt, 'fileContent': ''})
    mol = _parse_struct(struct_str)
    if mol is None:
        # Return empty success so Ketcher doesn't error on empty canvas
        return jsonify({'struct': '', 'format': out_fmt, 'fileContent': ''})
    try:
        if 'smiles' in out_fmt or 'daylight' in out_fmt:
            result = Chem.MolToSmiles(mol)
            fmt = 'chemical/x-daylight-smiles'
        elif 'ket' in out_fmt:
            result = _mol_to_ket(mol)
            fmt = 'chemical/x-indigo-ket'
        elif 'molfile' in out_fmt or 'mdl' in out_fmt:
            result = Chem.MolToMolBlock(mol)
            fmt = 'chemical/x-mdl-molfile'
        else:
            result = Chem.MolToSmiles(mol)
            fmt = 'chemical/x-daylight-smiles'
        return jsonify({'struct': result, 'format': fmt, 'fileContent': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _indigo_passthrough():
    if request.method == 'OPTIONS':
        return app.make_default_options_response()
    data = request.get_json(force=True) or {}
    s = data.get('struct', '')
    return jsonify({'struct': s, 'format': 'chemical/x-indigo-ket', 'fileContent': s})


@app.route('/v2/indigo/layout',      methods=['POST', 'OPTIONS'])
def indigo_layout():      return _indigo_passthrough()

@app.route('/v2/indigo/clean',       methods=['POST', 'OPTIONS'])
def indigo_clean():       return _indigo_passthrough()

@app.route('/v2/indigo/aromatize',   methods=['POST', 'OPTIONS'])
def indigo_aromatize():   return _indigo_passthrough()

@app.route('/v2/indigo/dearomatize', methods=['POST', 'OPTIONS'])
def indigo_dearomatize(): return _indigo_passthrough()

@app.route('/v2/indigo/check',       methods=['POST', 'OPTIONS'])
def indigo_check():
    if request.method == 'OPTIONS':
        return app.make_default_options_response()
    return jsonify({})

@app.route('/v2/indigo/calculate',   methods=['POST', 'OPTIONS'])
def indigo_calculate():
    if request.method == 'OPTIONS':
        return app.make_default_options_response()
    return jsonify({})

@app.route('/v2/indigo/render',      methods=['POST', 'OPTIONS'])
def indigo_render():
    if request.method == 'OPTIONS':
        return app.make_default_options_response()
    return '', 200


# ============================================================
# SDBS INTEGRATION
# ============================================================

_SDBS_BASE    = 'https://sdbs.db.aist.go.jp'
_SDBS_HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/124.0 Safari/537.36'),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}
_SDBS_SESSION      = None
_SDBS_SESSION_TS   = 0.0          # unix timestamp of last authentication
_SDBS_SESSION_LOCK = threading.Lock()
_SDBS_CACHE        = {}            # (formula, inchi_key) → (h_shifts, c_shifts, ts)
_SDBS_JOBS         = {}            # job_id → {'status', 'hnmr', 'cnmr'}
_SDBS_JOBS_LOCK    = threading.Lock()
_SDBS_CACHE_TTL    = 7200          # 2 h
_SDBS_SESSION_TTL  = 540           # re-auth before 10-min cookie expires


def _sdbs_session():
    """Return an authenticated requests.Session, re-authenticating if needed."""
    global _SDBS_SESSION, _SDBS_SESSION_TS
    if not _SDBS_AVAILABLE:
        return None
    with _SDBS_SESSION_LOCK:
        if _SDBS_SESSION and (time.time() - _SDBS_SESSION_TS) < _SDBS_SESSION_TTL:
            return _SDBS_SESSION
        # Create / refresh session
        s = _requests.Session()
        s.headers.update(_SDBS_HEADERS)
        try:
            r1 = s.get(_SDBS_BASE + '/', timeout=8)
            soup = _BS(r1.text, 'html.parser')
            vs   = soup.find('input', {'name': '__VIEWSTATE'})
            vsg  = soup.find('input', {'name': '__VIEWSTATEGENERATOR'})
            ev   = soup.find('input', {'name': '__EVENTVALIDATION'})
            post = {
                '__VIEWSTATE':            vs.get('value','')  if vs  else '',
                '__VIEWSTATEGENERATOR':   vsg.get('value','') if vsg else '',
                '__EVENTVALIDATION':      ev.get('value','')  if ev  else '',
                'ctl00$BodyContentPlaceHolder$DisclaimeraAccept':
                    'I agree the disclaimer and use SDBS.',
            }
            s.post(_SDBS_BASE + '/', data=post, timeout=8, allow_redirects=True)
            if 'SDBS_ENTRANCE_CK' in s.cookies:
                _SDBS_SESSION    = s
                _SDBS_SESSION_TS = time.time()
                return s
        except Exception as exc:
            print(f'[SDBS] auth error: {exc}')
    return None


def _sdbs_viewstate(sess, url, params=None):
    """GET a page and return (soup, vs, vsg, ev) for subsequent POST."""
    r = sess.get(url, params=params, timeout=8)
    soup = _BS(r.text, 'html.parser')
    vs   = soup.find('input', {'name': '__VIEWSTATE'})
    vsg  = soup.find('input', {'name': '__VIEWSTATEGENERATOR'})
    ev   = soup.find('input', {'name': '__EVENTVALIDATION'})
    return soup, (
        vs.get('value','')  if vs  else '',
        vsg.get('value','') if vsg else '',
        ev.get('value','')  if ev  else '',
    )


def _sdbs_search(sess, formula):
    """
    Search SDBS by molecular formula.
    Returns list of dicts {sdbsno, name, hnmr_fname, cnmr_fname}.
    """
    search_url = _SDBS_BASE + '/SearchInformation.aspx'
    try:
        # GET the blank search form to harvest ViewState tokens + default field values
        r1 = sess.get(search_url, timeout=8)
        soup1 = _BS(r1.text, 'html.parser')
        # Only include hidden & text/select fields — skip ALL submit buttons
        post = {}
        for el in soup1.find_all('input'):
            name = el.get('name')
            if not name: continue
            if el.get('type', 'text').lower() == 'submit': continue
            post[name] = el.get('value', '')
        for el in soup1.find_all('select'):
            name = el.get('name')
            if not name: continue
            opt = el.find('option', selected=True) or el.find('option')  # fallback to first
            post[name] = opt.get('value', '') if opt else ''
        # Set formula in both body input and header hidden (JS copies body→header on click)
        post['ctl00$BodyContentPlaceHolder$INP_formula'] = formula
        post['ctl00$CtlMasterHeader$formula']            = formula
        # Add the search button (simulates clicking it — only one submit in POST)
        post['ctl00$BodyContentPlaceHolder$SearchButton'] = 'Search'

        # POST with Referer/Origin to satisfy ASP.NET validation; handle redirect manually
        r2 = sess.post(search_url, data=post, timeout=12, allow_redirects=False,
                       headers={'Referer': search_url, 'Origin': _SDBS_BASE})
        if r2.status_code in (301, 302, 303):
            loc = r2.headers.get('Location', '')
            if loc and not loc.startswith('http'):
                loc = _SDBS_BASE + '/' + loc.lstrip('/')
            r2 = sess.get(loc, timeout=12, headers={'Referer': search_url})
        soup2 = _BS(r2.text, 'html.parser')
    except Exception as exc:
        print(f'[SDBS] search error: {exc}')
        return []

    results = []
    for row in soup2.select('tr'):
        sno_td  = row.select_one('td.sdbsno-val')
        name_td = row.select_one('td.comp-name')
        hnmr_td = row.select_one('td.hnmr-val')
        cnmr_td = row.select_one('td.cnmr-val')
        if not sno_td:
            continue
        m = re.search(r'CompoundView\.aspx\?sdbsno=(\d+)', str(sno_td))
        if not m:
            continue
        sdbsno = m.group(1)
        name   = name_td.get_text(strip=True) if name_td else ''

        def _fname(td, pat):
            if not td: return None
            fm = re.search(pat, str(td).replace('&amp;', '&'))
            return fm.group(2) if fm else None

        results.append({
            'sdbsno':     sdbsno,
            'name':       name,
            'hnmr_fname': _fname(hnmr_td,
                r'HNmrSpectralView\.aspx\?imgdir=(\w+)&fname=(\w+)&sdbsno=\d+'),
            'cnmr_fname': _fname(cnmr_td,
                r'CNmrSpectralView\.aspx\?imgdir=(\w+)&fname=(\w+)&sdbsno=\d+'),
        })
    return results


def _sdbs_inchikey(sess, sdbsno):
    """Return InChIKey from SDBS CompoundView page, or None."""
    try:
        r = sess.get(_SDBS_BASE + '/CompoundView.aspx',
                     params={'sdbsno': sdbsno}, timeout=8)
        soup = _BS(r.text, 'html.parser')
        el = soup.find(id='BodyContentPlaceHolder_InChIKey')
        if el:
            ik = el.get_text(strip=True)
            if re.match(r'^[A-Z]{14}-[A-Z]{10}-[A-Z]$', ik):
                return ik
        # Fallback: find InChI and convert
        el2 = soup.find(id='BodyContentPlaceHolder_InChI')
        if el2:
            inchi = el2.get_text(strip=True)
            if inchi.startswith('InChI='):
                ik2 = Chem.InchiToInchiKey(inchi)
                if ik2: return ik2
    except Exception:
        pass
    return None


def _sdbs_nmr_peaks(sess, sdbsno, fname, nmr_type):
    """
    Fetch peak shifts from SDBS spectral view.
    nmr_type: 'H' or 'C'. Returns sorted list of floats or None.
    """
    if not fname:
        return None
    try:
        if nmr_type == 'H':
            url    = _SDBS_BASE + '/HNmrSpectralView.aspx'
            params = {'imgdir': 'hsp', 'fname': fname, 'sdbsno': sdbsno}
            lo, hi = 0.0, 15.0
        else:
            url    = _SDBS_BASE + '/CNmrSpectralView.aspx'
            params = {'imgdir': 'cds', 'fname': fname, 'sdbsno': sdbsno}
            lo, hi = 0.0, 250.0
        r = sess.get(url, params=params, timeout=10)
        soup = _BS(r.text, 'html.parser')
        table = (soup.find(id='BodyContentPlaceHolder_OldShiftTable') or
                 soup.find('table', class_='ShiftTable'))
        if not table:
            return None
        shifts = []
        for row in table.find_all('tr'):
            cells = row.find_all('td')
            if len(cells) < 2:
                continue
            raw = cells[1].get_text(strip=True).rstrip('.')
            if not raw: continue
            try:
                v = float(raw)
                if lo <= v <= hi:
                    shifts.append(round(v, 3))
            except ValueError:
                pass
        return sorted(shifts, reverse=True) if len(shifts) >= 2 else None
    except Exception as exc:
        print(f'[SDBS] nmr fetch error ({nmr_type}): {exc}')
        return None


def _apply_sdbs_h(peaks, sdbs):
    """Replace ¹H predicted shifts with SDBS experimental values (proximity match)."""
    if not sdbs or not peaks:
        return peaks
    pred = sorted([dict(p) for p in peaks], key=lambda p: p['shift'], reverse=True)
    sdbs_s = sorted(sdbs, reverse=True)
    n_p, n_s = len(pred), len(sdbs_s)
    # Reject if count mismatch is too large
    if n_s == 0 or abs(n_p - n_s) > max(2, n_p // 3):
        return peaks
    if n_p == n_s:
        for pk, sh in zip(pred, sdbs_s):
            pk['shift'] = round(sh, 2)
    else:
        used = [False] * n_s
        for pk in pred:
            best_i, best_d = None, 99.0
            for i, sv in enumerate(sdbs_s):
                if not used[i] and abs(sv - pk['shift']) < best_d:
                    best_d, best_i = abs(sv - pk['shift']), i
            if best_i is not None:
                pk['shift'] = round(sdbs_s[best_i], 2)
                used[best_i] = True
    return pred


def _apply_sdbs_c(peaks, sdbs):
    """Replace ¹³C predicted shifts with SDBS experimental values."""
    if not sdbs or not peaks:
        return peaks
    pred = sorted([dict(p) for p in peaks], key=lambda p: p['shift'], reverse=True)
    sdbs_s = sorted(sdbs, reverse=True)
    n_p, n_s = len(pred), len(sdbs_s)
    if n_s == 0 or abs(n_p - n_s) > max(3, n_p // 2):
        return peaks
    if n_p == n_s:
        for pk, sh in zip(pred, sdbs_s):
            pk['shift'] = round(sh, 1)
    else:
        used = [False] * n_s
        for pk in pred:
            best_i, best_d = None, 999.0
            for i, sv in enumerate(sdbs_s):
                if not used[i] and abs(sv - pk['shift']) < best_d:
                    best_d, best_i = abs(sv - pk['shift']), i
            if best_i is not None:
                pk['shift'] = round(sdbs_s[best_i], 1)
                used[best_i] = True
    return pred


def _parse_jcamp_peaks(text, lo, hi):
    """Parse JCAMP-DX ##PEAK TABLE section; return list of shift floats in [lo, hi]."""
    peaks = []
    in_table = False
    for line in text.splitlines():
        line = line.strip()
        upper = line.upper()
        if '##PEAK TABLE' in upper or '##PEAKTABLE' in upper:
            in_table = True
            continue
        if in_table:
            if line.startswith('##'):
                break
            if not line:
                continue
            parts = line.replace(',', ' ').split()
            if parts:
                try:
                    v = float(parts[0])
                    if lo <= v <= hi:
                        peaks.append(round(v, 3))
                except ValueError:
                    pass
    return peaks


def _spinus_h_shifts(smiles):
    """
    Predict ¹H NMR shifts via NMRdb SPINUS service (nmrdb.org).
    Returns sorted list of unique shift floats (desc) or None.
    """
    try:
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        molblock = Chem.MolToMolBlock(mol)
        r = _requests.post(
            'https://www.nmrdb.org/service/predictor',
            data={'molfile': molblock},
            timeout=20
        )
        if r.status_code != 200 or not r.text.strip():
            return None
        if '<html' in r.text[:100].lower():
            return None
        # Each line: atom_idx \t atom_type \t shift \t ...
        all_shifts = []
        for line in r.text.strip().splitlines():
            parts = line.split('\t')
            if len(parts) >= 3:
                try:
                    s = float(parts[2])
                    if -2.0 <= s <= 16.0:
                        all_shifts.append(s)
                except ValueError:
                    pass
        if not all_shifts:
            return None
        # Deduplicate: group shifts within 0.05 ppm, take mean of each group
        all_shifts.sort()
        groups = [[all_shifts[0]]]
        for s in all_shifts[1:]:
            if s - groups[-1][-1] <= 0.05:
                groups[-1].append(s)
            else:
                groups.append([s])
        unique = sorted([round(sum(g) / len(g), 3) for g in groups], reverse=True)
        print(f'[spinus] {len(unique)} unique ¹H shifts for {smiles}')
        return unique
    except Exception as e:
        print(f'[spinus] error: {e}')
        return None


def _sdbs_worker(job_id, smiles, formula, inchi_key, h_peaks_base, c_peaks_base):
    """Background thread: fetch NMRdb SPINUS ¹H predictions and update job dict."""
    cache_key = (formula, inchi_key or '')
    cached = _SDBS_CACHE.get(cache_key)
    if cached and (time.time() - cached[2]) < _SDBS_CACHE_TTL:
        h_sdbs = cached[0]
    else:
        try:
            h_sdbs = _spinus_h_shifts(smiles)
        except Exception as exc:
            print(f'[spinus] worker exception: {exc}')
            h_sdbs = None
        _SDBS_CACHE[cache_key] = (h_sdbs, None, time.time(), None)

    if not h_sdbs:
        with _SDBS_JOBS_LOCK:
            _SDBS_JOBS[job_id]['status'] = 'not_found'
        return

    # Apply NMRdb shifts to ¹H peaks; keep ¹³C rule-based
    h_peaks = _apply_sdbs_h(h_peaks_base, h_sdbs)
    h_x, h_y = _build_hnmr_spectrum(h_peaks)
    c_x, c_y = _build_cnmr_spectrum(c_peaks_base)

    with _SDBS_JOBS_LOCK:
        _SDBS_JOBS[job_id].update({
            'status':  'done',
            'sdbsno':  None,
            'h_src':   'nmrdb',
            'c_src':   'predicted',
            'hnmr':    {'peaks': h_peaks, 'x': h_x, 'y': h_y},
            'cnmr':    {'peaks': c_peaks_base, 'x': c_x, 'y': c_y},
        })


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data   = request.get_json(force=True)
    smiles = data.get('smiles', '').strip()
    if not smiles:
        return jsonify({'error': 'No SMILES provided'}), 400

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({'error': 'Invalid SMILES string. Please enter a valid SMILES.'}), 400

    try:
        svg              = mol_to_svg(mol)
        fg               = get_functional_groups(mol)
        hnmr_pk, h_x, h_y = predict_hnmr(mol)
        cnmr_pk, c_x, c_y = predict_cnmr(mol)
        ir_x, ir_y         = generate_ir_spectrum(mol)
        formula = rdMolDescriptors.CalcMolFormula(mol)
        mw      = round(rdMolDescriptors.CalcExactMolWt(mol), 4)

        # Compute InChIKey for SDBS structure matching
        try:
            inchi     = Chem.MolToInchi(mol)
            inchi_key = Chem.InchiToInchiKey(inchi) if inchi else None
        except Exception:
            inchi_key = None

        # Kick off SDBS background lookup (if library available)
        job_id = None
        if _SDBS_AVAILABLE:
            job_id = f'{formula}_{inchi_key or smiles[:20]}'
            cache_key = (formula, inchi_key or '')
            cached = _SDBS_CACHE.get(cache_key)
            if cached and (time.time() - cached[2]) < _SDBS_CACHE_TTL:
                # Already cached — apply immediately
                h_sdbs, c_sdbs = cached[0], cached[1]
                if h_sdbs or c_sdbs:
                    hpk = _apply_sdbs_h(hnmr_pk, h_sdbs) if h_sdbs else hnmr_pk
                    cpk = _apply_sdbs_c(cnmr_pk, c_sdbs) if c_sdbs else cnmr_pk
                    hx2, hy2 = _build_hnmr_spectrum(hpk)
                    cx2, cy2 = _build_cnmr_spectrum(cpk)
                    hnmr_pk, h_x, h_y = hpk, hx2, hy2
                    cnmr_pk, c_x, c_y = cpk, cx2, cy2
                    job_id = None   # no polling needed
            else:
                with _SDBS_JOBS_LOCK:
                    if job_id not in _SDBS_JOBS:
                        _SDBS_JOBS[job_id] = {'status': 'pending'}
                        t = threading.Thread(
                            target=_sdbs_worker,
                            args=(job_id, smiles, formula, inchi_key,
                                  list(hnmr_pk), list(cnmr_pk)),
                            daemon=True
                        )
                        t.start()

        return jsonify({
            'svg':      svg,
            'formula':  formula,
            'mw':       mw,
            'fg':       list(fg.keys()),
            'hnmr':     {'peaks': hnmr_pk, 'x': h_x, 'y': h_y},
            'cnmr':     {'peaks': cnmr_pk, 'x': c_x, 'y': c_y},
            'ir':       {'x': ir_x, 'y': ir_y},
            'sdbs_job': job_id,          # None if already resolved or unavailable
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'detail': traceback.format_exc()}), 500


@app.route('/sdbs/<job_id>')
def sdbs_status(job_id):
    """Polling endpoint for SDBS background job."""
    with _SDBS_JOBS_LOCK:
        job = dict(_SDBS_JOBS.get(job_id, {'status': 'unknown'}))
    return jsonify(job)


@app.route('/nmrdb_debug')
def nmrdb_debug():
    """Diagnostic: test NMRdb SPINUS ¹H prediction."""
    smiles = request.args.get('smiles', 'CCO')
    out = {'available': _SDBS_AVAILABLE, 'smiles': smiles}
    if not _SDBS_AVAILABLE:
        out['error'] = 'requests not installed'
        return jsonify(out)
    try:
        out['h_shifts'] = _spinus_h_shifts(smiles)
        out['found'] = out['h_shifts'] is not None
    except Exception as e:
        import traceback
        out['error'] = str(e)
        out['tb'] = traceback.format_exc()
    return jsonify(out)


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, threaded=True, use_reloader=False)
