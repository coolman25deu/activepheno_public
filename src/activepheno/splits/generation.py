"""Deterministic compound/scaffold splits requiring validated image trajectories."""
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from activepheno.data.schema import ValidationError, require_strings
from activepheno.chemistry.structures import canonical_structure
from activepheno.data.validation import validate_trajectories

SPLITS = ["train", "validation", "test"]


def scaffold_key(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValidationError("invalid SMILES for scaffold")
    fragments = Chem.GetMolFrags(mol, asMols=True)
    organic = [f for f in fragments if any(a.GetAtomicNum() == 6 for a in f.GetAtoms())]
    if not organic:
        raise ValidationError("no organic fragment for scaffold")
    # Explicit deterministic parent rule; ties use canonical SMILES.
    parent = sorted(organic, key=lambda f: (-f.GetNumHeavyAtoms(), Chem.MolToSmiles(f)))[0]
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=parent, includeChirality=False)
    return scaffold or "ACYCLIC"


def validate_split(manifest: pd.DataFrame, compounds: pd.DataFrame, *, scaffold: bool) -> None:
    require_strings(manifest, ["compound_id", "series_id", "split"])
    if set(manifest.split) != set(SPLITS):
        raise ValidationError("train, validation, and test must all be nonempty")
    if manifest.series_id.duplicated().any():
        raise ValidationError("series appears multiple times in split manifest")
    if (manifest.groupby("compound_id").split.nunique() > 1).any():
        raise ValidationError("compound leakage across splits")
    joined = manifest.merge(compounds[["compound_id", "moa_id"]], on="compound_id", validate="many_to_one")
    if len(joined) != len(manifest):
        raise ValidationError("unknown compound in split manifest")
    for name in SPLITS:
        if set(joined.loc[joined.split == name, "moa_id"]) != set(compounds.moa_id):
            raise ValidationError(f"a MoA class is missing from {name}")
    if scaffold:
        if (manifest.groupby("scaffold").split.nunique() > 1).any():
            raise ValidationError("scaffold leakage across splits")


def make_split(compounds: pd.DataFrame, observations: pd.DataFrame, *, strategy: str = "compound_disjoint",
               seed: int = 42, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
               attempts: int = 2000) -> tuple[pd.DataFrame, dict[str, Any]]:
    if strategy not in {"compound_disjoint", "scaffold_disjoint"}:
        raise ValidationError("unsupported split strategy")
    if len(fractions) != 3 or not all(np.isfinite(f) and 0 < f < 1 for f in fractions) or not np.isclose(sum(fractions), 1):
        raise ValidationError("three positive split fractions must sum to one")
    if type(seed) is not int or not 0 <= seed < 2**32 or attempts < 1:
        raise ValidationError("invalid split seed or attempt count")
    require_strings(compounds, ["compound_id", "moa_id", "canonical_smiles", "inchikey", "structure_status"])
    if compounds.empty or compounds.compound_id.duplicated().any():
        raise ValidationError("unique nonempty compound table required")
    if not compounds.structure_status.isin(["resolved", "manual_override"]).all():
        raise ValidationError("resolve all compound structures before splitting")
    for row in compounds.itertuples():
        canonical_structure(row.canonical_smiles, row.inchikey)
    if compounds.inchikey.duplicated().any():
        raise ValidationError("duplicate InChIKeys require identity reconciliation before splitting")
    required = ["compound_id", "series_id", "observation_id", "moa_id", "cell_line", "plate_id", "well_id", "site_id", "channel", "replicate_id"]
    require_strings(observations, required)
    if set(compounds.compound_id) != set(observations.compound_id):
        missing = set(compounds.compound_id) - set(observations.compound_id)
        extra = set(observations.compound_id) - set(compounds.compound_id)
        raise ValidationError(f"compound/observation coverage mismatch: {len(missing)} compounds have no indexed trajectories; {len(extra)} unknown compounds")
    if "time_hours" not in observations or not pd.api.types.is_numeric_dtype(observations.time_hours):
        raise ValidationError("time_hours must be numeric")
    if not np.isfinite(observations.time_hours).all() or (observations.time_hours < 0).any():
        raise ValidationError("time_hours must be finite and nonnegative")
    if (observations.groupby("series_id")[required[:-1] + ["replicate_id"]].nunique().drop(columns=["observation_id"]) > 1).any().any():
        raise ValidationError("trajectory identity changes within a series")
    physical = ["cell_line", "plate_id", "well_id", "site_id", "channel"]
    if (observations.groupby(physical)[["series_id", "compound_id", "replicate_id"]].nunique() > 1).any().any():
        raise ValidationError("physical trajectory has conflicting identities")
    expected = compounds.set_index("compound_id").moa_id
    if not observations.moa_id.eq(observations.compound_id.map(expected)).all():
        raise ValidationError("observation/compound MoA disagreement")
    validate_trajectories(observations.sort_values(["series_id", "time_hours"]), 2.0)
    table = compounds.sort_values("compound_id").reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)
    table["scaffold"] = table.canonical_smiles.map(scaffold_key)
    assignments: dict[str, str] = {}
    if strategy == "compound_disjoint":
        for moa, group in table.groupby("moa_id", sort=True):
            ids = group.compound_id.to_numpy().copy()
            if len(ids) < 3:
                raise ValidationError(f"MoA {moa} requires at least three compounds")
            rng.shuffle(ids)
            counts = np.floor(np.array(fractions) * len(ids)).astype(int)
            counts = np.maximum(counts, 1)
            while counts.sum() < len(ids):
                counts[np.argmax(np.array(fractions) * len(ids) - counts)] += 1
            while counts.sum() > len(ids):
                candidates = np.where(counts > 1, counts - np.array(fractions) * len(ids), -np.inf)
                counts[np.argmax(candidates)] -= 1
            for name, subset in zip(SPLITS, np.split(ids, np.cumsum(counts)[:-1])):
                assignments.update({key: name for key in subset})
    else:
        counts = table.groupby("moa_id").scaffold.nunique()
        if (counts < 3).any():
            raise ValidationError(f"scaffold split requires at least three scaffolds per MoA: {counts[counts < 3].to_dict()}")
        groups = sorted(table.scaffold.unique())
        best_score = float("inf")
        best: dict[str, str] | None = None
        labels = sorted(table.moa_id.unique())
        for _ in range(attempts):
            candidate = dict(zip(groups, rng.choice(SPLITS, size=len(groups), p=fractions)))
            assigned = table.scaffold.map(candidate)
            counts_by_class = pd.crosstab(table.moa_id, assigned).reindex(index=labels, columns=SPLITS, fill_value=0)
            if (counts_by_class == 0).any().any():
                continue
            actual = counts_by_class.div(counts_by_class.sum(axis=1), axis=0).to_numpy()
            score = float(((actual - np.array(fractions)) ** 2).sum())
            if score < best_score:
                best_score, best = score, candidate
        if best is None:
            raise ValidationError(f"no valid scaffold split found in {attempts} seeded attempts; this does not prove infeasibility")
        assignments = {r.compound_id: best[r.scaffold] for r in table.itertuples()}
    manifest = observations[["compound_id", "series_id"]].drop_duplicates().copy()
    manifest["split"] = manifest.compound_id.map(assignments)
    manifest["split_strategy"] = strategy
    manifest["split_seed"] = seed
    manifest["scaffold"] = manifest.compound_id.map(table.set_index("compound_id").scaffold)
    manifest = manifest.sort_values(["compound_id", "series_id"]).reset_index(drop=True)
    validate_split(manifest, table, scaffold=strategy == "scaffold_disjoint")
    summary = table.assign(split=table.compound_id.map(assignments))
    report = {"strategy": strategy, "seed": seed, "target_fractions": list(fractions),
              "n_compounds": len(table), "n_series": len(manifest), "n_scaffolds": table.scaffold.nunique(),
              "compound_counts": summary.split.value_counts().to_dict(),
              "moa_counts": pd.crosstab(summary.moa_id, summary.split).to_dict(),
              "scaffold_policy": "nonchiral Bemis-Murcko on largest carbon-containing fragment; acyclic compounds share one group",
              "search_attempts": attempts if strategy == "scaffold_disjoint" else 0}
    return manifest, report
