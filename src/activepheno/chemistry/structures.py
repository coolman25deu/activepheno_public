"""Offline exact-name resolution with source validation and explicit overrides."""
import re
from pathlib import Path
from typing import Any

import json5
import pandas as pd
from rdkit import Chem, rdBase

from activepheno.data.observations import normalize_name
from activepheno.data.schema import ValidationError, require_strings

CATALOG_SOURCE = "https://github.com/edvinforsgren/LCTP/blob/main/assets/Supplementary_Table_6.html"


def read_catalog(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    else:
        match = re.search(r"const data\s*=\s*(\[.*?\]);", path.read_text(), re.S)
        if not match:
            raise ValidationError("supplementary HTML has no supported const data array")
        # Parse data only. Never execute downloaded JavaScript.
        records = json5.loads(match.group(1), allow_duplicate_keys=False)
        if not isinstance(records, list) or not all(isinstance(r, dict) for r in records):
            raise ValidationError("compound catalog must contain object records")
        frame = pd.DataFrame(records)
    require_strings(frame, ["compound_name_original", "compound_name", "metadata_moa",
                            "smiles", "inchi_key", "reference_url"])
    return frame


def canonical_structure(smiles: str, expected_key: str) -> tuple[str, str]:
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValidationError("invalid SMILES")
    key = Chem.MolToInchiKey(mol)
    if not key or key != expected_key:
        raise ValidationError(f"SMILES/InChIKey disagreement: listed={expected_key}, computed={key}")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True), key


def read_overrides(path: Path | None) -> pd.DataFrame:
    columns = ["compound_name_raw", "canonical_smiles", "inchikey", "structure_source", "reason", "evidence_sha256"]
    if path is None:
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    require_strings(frame, columns)
    if frame.compound_name_raw.duplicated().any():
        raise ValidationError("duplicate override compound_name_raw")
    if not frame.evidence_sha256.map(lambda x: bool(re.fullmatch(r"[0-9a-f]{64}", x))).all():
        raise ValidationError("override evidence_sha256 must be a SHA-256 hex digest")
    return frame


def resolve_compounds(compounds: pd.DataFrame, catalog: pd.DataFrame,
                      overrides: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    require_strings(compounds, ["compound_id", "compound_name_raw", "compound_name_normalized", "moa_id", "moa_name"])
    if compounds.empty or compounds.compound_id.duplicated().any():
        raise ValidationError("compound table must be nonempty with unique IDs")
    unknown = set(overrides.compound_name_raw) - set(compounds.compound_name_raw)
    if unknown:
        raise ValidationError(f"overrides reference unknown compounds: {sorted(unknown)}")
    aliases: dict[str, set[int]] = {}
    catalog = catalog.reset_index(drop=True)
    for index, row in catalog.iterrows():
        for name in [row.compound_name_original, row.compound_name]:
            aliases.setdefault(normalize_name(name), set()).add(index)
    override_lookup = overrides.set_index("compound_name_raw")
    records = []
    for _, row in compounds.sort_values("compound_id").iterrows():
        item = row.to_dict()
        item.update(canonical_smiles=None, inchikey=None, structure_source=None,
                    structure_status="missing", resolution_issue="no exact catalog alias",
                    source_smiles=None, source_inchikey=None, override_reason=None)
        matches = sorted(aliases.get(normalize_name(row.compound_name_raw), set()))
        if len(matches) > 1:
            item.update(structure_status="ambiguous", resolution_issue="multiple catalog matches")
        elif len(matches) == 1:
            candidate = catalog.iloc[matches[0]]
            item.update(source_smiles=candidate.smiles, source_inchikey=candidate.inchi_key,
                        structure_source=f"{CATALOG_SOURCE} | {candidate.reference_url}")
            try:
                if candidate.metadata_moa != row.moa_id:
                    raise ValidationError("catalog/metadata MoA disagreement")
                smiles, key = canonical_structure(candidate.smiles, candidate.inchi_key)
                item.update(canonical_smiles=smiles, inchikey=key,
                            structure_status="resolved", resolution_issue=None)
            except ValidationError as exc:
                item.update(structure_status="ambiguous", resolution_issue=str(exc))
        if row.compound_name_raw in override_lookup.index:
            override = override_lookup.loc[row.compound_name_raw]
            smiles, key = canonical_structure(override.canonical_smiles, override.inchikey)
            item.update(canonical_smiles=smiles, inchikey=key, structure_status="manual_override",
                        structure_source=override.structure_source, override_reason=override.reason,
                        override_evidence_sha256=override.evidence_sha256)
            # Preserve original issue even after an explicit correction.
        records.append(item)
    result = pd.DataFrame(records)
    duplicate_keys = {
        key: group.compound_id.tolist() for key, group in result.dropna(subset=["inchikey"]).groupby("inchikey")
        if len(group) > 1
    }
    counts = result.structure_status.value_counts().to_dict()
    unresolved = result.structure_status.isin(["missing", "ambiguous"])
    report = {"n_compounds": len(result), "rdkit_version": rdBase.rdkitVersion,
              "counts": {s: int(counts.get(s, 0)) for s in ["resolved", "ambiguous", "missing", "manual_override"]},
              "duplicate_inchikeys": duplicate_keys, "ready_for_chemistry": not unresolved.any() and not duplicate_keys,
              "issues": result.loc[result.resolution_issue.notna(), ["compound_id", "compound_name_raw", "structure_status", "resolution_issue", "override_reason"]].to_dict("records"),
              "structure_policy": "source salt/fragment composition and supplied stereochemistry retained; no salt stripping or stereochemistry inference"}
    report["ready_for_chemistry"] = bool(report["ready_for_chemistry"])
    return result, report
