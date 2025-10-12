import pandas as pd
import argparse
import logging

from rdkit import Chem

class DataPreparer:
    def __init__(self, raw_data_file, preprocessed_file, curated_file):
        self.raw_data_file = raw_data_file
        self.preprocessed_file = preprocessed_file
        self.curated_file = curated_file
        logging.info(f"Initialized DataPreparer with raw data file: {self.raw_data_file}")

    def handle_missing_data_and_duplicates(self, smiles_column=None, properties_of_interest=None):
        """Load data, identify the SMILES column, handle missing values, and remove duplicates.
        - Auto-detect the SMILES column if not provided (try 'canonical_smiles', 'smiles', 'SMILES').
        - Keep only rows with non-null SMILES.
        - Rename the SMILES column to 'canonical_smiles' for downstream consistency.
        - If properties_of_interest is provided, keep only ID + 'canonical_smiles' + those properties (if present).
          Otherwise, keep only ID (if present) and 'canonical_smiles' (no dataset-specific defaults).
        """
        try:
            logging.info(f"Loading data from {self.raw_data_file}")
            df = pd.read_csv(self.raw_data_file)

            # Detect the SMILES column
            candidate_smiles = [smiles_column] if smiles_column else ['canonical_smiles', 'smiles', 'SMILES', 'Smiles']
            smiles_col = next((c for c in candidate_smiles if c and c in df.columns), None)
            if smiles_col is None:
                raise ValueError("Could not find a SMILES column. Tried: " + ", ".join([c for c in candidate_smiles if c]))

            logging.info(f"Using '{smiles_col}' as the SMILES column.")

            # Keep only rows with SMILES
            df_clean = df[df[smiles_col].notna()].copy()

            # Normalize name for downstream steps
            if smiles_col != 'canonical_smiles':
                df_clean = df_clean.rename(columns={smiles_col: 'canonical_smiles'})
                smiles_col = 'canonical_smiles'

            # Remove exact duplicates by SMILES text (pre-canonicalization)
            before = len(df_clean)
            df_clean = df_clean.drop_duplicates(subset=[smiles_col])
            logging.info(f"Removed {before - len(df_clean)} duplicate rows based on '{smiles_col}'.")

            # Always keep an ID column if present
            id_candidates = ['molecule_chembl_id', 'mol_id', 'compound_id', 'id', 'ID']
            id_col = next((c for c in id_candidates if c in df_clean.columns), None)

            # Decide which columns to keep
            present_props = [c for c in (properties_of_interest or []) if c in df_clean.columns]
            selected_columns = []
            if id_col: selected_columns.append(id_col)
            selected_columns.append(smiles_col)
            selected_columns.extend([c for c in present_props if c not in selected_columns])

            df_clean = df_clean[selected_columns]
            logging.info(f"Selected columns for preprocessing: {selected_columns}")

            # Save preprocessed data
            df_clean.to_csv(self.preprocessed_file, index=False)
            logging.info(f"Preprocessed data saved to {self.preprocessed_file}")
        except Exception as e:
            logging.error(f"Error during data preprocessing: {e}")
            raise

    def label_bioactivity(self, active_threshold=1000, inactive_threshold=10000):
        """ If a 'standard_value' column exists, label classes; otherwise, skip gracefully.
        - In datasets without bioactivity, this will be a no-op (preprocessed -> curated).
        - If 'standard_value' exists (e.g., ChEMBL downloads), apply labeling using thresholds.
        """
        try:
            logging.info(f"Loading preprocessed data from {self.preprocessed_file}")
            df = pd.read_csv(self.preprocessed_file)

            if 'standard_value' not in df.columns:
                logging.info("No 'standard_value' column found; skipping bioactivity labeling and passing data through.")
                df.to_csv(self.curated_file, index=False)
                logging.info(f"Curated data (pass-through) saved to {self.curated_file}")
                return

            # Label as active / intermediate / inactive
            import numpy as np
            conditions = [
                (df['standard_value'] <= active_threshold),
                (df['standard_value'] > inactive_threshold)
            ]
            choices = ['active', 'inactive']
            labeled = np.select(conditions, choices, default='intermediate')
            df['bioactivity_class'] = pd.Categorical(labeled, categories=['active', 'intermediate', 'inactive'], ordered=True)

            # Keep typical useful columns if present
            keep_cols = [c for c in ['molecule_chembl_id', 'canonical_smiles', 'standard_value', 'bioactivity_class'] if c in df.columns]
            if not keep_cols:
                keep_cols = df.columns.tolist()
            df[keep_cols].to_csv(self.curated_file, index=False)
            logging.info(f"Curated data with labels saved to {self.curated_file}")
        except Exception as e:
            logging.error(f"Error during labeling: {e}")
            raise


    def clean_smiles_column(self, curated_smiles_output, require_neutral_charge=False, prefer_largest_fragment=True):
        """Clean the 'canonical_smiles' column:
        - Canonicalize SMILES with RDKit if available.
        - For multi-fragment SMILES, keep the largest (by heavy-atom count) fragment by default (toggle with prefer_largest_fragment).
        - Drop molecules that fail sanitization; optionally enforce neutral charge.
        - Remove duplicates *after* canonicalization.
        """
        try:
            logging.info(f"Loading curated data from {self.curated_file}")
            df = pd.read_csv(self.curated_file)

            if 'canonical_smiles' not in df.columns:
                raise ValueError("Expected a 'canonical_smiles' column in curated data.")

            def rdkit_canonical(smiles):
                if Chem is None:
                    return smiles, True  # RDKit not available; pass-through
                try:
                    mol = Chem.MolFromSmiles(smiles, sanitize=False)
                    if mol is None:
                        return None, False

                    # Split into fragments if any '.'
                    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False) or []
                    if not frags:
                        frags = [mol]

                    # Sanitize each fragment, optionally pick largest by heavy atom count
                    best = None
                    best_heavy = -1
                    for m in frags:
                        try:
                            Chem.SanitizeMol(m)
                        except Exception:
                            continue
                        heavy = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 1)
                        if not prefer_largest_fragment:
                            best = m
                            best_heavy = heavy
                            break
                        if heavy > best_heavy:
                            best_heavy = heavy
                            best = m
                    if best is None:
                        return None, False

                    # Optional element/charge checks
                    if require_neutral_charge and best.GetFormalCharge() != 0:
                        return None, False

                    can = Chem.MolToSmiles(best, canonical=True)
                    return can, True
                except Exception:
                    return None, False

            # Apply canonicalization row-wise
            cans = []
            keep_mask = []
            for s in df['canonical_smiles'].astype(str):
                c, ok = rdkit_canonical(s)
                cans.append(c)
                keep_mask.append(ok and c is not None)

            if Chem is None:
                logging.warning("RDKit not available; SMILES were not re-canonicalized. Install RDKit for full cleaning.")

            df['canonical_smiles'] = cans
            df = df[keep_mask].copy()
            logging.info(f"After sanitization/canonicalization: {len(df)} rows remain.")

            # Drop duplicates after canonicalization
            before = len(df)
            df = df.drop_duplicates(subset=['canonical_smiles'])
            logging.info(f"Removed {before - len(df)} duplicates after canonicalization.")

            # Save cleaned SMILES to a separate file
            df[['canonical_smiles']].to_csv(curated_smiles_output, index=False)
            logging.info(f"Curated SMILES saved to {curated_smiles_output}")

            # Overwrite curated_file with the cleaned DataFrame too
            df.to_csv(self.curated_file, index=False)
            logging.info(f"Updated curated dataset saved to {self.curated_file}")
        except Exception as e:
            logging.error(f"Error during SMILES cleaning: {e}")
            raise


def main(raw_data_file, preprocessed_file, curated_file, curated_smiles_output,
         active_threshold, inactive_threshold,
         smiles_column=None, properties_of_interest=None, require_neutral_charge=False, prefer_largest_fragment=True):
    """Main function to preprocess, optionally label, and clean SMILES data."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    preparer = DataPreparer(raw_data_file, preprocessed_file, curated_file)
    preparer.handle_missing_data_and_duplicates(smiles_column=smiles_column,
                                                properties_of_interest=properties_of_interest)
    # Label if bioactivity present; otherwise pass-through
    preparer.label_bioactivity(active_threshold=active_threshold, inactive_threshold=inactive_threshold)
    # Canonicalize and filter
    preparer.clean_smiles_column(curated_smiles_output,
                                 require_neutral_charge=require_neutral_charge,
                                 prefer_largest_fragment=prefer_largest_fragment)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and process molecular data (generalized, dataset-agnostic)."
    )
    parser.add_argument('raw_data_file', type=str, help="Input CSV file with raw molecular data.")
    parser.add_argument('preprocessed_file', type=str, help="Output CSV file for preprocessed data (selected columns).")
    parser.add_argument('curated_file', type=str, help="Output CSV file for curated data (labeled if bioactivity present).")
    parser.add_argument('curated_smiles_output', type=str, help="Output CSV file for curated, canonical SMILES.")
    parser.add_argument('--active_threshold', type=float, default=1000, help="(ChEMBL) Threshold for labeling active compounds.")
    parser.add_argument('--inactive_threshold', type=float, default=10000, help="(ChEMBL) Threshold for labeling inactive compounds.")
    parser.add_argument('--smiles_column', type=str, default=None, help="Name of the SMILES column if not auto-detected.")
    parser.add_argument('--properties', type=str, default=None,
                        help="Comma-separated list of property column names to retain.")
    parser.add_argument('--require_neutral_charge', action='store_true',
                        help="If set, drop molecules with non-zero formal charge.")
    parser.add_argument('--prefer_largest_fragment', action='store_true',
                        help="If set, keep the largest fragment by heavy-atom count when multiple fragments are present.")

    args = parser.parse_args()

    props = None
    if args.properties:
        props = [p.strip() for p in args.properties.split(",") if p.strip()]

    main(args.raw_data_file, args.preprocessed_file, args.curated_file, args.curated_smiles_output,
         args.active_threshold, args.inactive_threshold,
         smiles_column=args.smiles_column, properties_of_interest=props,
         require_neutral_charge=args.require_neutral_charge,
         prefer_largest_fragment=args.prefer_largest_fragment)
