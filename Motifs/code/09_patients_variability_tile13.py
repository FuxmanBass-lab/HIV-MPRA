#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
import sys

"""
python3 09_patients_variability.py \
  --selected-patients ../data/patients/tile6_patients_selected.tsv \
  --patient-map ../data/patients/patients_accession_map.tsv \
  --site-presence ../results/motif_grammar_patients/tile6_site_presence_fixedbins.tsv \
  > ../results/patients_variability/patient_tile6_signature_counts.tsv

python3 09_patients_variability_tile13.py \
  --selected-patients ../data/patients/tile13_patients_selected.tsv \
  --patient-map ../data/patients/patients_accession_map.tsv \
  --site-presence ../results/motif_grammar_patients/tile13_site_presence_fixedbins.tsv \
  > ../results/patients_variability/patient_tile13_signature_counts.tsv
"""

def load_selected_patients(path):
    patients = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            base_id = line.split("|", 1)[0]
            patients.append(base_id)
    return patients


def load_patient_to_accessions(path):
    """
    Build:
      - patient -> set(all accessions)
      - patient -> {tile -> set(accessions)}
    so we can later filter to tile 13.
    """
    patient_to_acc = defaultdict(set)
    patient_to_tile_acc = defaultdict(lambda: defaultdict(set))
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            accession = row.get("Accession", "").strip()
            patient_field = row.get("patient_ID", "").strip()
            tile = row.get("tile", "").strip()
            if not accession or not patient_field:
                continue
            for pid in [p.strip() for p in patient_field.replace(",", ";").split(";")]:
                if not pid:
                    continue
                patient_to_acc[pid].add(accession)
                patient_to_tile_acc[pid][tile].add(accession)
    return patient_to_acc, patient_to_tile_acc


def load_accession_signatures(path):
    accession_to_sig = {}
    all_signatures = set()
    with open(path, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            acc = row.get("isolate", "").strip()
            sig = row.get("signature", "").strip()
            if not acc:
                continue
            accession_to_sig[acc] = sig
            if sig:
                all_signatures.add(sig)
    return accession_to_sig, sorted(all_signatures)


def main(args):
    selected_patients = load_selected_patients(args.selected_patients)
    _, patient_to_tile_accessions = load_patient_to_accessions(args.patient_map)
    accession_to_sig, all_sigs = load_accession_signatures(args.site_presence)

    out_writer = csv.writer(sys.stdout, delimiter="\t")
    # extra column to debug the mismatch
    header = ["patient_ID", "total_isolates_tile13", "unmatched_tile13"] + all_sigs
    out_writer.writerow(header)

    for patient in selected_patients:
        tile_acc_dict = patient_to_tile_accessions.get(patient, {})
        # only tile 13 from the patient-accession file
        tile13_accessions = set()
        for tile, accs in tile_acc_dict.items():
            # your file shows: HIV-1:REJO:13:+
            if tile.startswith("HIV-1:REJO:13:"):
                tile13_accessions.update(accs)

        sig_counts = {sig: 0 for sig in all_sigs}
        unmatched = 0

        for acc in tile13_accessions:
            sig = accession_to_sig.get(acc)
            if sig is None:
                # this is the part that’s likely happening now
                unmatched += 1
                continue
            sig_counts[sig] += 1

        row = [
            patient,
            len(tile13_accessions),   # from the patient-accession file ONLY
            unmatched,               # how many of those we couldn’t find in site_presence
        ] + [sig_counts[sig] for sig in all_sigs]
        out_writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count NFKB/REL and SP/KLF signature occurrences per patient (tile 13 only)."
    )
    parser.add_argument(
        "--selected-patients",
        required=True,
        help="tile13_patients_selected.tsv"
    )
    parser.add_argument(
        "--patient-map",
        required=True,
        help="patients_accession_map.tsv"
    )
    parser.add_argument(
        "--site-presence",
        required=True,
        help="tile13_site_presence_fixedbins.tsv"
    )
    args = parser.parse_args()
    main(args)


    