#!/usr/bin/env python3
# LucaMenestrina (2022), https://github.com/LucaMenestrina

################################################################################
#
# This is the main file launching an automated pipeline
# that provides plausible drugs for repurposing and prioritizes their combinations
# as described in the paper L. Menestrina, M. Recanatini, Unsupervised Computational Framework for Drug Repurposing: Application to two Neurological Disorders
# Available from the GitHub repository: https://github.com/LucaMenestrina/UnsupervisedComputationalFrameworkForDrugRepurposing
# It is provided under MIT license (see the LINCESE file)
# This script is written in python3 (not fully compatible with python2)
# It can be launched using 'python run.py' (for reproducing the papers analysis) or as below:
#
################################################################################

"""
usage: run.py [-h] [-d DISEASE] [-g GENES [GENES ...]] [-gf GENES_FILE]

Integrated Network Analysis for Drug Repurposing and Combinations Prioritization

optional arguments:
  -h,  --help                                       show this help message and exit
  -d,  --disease DISEASE                            Disease Name
  -g,  --genes GENES [GENES ...]                    Disease-Related Genes (separated by a space)
  -gf, --genes_file GENES_FILE                      File Containing Disease-Related Genes (one per line)
  -cl, --cell_lines CELL_LINES [CELL_LINES ...]     Base Disease-Related Cell Lines in LINCS Database (separated by a space)

Specify only one of 'genes' and 'genes_file'
If no Disease Name is provided the analysis will be performed for Huntington's disease and multiple sclerosis as in the paper
A logfile 'run.log' will be saved
"""


import argparse
import pandas as pd
import os
import sys
import numpy as np
import re

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(
        "run.log", mode="a"), logging.StreamHandler()],
)
logging_name = "run"
log = logging.getLogger(logging_name)


class analysis:
    """
        Drug Candidates Identification by Integrated Network Analysis
    """

    def __init__(self, disease_name, disease_genes, cell_lines):
        log = logging.getLogger("run:analysis")

        self.__disease_name = disease_name
        self.__disease_genes = disease_genes
        self.__cell_lines = cell_lines

        log.info(f"Analyzing {self.__disease_name}\n")
        log.debug(
            f"Disease-related genes: {', '.join(self.__disease_genes)}\n")
        log.debug(f"Cell lines: {', '.join(self.__cell_lines)}\n")

        try:
            os.makedirs(
                f"data/results/{disease_name.replace(' ', '')}", exist_ok=True)
            log.info(
                f"Performing Gene Ontology Enrichment Analysis for {self.__disease_name}"
            )
            from GOEA import GOEA
            from utils import get_human_gene_ids

            hsa_gene_ids = get_human_gene_ids()
            self.__goea_results = GOEA(
                hsa_gene_ids=hsa_gene_ids,
                disease_genes_ids=self.__disease_genes.values(),
                disease_name=self.__disease_name,
            )

            log.info(
                f"Performing Human Phenotype Ontology Enrichment Analysis for {self.__disease_name}"
            )
            from HPOEA import HPOEA

            self.__hpoea_results = HPOEA(
                hsa_gene_ids=hsa_gene_ids,
                disease_genes_ids=self.__disease_genes.values(),
                disease_name=self.__disease_name,
            )

            log.info("Preparing Interactome")
            from interactome import PPI

            self.__ppi = PPI()
            self.__interactome = self.__ppi.interactome

            log.info(f"Computing Drugs Proximities for {self.__disease_name}")

            from network_proximity import get_proximities

            self.__network_proximity_results = get_proximities(
                self.__disease_name, self.__disease_genes.keys()
            )
            # self.__network_proximity_results = pd.read_csv(
            #     f"data/results/{self.__disease_name.replace(' ', '')}/network_proximities.tsv", sep="\t", comment="#")

            with open(
                f"data/results/{self.__disease_name.replace(' ', '')}/network_proximities.tsv"
            ) as infile:
                infile.readline()
                self.__distance_threshold = float(
                    re.findall("\# Threshold: ([0-9]\.[0-9]{1,2})", infile.readline())[
                        0
                    ]
                )

            log.info(
                f"Performing Inverted Gene Set Enrichment Analysis for {self.__disease_name}"
            )
            from IGSEA import IGSEA
            from databases import LINCS

            lincs = LINCS(base_cell_lines=self.__cell_lines, batch_size=2500)
            self.__igsea_results = IGSEA(
                self.__disease_name,
                {str(id) for id in self.__disease_genes.values()},
                lincs,
            )
            # self.__igsea_results = pd.read_csv(
            #     f"data/results/{self.__disease_name.replace(' ', '')}/IGSEA_results.tsv", sep="\t", comment="#")

            self.__proximal_igsea_results = self.__igsea_results[
                self.__igsea_results["DrugBank_ID"].isin(
                    set(
                        self.__network_proximity_results[
                            self.__network_proximity_results["Distance"]
                            < self.__distance_threshold
                        ].index
                    )
                )
            ].copy()

            self.__significant_proximal_igsea_results = self.__proximal_igsea_results[
                self.__proximal_igsea_results["FDR"].apply(
                    lambda fdr: True
                    if isinstance(fdr, str) and "<" in fdr
                    else float(fdr) < 0.25
                )
            ]
            self.__significant_proximal_igsea_results = self.__significant_proximal_igsea_results.merge(
                self.__network_proximity_results.reset_index()[
                    ["DrugBank_ID", "Distance", "Proximity"]
                ],
                how="left",
                on="DrugBank_ID",
            )

            # self.__significant_proximal_igsea_results.to_csv(
            #     f"data/results/{self.__disease_name}/significant_proximal_IGSEA_results.tsv",
            #     sep="\t",
            # )

            log.info(
                f"Selecting Promising Drug Candidates for {self.__disease_name}")

            self.__promising_drug_candidates = self.__significant_proximal_igsea_results.drop_duplicates(
                subset=["DrugBank_ID"], ignore_index=True
            ).copy()
            from databases import DrugBank

            drugbank = DrugBank()
            self.__promising_drug_candidates[
                "Targets"
            ] = self.__promising_drug_candidates["DrugBank_ID"].apply(
                lambda id: ", ".join(
                    {
                        target.symbol
                        for target in drugbank.get(id).targets
                        if target.type == "protein" and target.symbol
                    }
                )
            )
            self.__promising_drug_candidates[
                "Indication"
            ] = self.__promising_drug_candidates["DrugBank_ID"].apply(
                lambda id: drugbank.get(id).indication
            )
            self.__promising_drug_candidates = self.__promising_drug_candidates[
                [
                    "DrugBank_ID",
                    "DrugBank_Name",
                    "Indication",
                    "Targets",
                    "Proximity",
                    "Normalized_Enrichment_Score",
                    "FDR",
                ]
            ].sort_values(
                by=["FDR", "Proximity"],
                ascending=True,
                key=np.vectorize(
                    lambda value: 0.0
                    if isinstance(value, str) and "<" in value
                    else float(value)
                ),
                ignore_index=True,
            )

            self.__promising_drug_candidates.to_csv(
                f"data/results/{self.__disease_name.replace(' ', '')}/promising_drug_candidates.tsv",
                sep="\t",
            )

            from drug_target_gene import (
                draw_drug_target_gene_network,
                draw_gene_target_drug_sankey,
            )

            draw_drug_target_gene_network(
                self.__disease_name, self.__disease_genes.keys()
            )
            draw_gene_target_drug_sankey(
                self.__disease_name, self.__disease_genes.keys()
            )

            from drug_combinations import study_drug_combinations

            study_drug_combinations(self.__disease_name)

            print("\n")
            log.info(f"Analysis of {disease_name} Finished!\n")
        except Exception as error:
            log.error(f"{error}\nAnalysis of {disease_name} FAILED!\n")

    @property
    def disease_name(self):
        return self.__disease_name

    @property
    def disease(self):
        return self.__disease_name

    @property
    def disease_genes(self):
        return self.__disease_genes

    @property
    def cell_lines(self):
        return self.__cell_lines

    @property
    def goea_results(self):
        return self.__goea_results

    @property
    def hpoea_results(self):
        return self.__hpoea_results

    @property
    def network_proximity_results(self):
        return self.__network_proximity_results

    @property
    def distance_threshold(self):
        return self.__distance_threshold

    @property
    def igsea_results(self):
        return self.__igsea_results

    @property
    def proximal_igsea_results(self):
        return self.__proximal_igsea_results

    @property
    def significant_proximal_igsea_results(self):
        return self.__significant_proximal_igsea_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Integrated Network Analysis for Drug Repurposing and Combinations Prioritization",
        epilog="Specify only one of 'genes' and 'genes_file'\nIf no Disease Name is provided the analysis will be performed for Huntington's disease and multiple sclerosis as in the paper\nA logfile 'run.log' will me saved",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "-d", "--disease", type=str, nargs="+", help="Disease Name", dest="disease"
    )
    parser.add_argument(
        "-g",
        "--genes",
        type=str,
        nargs="+",
        help="Disease-Related Genes (separated by a space)",
        default=None,
        dest="genes",
    )
    parser.add_argument(
        "-gf",
        "--genes_file",
        type=str,
        help="File Containing Disease-Related Genes (one per line)",
        default=None,
        dest="genes_file",
    )
    parser.add_argument(
        "-cl",
        "--cell_lines",
        type=str,
        nargs="+",
        help="Base Disease-Related Cell Lines in LINCS Database (separated by a space)",
        default=None,
        dest="cell_lines",
    )
    args = parser.parse_args()
    disease_name = " ".join(args.disease) if args.disease else None

    if not disease_name:
        print(
            "\nNo Disease Name specified, the analysis will be performed for Huntington's Disease and Multiple Sclerosis as in the paper"
        )
        # from data.sources.Alzheimer.get_alzheimer_related_genes import alzheimer_genes
        from data.sources.Huntington.get_huntington_related_genes import (
            huntington_genes,
        )
        from data.sources.MultipleSclerosis.get_multiple_sclerosis_related_genes import (
            multiple_sclerosis_genes,
        )

        for disease_name, disease_genes, cell_lines in (
            # ("Alzheimer", alzheimer_genes, ["NEU", "NPC"]),
            ("Huntington", huntington_genes, [
             "NEU", "NPC", "SHSY5Y", "SKB", "SKL"]),
            (
                "Multiple Sclerosis",
                multiple_sclerosis_genes,
                [
                    "NEU",
                    "NPC",
                    "SHSY5Y",
                    "HL60",
                    "JURKAT",
                    "NOMO1",
                    "PL21",
                    "SKM1",
                    "THP1",
                    "U937",
                    "WSUDLCL2",
                ],
            ),
        ):

            results = analysis(disease_name, disease_genes, cell_lines)
    else:
        if os.path.isdir(f"data/results/{disease_name.replace(' ', '')}"):
            confirm = "wrong answer"
            while confirm.lower() not in ["yes", "y", "no", "n", ""]:
                try:
                    confirm = input(
                        f"It seems {disease_name} was already analyzed, are you sure you want analyze it again?\nIt will be impossible to recover previous data\n\tType: yes or no (y or n)\n"
                    )
                    if confirm.lower() in ["yes", "y", ""]:
                        print("Overwriting old data ...\n")
                        # os.rmdir(f"data/results/{disease_name.replace(' ', '')}")  # No need to remove, data will be overwritten
                    elif confirm.lower() in ["no", "n"]:
                        sys.exit("Aborting ...\n")
                    else:
                        print("\nPlease type: yes or no (y or n)")
                except ValueError:
                    print("\nInput not recognized, reenter please:")

        log.info(f"Checking Inputs for {disease_name}")

        if args.genes_file:
            if args.genes:
                log.error(
                    f"Both 'genes' and 'genes_file' provided, specify only one")
                sys.exit("Aborting ...\n")
            else:
                try:
                    disease_genes = []
                    with open(f"{args.genes_file}", "r") as infile:
                        for line in infile:
                            disease_genes.append(line.strip())
                    disease_genes = set(disease_genes)
                    log.info(
                        f"Using {len(disease_genes)} genes found in {args.genes_file}"
                    )
                except FileNotFoundError:
                    log.error(
                        f"Unable to read genes from {args.genes_file}, file not found"
                    )
                    sys.exit("Aborting ...\n")
                except:
                    log.error(
                        f"Unable to read genes from {args.genes_file}, check that they are one per line"
                    )
                    sys.exit("Aborting ...\n")
        else:
            if args.genes:
                log.info(f"Using {len(args.genes)} disease-related genes")
                disease_genes = set(args.genes)
            else:
                log.error(
                    f"None of 'genes' or 'genes_file' provided, specify one")
                sys.exit("Aborting ...\n")
        if args.cell_lines is None:
            log.error(
                f"No 'cell_lines' provided, specify at least one")
            sys.exit("Aborting ...\n")

        from databases import NCBI

        ncbi = NCBI()

        disease_genes = {
            ncbi.check_symbol(gene): ncbi.get_id_by_symbol(gene)
            for gene in disease_genes
            if ncbi.get_id_by_symbol(gene) and ncbi.check_symbol(gene)
        }

        log.info(
            f"{len(disease_genes)} genes kept after mapping to official symbols")
        log.info(
            f"Saving them in data/sources/{disease_name.replace(' ', '')}/{disease_name.lower().replace(' ','_')}_genes.tsv"
        )
        os.makedirs(
            f"data/sources/{disease_name.replace(' ', '')}", exist_ok=True)
        pd.DataFrame(
            ((symbol, id) for symbol, id in disease_genes.items()),
            columns=["geneSymbol", "geneId"],
        ).to_csv(
            f"data/sources/{disease_name.replace(' ', '')}/{disease_name.lower().replace(' ','_')}_genes.tsv",
            sep="\t",
        )
        results = analysis(disease_name, disease_genes, args.cell_lines)
        )
            os.makedirs(
        f"data/sources/{disease_name.replace(' ', '')}", exist_ok = True)
            pd.DataFrame(
        ((symbol, id) for symbol, id in disease_genes.items()),
        columns = ["geneSymbol", "geneId"],
        ).to_csv(
        f"data/sources/{disease_name.replace(' ', '')}/{disease_name.lower().replace(' ','_')}_genes.tsv",
        sep = "\t",
        )
            results=analysis(disease_name, disease_genes, args.cell_lines)
