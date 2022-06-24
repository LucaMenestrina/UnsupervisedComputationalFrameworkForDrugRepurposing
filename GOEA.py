import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from goatools.obo_parser import GODag
from goatools.goea.go_enrichment_ns import GOEnrichmentStudyNS

import logging

logging.basicConfig(level=logging.INFO)
logging_name = "GOEA"
log = logging.getLogger(logging_name)


################################################################################
#
#                       Gene Ontology Enrichment Analysis
#
################################################################################


from utils import get_human_gene_ids  # , profile

hsa_gene_ids = get_human_gene_ids()

from databases import NCBI, GO

ncbi = NCBI()
go = GO(basic=True)


def get_gene2go(filepath="data/sources/GO/goa_human.gaf.gz"):
    import os

    if os.path.isfile(filepath):
        from tqdm import tqdm

        goa_columns = [
            "DB",
            "DB Object ID",
            "DB Object Symbol",
            "Qualifier",
            "GO ID",
            "DB Reference",
            "Evidence Code",
            "With (or) From",
            "Aspect",
            "DB Object Name",
            "DB Object Synonym",
            "DB Object Type",
            "Taxon",
            "Date",
            "Assigned By",
            "Annotatoin Extension",
            "Gene Product From ID",
        ]

        goa = pd.read_csv(
            "data/sources/GO/goa_human.gaf.gz",
            sep="\t",
            skiprows=41,
            names=goa_columns,
            dtype="string",
            usecols=["DB Object Symbol", "DB Object Synonym", "GO ID", "Aspect"],
        )

        # needs some explaination
        gene2go = {
            {"P": "BP", "F": "MF", "C": "CC"}[namespace]: {
                ncbi.get_id_by_symbol(
                    ncbi.check_symbol(
                        symbol, str(list(set(group["DB Object Synonym"]))[0]).split("|")
                    )
                ): set(group["GO ID"])
                for symbol, group in tqdm(grouped_namespace.groupby("DB Object Symbol"))
                if ncbi.check_symbol(
                    symbol, str(list(set(group["DB Object Synonym"]))[0]).split("|")
                )
            }
            for namespace, grouped_namespace in goa.groupby("Aspect")
        }

        return gene2go

    else:
        log.error(f"File {filepath} not found")


gene2go = get_gene2go()

# @profile
class GOEA:
    """
    Gene Set Enrichment Analysis on Gene Ontology

    Adapted from https://github.com/tanghaibao/goatools/blob/main/notebooks/goea_nbt3102.ipynb
    """

    def __init__(
        self,
        hsa_gene_ids,
        disease_genes_ids,
        disease_name,
        significance_treshold=1e-4,
        propagate_counts=False,
    ):

        self.__disease_name = disease_name

        goeaobj = GOEnrichmentStudyNS(
            hsa_gene_ids,  # List of human protein-coding genes
            gene2go,
            GODag("data/sources/GO/go-basic.obo"),  # Ontologies
            propagate_counts=propagate_counts,  # improves sensitivity
            alpha=0.05,  # default significance cut-off
            methods=[
                "fdr_bh",
                "bonferroni",
            ],  # multipletests Benjamini–Hochberg and Bonferroni methods
            pvalcalc="fisher_scipy_stats",
        )

        self.__results = goeaobj.run_study(disease_genes_ids)

        self.__results_sig = [
            r for r in self.__results if r.p_fdr_bh < significance_treshold
        ]
        # for saving GOEApy standard output
        # goeaobj.wr_txt(
        #     f"data/results/{disease_name.replace(' ', '')}/GOEA_results.txt",
        #     self.__results_sig,
        # )

        for namespace in [
            "biological_processes",
            "molecular_functions",
            "cellular_components",
        ]:
            setattr(
                self,
                f"_{self.__class__.__name__}__{''.join([word[0].upper() for word in namespace.split('_')])+'s'}",
                (
                    pd.DataFrame(
                        [
                            (
                                r.GO,
                                r.name,
                                r.study_count,
                                r.ratio_in_study[0] / r.ratio_in_study[1],
                                (r.ratio_in_study[0] / r.ratio_in_study[1])
                                / (r.ratio_in_pop[0] / r.ratio_in_pop[1]),
                                r.p_uncorrected,
                                r.p_fdr_bh,
                                -np.log10(r.p_fdr_bh),
                                r.p_bonferroni,
                                ", ".join(
                                    {ncbi.get_symbol_by_id(id) for id in r.study_items}
                                ),
                            )
                            for r in self.__results_sig
                            if r.NS
                            == "".join(
                                [word[0].upper() for word in namespace.split("_")]
                            )
                        ],
                        columns=[
                            "GO_ID",
                            "Name",
                            "Gene Count",
                            "Gene Ratio",
                            "Fold Enrichment",
                            "p-value",
                            "FDR",  # False Discovery Rate, p-value corrected by Benjamini-Hochberg multitest method
                            "-log10 FDR",
                            "FWER",  # Family-Wise Error Rate, p-value corrected by Bonferroni multitest method
                            "Associated Genes",
                        ],
                    )
                    .sort_values("Fold Enrichment", ascending=False)
                    .reset_index(drop=True)
                ),
            )
            getattr(self, namespace)().to_csv(
                f"data/results/{self.__disease_name.replace(' ', '')}/{namespace.title().replace('_', '')}_GOEA_results.tsv",
                sep="\t",
                index=False,
            )

            # on init automatically saves bubbleplots for the top 20 terms in each namespace
            self.plot(namespace)

    @property
    def results(self):
        return self.__results

    @property
    def results_sig(self):
        return self.__results_sig

    def biological_processes(self, ntop=None):
        if ntop:
            return self.__BPs.copy()[:ntop]
        else:
            return self.__BPs.copy()

    def molecular_functions(self, ntop=None):
        if ntop:
            return self.__MFs.copy()[:ntop]
        else:
            return self.__MFs.copy()

    def cellular_components(self, ntop=None):
        if ntop:
            return self.__CCs.copy()[:ntop]
        else:
            return self.__CCs.copy()

    def plot(self, namespace, ntop=20):
        sns.scatterplot(
            data=getattr(self, namespace.replace(" ", ""))(ntop=ntop),
            x="Fold Enrichment",
            y="Name",
            size="Gene Ratio",
            hue="-log10 FDR",
            palette="coolwarm",
            sizes=(20, 150),
        )
        plt.title(f"Top 20 Enriched {namespace.replace('_', ' ').title()}")
        plt.xlabel("Fold Enrichment")
        plt.ylabel("")
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
        plt.savefig(
            f"data/results/{self.__disease_name.replace(' ', '')}/{namespace.title().replace('_', '')}_GOEA_plot.svg",
            format="svg",
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    # from data.sources.Alzheimer.get_alzheimer_related_genes import alzheimer_genes
    from data.sources.Huntington.get_huntington_related_genes import huntington_genes
    from data.sources.MultipleSclerosis.get_multiple_sclerosis_related_genes import (
        multiple_sclerosis_genes,
    )

    for disease_name, disease_genes in (
        # ("Alzheimer", alzheimer_genes.values()),
        ("Huntington", huntington_genes.values()),
        ("Multiple Sclerosis", multiple_sclerosis_genes.values()),
    ):
        log.info(f"Performing Gene Ontology Enrichment Analysis for {disease_name}")

        goea = GOEA(
            hsa_gene_ids=hsa_gene_ids,
            disease_genes_ids=disease_genes,
            disease_name=disease_name,
        )
