import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from goatools.obo_parser import GODag
from goatools.go_enrichment import GOEnrichmentStudy

import logging

logging.basicConfig(level=logging.INFO)
logging_name = "HPOEA"
log = logging.getLogger(logging_name)


################################################################################
#
#                 Human Phenotype Ontology Enrichment Analysis
#
################################################################################


from utils import get_human_gene_ids  # , profile

hsa_gene_ids = get_human_gene_ids()

from databases import NCBI, HPO

ncbi = NCBI()
hpo = HPO()

gene2hp = hpo.geneId2hpo

# @profile
class HPOEA:
    """
    Gene Set Enrichment Analysis on Human Phenotype Ontology

    Adapted from https://github.com/tanghaibao/goatools/blob/main/notebooks/goea_nbt3102.ipynb
    and
    https://github.com/tanghaibao/goatools/blob/main/notebooks/Enrichment_analyses_human_phenotype_ontology.ipynb
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

        hpoeaobj = GOEnrichmentStudy(
            hsa_gene_ids,  # List of human protein-coding genes
            # Gene2GoReader(
            #     "data/sources/NCBI/gene2go", taxids=[9606]
            # ).get_ns2assc(),  # geneid/GO associations
            gene2hp,
            GODag("data/sources/HPO/hp.obo"),  # Ontologies
            propagate_counts=propagate_counts,
            alpha=0.05,  # default significance cut-off
            methods=[
                "fdr_bh",
                "bonferroni",
            ],  # multipletests Benjamini–Hochberg and Bonferroni methods
            pvalcalc="fisher_scipy_stats",
        )

        self.__results = hpoeaobj.run_study(disease_genes_ids)

        self.__results_sig = [
            r for r in self.__results if r.p_fdr_bh < significance_treshold
        ]
        # for saving GOEApy standard output for HPO
        # hpoeaobj.wr_txt(
        #     f"data/results/{disease_name.replace(' ', '')}/hpoea_results.txt",
        #     self.__results_sig,
        # )

        self.__results_df = (
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
                        ", ".join({ncbi.get_symbol_by_id(id) for id in r.study_items}),
                    )
                    for r in self.__results_sig
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
        )

        self.__results_df.to_csv(
            f"data/results/{self.__disease_name.replace(' ', '')}/HPOEA_results.tsv",
            sep="\t",
            index=False,
        )

        # on init automatically saves bubbleplots for the top 20 terms
        self.plot()

    @property
    def results(self):
        return self.__results

    @property
    def results_sig(self):
        return self.__results_sig

    @property
    def results_df(self):
        return self.__results_df

    def plot(self, ntop=20):
        if ntop:
            data = self.__results_df.head(ntop)
        else:
            data = self.__results_df.copy()
        sns.scatterplot(
            data=data,
            x="Fold Enrichment",
            y="Name",
            size="Gene Ratio",
            hue="-log10 FDR",
            palette="coolwarm",
            sizes=(20, 150),
        )
        plt.title(f"Top 20 Enriched Phenotypes")
        plt.xlabel("Fold Enrichment")
        plt.ylabel("")
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
        plt.savefig(
            f"data/results/{self.__disease_name.replace(' ', '')}/HPOEA_plot.svg",
            format="svg",
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    from data.sources.Huntington.get_huntington_related_genes import huntington_genes
    from data.sources.MultipleSclerosis.get_multiple_sclerosis_related_genes import (
        multiple_sclerosis_genes,
    )
    from data.sources.Alzheimer.get_alzheimer_related_genes import alzheimer_genes

    for disease_name, disease_genes in (
        ("Alzheimer", alzheimer_genes.values()),
        ("Huntington", huntington_genes.values()),
        ("Multiple Sclerosis", multiple_sclerosis_genes.values()),
    ):
        log.info(
            f"Performing Human Phenotype Ontology Enrichment Analysis for {disease_name}"
        )

        hpoea = HPOEA(
            hsa_gene_ids=hsa_gene_ids,
            disease_genes_ids=disease_genes,
            disease_name=disease_name,
        )
        # hpoea.plot()
