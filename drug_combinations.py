import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations, combinations_with_replacement
from collections import namedtuple
from utils import profile, tqdm4ray
import re
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from scipy.cluster import hierarchy

from databases import DrugBank

drugbank = DrugBank()


import logging

logging.basicConfig(level=logging.INFO)
logging_name = "drug_combinations"
log = logging.getLogger(logging_name)


################################################################################
#
#                               Drug Combinations
#
################################################################################


shortest_path_length = namedtuple(
    typename="shortest_path", field_names=["source", "target", "length"]
)

interactome = nx.from_pandas_edgelist(
    pd.read_csv("data/sources/interactome_slim.tsv.gz", sep="\t", compression="gzip"),
    source="source",
    target="target",
)
interactome = interactome.to_undirected()

# @profile()
def all_shortest_paths_lengths(interactome, drug_targets, n_batches=None):
    """
        For computing all possible shortest paths among the drug targets
    """
    log = logging.getLogger("drug_combinations:all_shortest_paths_lengths")

    nodes = set(interactome.nodes())
    drug_targets = set(
        drug_targets
    )  # & nodes would keep only drug-targets that are in the interactome

    import ray

    @ray.remote
    def shortest_path(network, batch):
        batch_shortest_paths_lengths = tuple(
            shortest_path_length(
                source, target, nx.shortest_path_length(network, source, target)
            )
            if (source in nodes and target in nodes)
            and nx.has_path(network, source, target)
            else shortest_path_length(source, target, np.inf)
            for source, target in batch
        )

        return batch_shortest_paths_lengths

    try:
        ray.init()
    except RuntimeError:
        ray.shutdown()
        ray.init()

    all_paths = tuple(combinations_with_replacement(drug_targets, 2))

    log.debug("ray put interactome")
    interactome_id = ray.put(interactome)

    if not n_batches:
        # empirical value (on laptop with a 8 cores cpu and 16 GB of ram)
        # should be a good trade of for performance and memory consumption
        n_batches = int(ray.available_resources()["CPU"]) * 20

    batches_ids = [ray.put(batch) for batch in np.array_split(all_paths, n_batches)]

    ids = [shortest_path.remote(interactome_id, batch_id) for batch_id in batches_ids]

    shortest_paths_lengths = {t: {} for t in drug_targets}
    for batch_shortest_paths_lengths in tqdm4ray(
        ids,
        total=n_batches,
        desc=f"Computing all possible shortest paths lengths between drug targets (in {n_batches} batches)",
    ):
        for sp in batch_shortest_paths_lengths:
            shortest_paths_lengths[sp.source][sp.target] = sp.length
            shortest_paths_lengths[sp.target][sp.source] = sp.length

    ray.shutdown()

    sources = namedtuple(
        typename="shortest_paths_lengths",
        field_names=[s.replace("-", "_") for s in drug_targets],
    )
    targets = namedtuple(
        typename="targets", field_names=[t.replace("-", "_") for t in drug_targets]
    )
    log.debug("Converting shortest paths lenghts nested dict to nested namedtuple")
    shortest_paths_lengths = sources(
        *[
            targets(**{t.replace("-", "_"): l for t, l in ts.items()})
            for s, ts in shortest_paths_lengths.items()
        ]
    )  # namedtuples are both faster and lighter than dicts

    return shortest_paths_lengths


def internal_distance(targets, shortest_paths_lengths):
    """
        Compute mean shortest distance within the interactome between the targets of a drug
    """
    return np.mean(
        [
            np.min(
                [
                    getattr(
                        getattr(shortest_paths_lengths, target1.replace("-", "_")),
                        target2.replace("-", "_"),
                    )
                    for target2 in targets
                    if target1 != target2
                ]
            )
            for target1 in targets
        ]
    )


def between_distance(A_targets, B_targets, shortest_paths_lengths):
    """
        Compute mean shortest distance within the interactome between the targets of two drugs
    """
    return np.mean(
        [
            np.min(
                [
                    getattr(
                        getattr(shortest_paths_lengths, target1.replace("-", "_")),
                        target2.replace("-", "_"),
                    )
                    for target2 in B_targets
                ]
            )
            for target1 in A_targets
        ]
        + [
            np.min(
                [
                    getattr(
                        getattr(shortest_paths_lengths, target1.replace("-", "_")),
                        target2.replace("-", "_"),
                    )
                    for target2 in A_targets
                ]
            )
            for target1 in B_targets
        ]
    )


def get_separation(A, B, shortest_paths_lengths):
    A_targets = {
        target.symbol
        for target in drugbank.get(A).targets
        if target.type == "protein" and target.organism == "Humans" and target.symbol
    }
    B_targets = {
        target.symbol
        for target in drugbank.get(B).targets
        if target.type == "protein" and target.organism == "Humans" and target.symbol
    }
    separation = between_distance(
        A_targets, B_targets, shortest_paths_lengths
    ) - np.mean(
        [
            internal_distance(A_targets, shortest_paths_lengths),
            internal_distance(B_targets, shortest_paths_lengths),
        ]
    )

    return separation


def jaccard(set1, set2):
    u = len(set1.union(set2))
    if u:
        return len(set1.intersection(set2)) / len(set1.union(set2))
    else:
        return 0.0


def draw_heatmap(disease_name, drugs, shortest_paths_length):
    data = pd.DataFrame(
        {
            A: {B: get_separation(A, B, shortest_paths_length) for B in drugs}
            for A in drugs
        }
    )
    np.fill_diagonal(data.values, 0)

    # ATC info
    ATC_codes = {
        drug: {code.level1.code for code in drugbank.get(drug).atc_codes}
        for drug in data.index
    }
    ATC_code_names = {
        code.level1.code: code.level1.code + ": " + code.level1.name.title()
        for drug in data.index
        for code in drugbank.get(drug).atc_codes
    }
    ATC_distance_matrix = pd.DataFrame(
        {
            drug1: {
                drug2: jaccard(codes1, codes2) for drug2, codes2 in ATC_codes.items()
            }
            for drug1, codes1 in ATC_codes.items()
        }
    )
    all_ATC = sorted({code for codes in ATC_codes.values() for code in codes})
    ATC_cmap = dict(
        zip(all_ATC, sns.color_palette("Spectral_r", n_colors=len(all_ATC)))
    )
    # cluster on ATC codes
    Z_ATC = hierarchy.linkage(ATC_distance_matrix, "ward")
    labels = [
        data.index[n] for n in hierarchy.dendrogram(Z_ATC, no_plot=True)["leaves"]
    ]
    data_reordered = data.reindex(labels)[
        labels
    ]  # reorder data on the basis of clusters
    # colors for side bar
    ATC_colors = pd.DataFrame(
        {
            drug: [
                ATC_cmap[code] if code in ATC_codes[drug] else (1, 1, 1)
                for code in all_ATC
            ]
            for drug in data_reordered.index
        },
        index=all_ATC,
    ).T

    # Additional Data
    exposure = pd.DataFrame(
        np.vectorize(lambda s: s < 0)(data_reordered.copy().values),
        index=data_reordered.index,
        columns=data_reordered.index,
    )
    reported_combinations = pd.DataFrame(
        np.array(
            [
                [
                    "\u25CB"  # "O"
                    if B
                    in {
                        drugbank.get(ingredient).id
                        for ingredient in drugbank.get(A).combined_ingredients
                    }
                    or A
                    in {
                        drugbank.get(ingredient).id
                        for ingredient in drugbank.get(B).combined_ingredients
                    }
                    else ""
                    for B in data_reordered.index
                ]
                for A in data_reordered.index
            ]
        ),
        index=data_reordered.index,
        columns=data_reordered.index,
    )
    reported_interactions = np.array(
        [
            [
                "\u00D7"  # "X"
                if B
                in {
                    d.drugbank_id
                    for d in drugbank.get(A).drug_interactions
                    if bool(
                        re.search("The risk o[a-z,A-Z,\ ]+increased", d.description)
                    )
                }
                or A
                in {
                    d.drugbank_id
                    for d in drugbank.get(B).drug_interactions
                    if bool(
                        re.search("The risk o[a-z,A-Z,\ ]+increased", d.description)
                    )
                }
                else ""
                for B in data_reordered.index
            ]
            for A in data_reordered.index
        ]
    )
    annotations = reported_combinations + reported_interactions
    annotations = annotations.apply(
        np.vectorize(
            lambda annot: "\u2A02"
            if annot == "\u25CB\u00D7"
            else annot  # "OX" in case a reported combination has also a reported interaction
        )  # circled cross
    )

    # Initialize Figure
    plt.figure(figsize=(3, 4), dpi=400)

    heatmap = sns.clustermap(
        data,
        cmap="RdYlBu_r",
        cbar_pos=(0.1, -0.01, 0.25, 0.025),
        cbar_kws={"orientation": "horizontal"},
        row_linkage=Z_ATC,
        col_linkage=Z_ATC,
        xticklabels=1,
        yticklabels=1,
        dendrogram_ratio=0.15,
        colors_ratio=0.004,
        row_colors=ATC_colors,
    )
    heatmap.ax_heatmap.set_title("Drug Combinations", fontsize="x-large")
    heatmap.ax_col_dendrogram.set_visible(False)
    heatmap.ax_row_colors.set_xticklabels(
        heatmap.ax_row_colors.get_xmajorticklabels(),
        fontsize=(heatmap.ax_row_colors.bbox.width / len(all_ATC)) * 0.8,
    )

    # Grey Diagonal
    diagonal = sns.heatmap(
        np.identity(len(data)),
        cmap=["silver", "none"],
        cbar=False,  # don't draw colorbar
        mask=np.logical_not(np.identity(len(data))),
        ax=heatmap.ax_heatmap,
    )

    # overwrite heatmap content (for displaying multiple data)
    separation_heatmap = sns.heatmap(
        data_reordered,
        cmap="RdYlBu_r",
        cbar=False,  # don't draw colorbar
        mask=np.triu(np.ones(data_reordered.shape)),
        xticklabels=False,
        yticklabels=False,
        annot=annotations,
        fmt="",  # necessary because labels/annotations are not numerical
        annot_kws={
            "fontsize": ((heatmap.ax_heatmap.bbox.height / len(data_reordered.index)))
            * 0.8
        },
        ax=heatmap.ax_heatmap,
    )
    exposure_heatmap = sns.heatmap(
        exposure,
        cmap=sns.color_palette("Accent", n_colors=2),
        cbar=False,  # don't draw colorbar
        mask=np.tril(np.ones(exposure.shape)),
        xticklabels=1,
        yticklabels=1,
        annot=annotations,
        fmt="",  # necessary because labels/annotations are not numerical
        annot_kws={
            "fontsize": ((heatmap.ax_heatmap.bbox.height / len(data_reordered.index)))
            * 0.8
        },
        ax=heatmap.ax_heatmap,
    )
    exposure_heatmap.set_xticklabels(
        exposure_heatmap.get_xmajorticklabels(),
        fontsize=(heatmap.ax_heatmap.bbox.height / len(data_reordered.index)) * 0.8,
    )
    exposure_heatmap.set_yticklabels(
        exposure_heatmap.get_ymajorticklabels(),
        fontsize=(heatmap.ax_heatmap.bbox.height / len(data_reordered.index)) * 0.8,
    )

    # Legends
    heatmap.ax_cbar.set_title("Separation (s)", fontsize="medium")
    exposure_legend = heatmap.ax_cbar.legend(
        handles=[
            Patch(color=color, label=label)
            for color, label in zip(
                sns.color_palette("Accent", n_colors=2),
                ["Complementary", "Overlapping"],
            )
        ],
        fontsize="small",
        title="Exposure",
        title_fontsize="medium",
        loc="upper center",
        frameon=False,
        borderpad=0,
        borderaxespad=0,
    )
    annotations_legend = heatmap.ax_cbar.legend(
        handles=[
            Patch(color="none", label="\u00D7  Reported Interactions"),  # "X"
            Patch(color="none", label="\u25CB  Reported Combination"),  # "O"
        ],
        fontsize="small",
        title="Annotations",
        title_fontsize="medium",
        loc="upper center",
        frameon=False,
        borderpad=0,
        borderaxespad=0,
    )
    row_colors_legend = heatmap.ax_cbar.legend(
        handles=[
            Patch(color=ATC_cmap[code], label=ATC_code_names[code]) for code in all_ATC
        ],
        fontsize="small",
        title="ATC codes",
        title_fontsize="medium",
        loc="upper left",
        frameon=False,
        borderpad=0,
        borderaxespad=0,
    )
    # add again legends (by default the last one overwrites the previous ones)
    heatmap.ax_cbar.add_artist(exposure_legend)
    heatmap.ax_cbar.add_artist(annotations_legend)
    # Legends positionings
    transFig = (
        lambda tpl: plt.gcf().transFigure.inverted().transform(tpl)
    )  # convenient function for transforming coordinates
    transCbar = lambda tpl: heatmap.ax_cbar.transAxes.inverted().transform(
        tpl
    )  # convenient function for transforming coordinates
    vspacing = (
        (heatmap.ax_heatmap.bbox.height / len(data_reordered.index))
        * 0.9  # instead of 0.8, for a little bit of extra padding
        * 8
        * 0.5  # approx x_labels height (fontsize[*0.9] * 8 chars * approx ratio height/width)
    )
    hpadding = (
        plt.gcf().bbox.width
        - sum([heatmap.ax_cbar.bbox.width, row_colors_legend.get_window_extent().width])
    ) / 3
    heatmap.ax_cbar.set_position(
        np.concatenate([transFig((hpadding, -vspacing)), [0.25, 0.025]])
    )
    row_colors_legend.set_bbox_to_anchor(
        transCbar(
            (
                heatmap.ax_cbar.bbox.width + 2 * hpadding,
                heatmap.ax_cbar.title.get_window_extent().y1,
            )
        )
    )

    vpadding = heatmap.ax_cbar.title.get_window_extent().height * 1.5
    exposure_legend.set_bbox_to_anchor(
        transCbar(
            (
                hpadding + (heatmap.ax_cbar.bbox.width / 2),
                heatmap.ax_cbar.bbox.y0
                - vpadding
                - (
                    heatmap.ax_cbar.get_yaxis().label._fontproperties._size
                    + heatmap.ax_cbar.get_yaxis().labelpad
                ),
            )
        )
    )
    annotations_legend.set_bbox_to_anchor(
        transCbar(
            (
                hpadding + (heatmap.ax_cbar.bbox.width / 2),
                exposure_legend.get_window_extent().y0 - vpadding,
            )
        )
    )

    # Save Figure
    plt.savefig(
        f"data/results/{disease_name.replace(' ', '')}/drug_combinations.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.close()


def study_drug_combinations(disease_name):
    # Retrieve Promising Drug Candidates
    promising_drug_candidates = pd.read_csv(
        f"data/results/{disease_name.replace(' ', '')}/promising_drug_candidates.tsv",
        sep="\t",
        index_col=0,
    ).set_index("DrugBank_ID")
    # Keep only drugs that have at least two targets
    promising_drug_candidates = promising_drug_candidates[
        promising_drug_candidates["Targets"].str.contains(", ")
    ]

    shortest_paths_lengths = all_shortest_paths_lengths(
        interactome,
        {
            target
            for targets in promising_drug_candidates["Targets"]
            for target in targets.split(", ")
        },
    )

    drug_combinations = pd.DataFrame(
        [
            tuple(
                [
                    A,
                    B,
                    drugbank.get(A).name,
                    drugbank.get(B).name,
                    get_separation(A, B, shortest_paths_lengths),
                    ", ".join({code.level1.code for code in drugbank.get(A).atc_codes}),
                    ", ".join({code.level1.code for code in drugbank.get(B).atc_codes}),
                    B
                    in {
                        d.drugbank_id
                        for d in drugbank.get(A).drug_interactions
                        if bool(
                            re.search("The risk o[a-z,A-Z,\ ]+increased", d.description)
                        )
                    }
                    or A
                    in {
                        d.drugbank_id
                        for d in drugbank.get(B).drug_interactions
                        if bool(
                            re.search("The risk o[a-z,A-Z,\ ]+increased", d.description)
                        )
                    },
                    B
                    in {
                        drugbank.get(ingredient).id
                        for ingredient in drugbank.get(A).combined_ingredients
                    }
                    or A
                    in {
                        drugbank.get(ingredient).id
                        for ingredient in drugbank.get(B).combined_ingredients
                    },
                ]
            )
            for A, B in combinations(promising_drug_candidates.index, 2)
        ],
        columns=[
            "ID1",
            "ID2",
            "Name1",
            "Name2",
            "Separation",
            "ATC_Codes1",
            "ATC_Codes2",
            "Reported_Interaction",
            "Reported_Combination",
        ],
    ).sort_values(by="Separation", ascending=False)
    drug_combinations.to_csv(
        f"data/results/{disease_name.replace(' ', '')}/drug_combinations.tsv",
        sep="\t",
        index=False,
    )

    draw_heatmap(disease_name, promising_drug_candidates.index, shortest_paths_lengths)


if __name__ == "__main__":
    for disease_name in ["Huntington", "Multiple Sclerosis"]:  # "Alzheimer"
        log.info(f"Studying Promising Drug Combinations for {disease_name}")
        study_drug_combinations(disease_name)
