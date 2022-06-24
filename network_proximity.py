import networkx as nx
from statsmodels.nonparametric.bandwidths import bw_silverman  # , bw_scott
from scipy.signal import find_peaks, peak_widths
from sklearn.neighbors import KernelDensity
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from itertools import product
from collections import namedtuple, deque
from utils import profile, tqdm4ray

import logging

logging.basicConfig(level=logging.INFO)
logging_name = "network_proximity"
log = logging.getLogger(logging_name)


################################################################################
#
#                        Drug-Disease Network Proximity
#
################################################################################


shortest_path_length = namedtuple(
    typename="shortest_path", field_names=["source", "target", "length"]
)
# named tuple for storing statistics (mean and std) of a random distribution
reference_distribution_statistics = namedtuple(
    typename="random_distribution_statistics", field_names=["mean", "std"]
)
# named tuple for storing distance and proximity data of a drug
drug_proximity = namedtuple(
    typename="drug_proximity", field_names=["distance", "proximity"],
)
# named tuple for storing distance and distances random distribution data of a drug
drug_distance = namedtuple(
    typename="drug_distance", field_names=["drug", "distance", "random_distribution"],
)

# @profile()
def all_shortest_paths_lengths(
    interactome, disease_genes, drug_targets=None, n_batches=None
):
    """
        For computing all possible shortest paths
        between the drug targets and the disease genes
    """
    log = logging.getLogger("network_proximity:all_shortest_paths_lengths")

    nodes = set(interactome.nodes())
    if drug_targets:
        drug_targets = (
            set(drug_targets) & nodes
        )  # keeps only drug-targets that are in the interactome
    else:
        drug_targets = nodes
    disease_genes = (
        set(disease_genes) & nodes
    )  # keeps only genes related to disease that are in the interactome

    import ray

    @ray.remote
    def shortest_path(network, batch):
        batch_shortest_paths_lengths = tuple(
            shortest_path_length(
                source, target, nx.shortest_path_length(network, source, target)
            )
            if nx.has_path(network, source, target)
            else shortest_path_length(source, target, np.inf)
            for source, target in batch
        )

        return batch_shortest_paths_lengths

    try:
        ray.init()
    except RuntimeError:
        ray.shutdown()
        ray.init()

    all_paths = tuple(product(drug_targets, disease_genes))

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
        desc=f"Computing all possible shortest paths lengths (in {n_batches} batches)",
    ):
        for sp in batch_shortest_paths_lengths:
            shortest_paths_lengths[sp.source][sp.target] = sp.length

    ray.shutdown()

    sources = namedtuple(
        typename="shortest_paths_lengths",
        field_names=[s.replace("-", "_") for s in drug_targets],
    )
    targets = namedtuple(
        typename="targets", field_names=[t.replace("-", "_") for t in disease_genes]
    )
    log.debug("Converting shortest paths lenghts nested dict to nested namedtuple")
    shortest_paths_lengths = sources(
        *[
            targets(**{t.replace("-", "_"): l for t, l in ts.items()})
            for s, ts in shortest_paths_lengths.items()
        ]
    )  # namedtuples are both faster and lighter than dicts

    return shortest_paths_lengths


# @profile()
def compute_distance(
    interactome, drug_targets, disease_genes, shortest_paths_lengths=None, verbose=True,
):
    """
        Compute Distance between a drug and a disease
        (the targets of the drug and the genes related to the disease)

        adapted from by https://github.com/emreg00/toolbox/blob/3d707221092937ed45f13409f8cee38b8dc0316e/wrappers.py#L556
    """
    nodes = set(interactome.nodes())
    disease_genes = (
        set(disease_genes) & nodes
    )  # keeps only genes related to disease that are in the interactome
    drug_targets = (
        set(drug_targets) & nodes
    )  # keeps only drug targets that are in the interactome
    D = interactome.degree(disease_genes)
    if not shortest_paths_lengths:
        shortest_paths_lengths = all_shortest_paths_lengths(
            interactome, disease_genes, drug_targets
        )

    min_paths_from_targets = deque()
    for t in drug_targets:
        paths_from_target = deque()
        tmp_shortest_paths_lengths = getattr(
            shortest_paths_lengths, t.replace("-", "_")
        )  # in this iteration it only searches in paths starting from t
        for g in disease_genes:
            sp = getattr(tmp_shortest_paths_lengths, g.replace("-", "_"))
            if t in disease_genes:
                w = -np.log(D[t] + 1)
            else:
                w = 0
            paths_from_target.append(sp + w)
        min_paths_from_targets.append(np.min(paths_from_target))

    d = np.mean(min_paths_from_targets) if min_paths_from_targets else np.inf

    return d


# @profile()
def get_proximities(
    disease_name,
    disease_genes,
    shortest_paths_lengths=None,
    reps=10000,
    seed=12345,
    verbose=False,
):
    """
        Calculates Drug-Disease Proximity
        (Normalizes Distance)
    """

    log = logging.getLogger("network_proximity:get_all_proximities")

    from databases import DrugBank

    drugbank = DrugBank()

    drugbank_drug_targets = {
        k: list(ts)
        for k, ts in {
            d.id: {
                t.symbol
                for t in d.targets
                if t.type == "protein" and t.organism == "Humans" and t.symbol != None
            }
            for d in drugbank.database
        }.items()
        if ts
    }

    interactome = nx.from_pandas_edgelist(
        pd.read_csv(
            "data/sources/interactome_slim.tsv.gz", sep="\t", compression="gzip"
        ),
        source="source",
        target="target",
    )
    interactome = interactome.to_undirected()

    if not shortest_paths_lengths:
        shortest_paths_lengths = all_shortest_paths_lengths(interactome, disease_genes)

    import ray

    try:
        ray.init()
    except RuntimeError:
        ray.shutdown()
        ray.init()

    @ray.remote
    def get_distances(
        drug,
        interactome,
        drug_targets,
        disease_genes,
        shortest_paths_lengths,
        reps=10000,
        seed=12345,
        verbose=False,
    ):
        if not isinstance(reps, int):
            try:
                reps = int(reps)
            except ValueError:
                raise TypeError(f"reps: {reps}, is not of type int")

        nodes = set(interactome.nodes())
        disease_genes = (
            set(disease_genes) & nodes
        )  # keeps only genes related to disease that are in the interactome
        drug_targets = (
            set(drug_targets) & nodes
        )  # keeps only drug targets that are in the interactome

        distance = compute_distance(
            interactome,
            drug_targets,
            disease_genes,
            shortest_paths_lengths=shortest_paths_lengths,
            verbose=verbose,
        )

        rng = np.random.default_rng(seed)  # set random seed
        random_distances = np.array(
            [
                compute_distance(
                    interactome,
                    rng.choice(tuple(nodes), len(drug_targets), replace=False),
                    disease_genes,
                    shortest_paths_lengths=shortest_paths_lengths,
                    verbose=verbose,
                )
                for _ in tqdm(
                    range(reps),
                    desc="Computing reference distance distribution",
                    disable=not verbose,
                )
            ]
        )
        return drug_distance(drug, distance, random_distances)

    interactome_id = ray.put(interactome)
    disease_genes_id = ray.put(disease_genes)
    shortest_paths_lengths_id = ray.put(shortest_paths_lengths)

    ids = [
        get_distances.remote(
            drug,
            interactome_id,
            drug_targets,
            disease_genes_id,
            shortest_paths_lengths=shortest_paths_lengths_id,
            reps=reps,
            seed=seed,
            verbose=verbose,
        )
        for drug, drug_targets in drugbank_drug_targets.items()
    ]

    distances = [
        distance
        for distance in tqdm4ray(
            ids,
            desc="Computing Drug Distances and Reference Distribution",
            total=len(drugbank_drug_targets),
        )
    ]

    reference_distribution = np.stack(
        [distance.random_distribution for distance in distances]
    ).reshape((-1,))

    ray.shutdown()

    # with open(
    #     f"data/results/{disease_name.replace(' ', '')}/distances_reference_distribution.txt",
    #     "w",
    # ) as outfile:
    #     for dist in reference_distribution:
    #         outfile.write(f"{dist}\n")

    reference_distribution = reference_distribution[
        (reference_distribution != np.inf)
    ]  # ignore infinite (nodes not connected) paths

    mean = np.mean(reference_distribution)
    std = np.std(reference_distribution)

    log.debug(f"Mean:{mean}, Standard Deviation: {std}")

    proximities = namedtuple(
        typename="proximities",
        field_names=(
            distance.drug for distance in distances if distance.distance != np.inf
        ),  # drugbank_drug_targets.keys()
    )

    drug_proximities = proximities(
        **{
            distance.drug: drug_proximity(
                distance.distance, ((distance.distance - mean) / std)
            )
            for distance in distances
            if distance.distance != np.inf
        }
    )

    # Determine Distance Threshold
    bandwidth = np.mean(
        (
            bw_silverman(
                np.array(
                    [
                        distance.distance
                        for distance in distances
                        if distance.distance != np.inf
                    ]
                )
            ),
            bw_silverman(reference_distribution),
        )
    )
    grid_size = 2000
    grid = np.linspace(
        np.min(
            (
                np.min(
                    np.array(
                        [
                            distance.distance
                            for distance in distances
                            if distance.distance != np.inf
                        ]
                    )
                ),
                np.min(reference_distribution),
            )
        ),
        np.max(
            (
                np.max(
                    np.array(
                        [
                            distance.distance
                            for distance in distances
                            if distance.distance != np.inf
                        ]
                    )
                ),
                np.max(reference_distribution),
            )
        ),
        grid_size,
    )
    distances_kde = np.exp(
        KernelDensity(bandwidth=bandwidth, kernel="gaussian", rtol=1 / grid_size)
        .fit(
            np.array(
                [
                    distance.distance
                    for distance in distances
                    if distance.distance != np.inf
                ]
            )[:, np.newaxis]
        )
        .score_samples(grid[:, np.newaxis])
    )
    reference_kde = np.exp(
        KernelDensity(bandwidth=bandwidth, kernel="gaussian", rtol=1 / grid_size)
        .fit(reference_distribution[:, np.newaxis])
        .score_samples(grid[:, np.newaxis])
    )
    peaks, peaks_properties = find_peaks(
        reference_kde, height=(np.mean(reference_kde), None)
    )  # find all peaks in reference_kde above average
    widths, width_heights, left_ips, right_ips = peak_widths(
        reference_kde,
        peaks[[0]],
        rel_height=1
        - np.mean(reference_kde).item()
        / peaks_properties["peak_heights"][0],  # leftmost peak
    )
    threshold = round(
        grid[left_ips.astype(int)].item(), 2
    )  # set threshold to the distance at the beginning of the leftmost peak in kde

    # Save results
    results_df = (
        pd.DataFrame(drug_proximities, index=drug_proximities._fields)
        .rename(columns={"distance": "Distance", "proximity": "Proximity"})
        .rename_axis("DrugBank_ID")
        .merge(
            pd.DataFrame(
                ((d.id, d.name) for d in drugbank.database),
                columns=("DrugBank_ID", "DrugBank_Name"),
            ).set_index("DrugBank_ID"),
            how="left",
            on="DrugBank_ID",
        )
        .sort_values("Proximity")
    )

    with open(
        f"data/results/{disease_name.replace(' ', '')}/network_proximities.tsv", "w"
    ) as outfile:
        outfile.write(
            f"# Reference distribution mean: {mean} and standard deviation: {std}\n"
        )
        outfile.write(f"# Threshold: {threshold}\n")
        outfile.write(
            f"# {round((len(reference_distribution[reference_distribution < threshold]) / len(reference_distribution))*100,2)}% of reference distribution below threshold, {round(len([d.distance for d in drug_proximities if d.distance < threshold]) / len([d.distance for d in drug_proximities])*100,2)}% of distances distribution below threshold\n"
        )
        results_df.to_csv(outfile, sep="\t")

    # distributions_df = pd.DataFrame(
    #     {
    #         "Distance": [d.distance for d in drug_proximities]
    #         + reference_distribution.tolist(),
    #         "Distribution": ["Drugs"] * len(drug_proximities)
    #         + [
    #             f"Reference\n(mean:{round(mean, 3)},\nstandard deviation:{round(std,3)})"
    #         ]
    #         * len(reference_distribution),
    #     }
    # )

    # Draw plot distributions of distances
    plt.plot(
        grid,
        distances_kde,
        linewidth=1,
        color=sns.color_palette("colorblind")[0],
        label="Drugs",
    )
    plt.plot(
        grid,
        reference_kde,
        color=sns.color_palette("colorblind")[3],
        linestyle="--",
        linewidth=1,
        label="Reference",
    )
    plt.axvline(
        threshold, color=sns.color_palette("colorblind")[2], linewidth=1.5, zorder=10,
    )
    plt.axvspan(
        np.min(grid), threshold, alpha=0.1, facecolor=sns.color_palette("colorblind")[2]
    )
    plt.legend()
    plt.title("Distance Distributions")
    plt.xlabel("Distance")
    plt.ylabel("Density")

    plt.savefig(
        f"data/results/{disease_name.replace(' ', '')}/distances_distributions.svg",
        format="svg",
        bbox_inches="tight",
    )
    plt.close()

    return results_df


if __name__ == "__main__":
    # from data.sources.Alzheimer.get_alzheimer_related_genes import alzheimer_genes
    from data.sources.Huntington.get_huntington_related_genes import huntington_genes
    from data.sources.MultipleSclerosis.get_multiple_sclerosis_related_genes import (
        multiple_sclerosis_genes,
    )

    for disease_name, disease_genes in (
        # ("Alzheimer", alzheimer_genes.keys()),
        ("Huntington", huntington_genes.keys()),
        ("Multiple Sclerosis", multiple_sclerosis_genes.keys()),
    ):
        log.info(f"Computing Drugs Proximities for {disease_name}")

        proximities = get_proximities(disease_name, disease_genes)
