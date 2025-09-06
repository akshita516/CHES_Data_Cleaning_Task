from pathlib import Path
from matplotlib import pyplot as plt

from political_party_analysis.loader import DataLoader
from political_party_analysis.dim_reducer import DimensionalityReducer
from political_party_analysis.estimator import DensityEstimator
from political_party_analysis.visualization import scatter_plot
from tests.test_data_dim_reducer import mock_df


if __name__ == "__main__":

    # 1. Data pre-processing step
    data_loader = DataLoader()
    data_loader.non_features = [
        "party_id", "party", "partyabbrev", "country", "year", "family", "famabbrev",
        "cmp", "parfam", "lhcmp", "lrcmp", "salience", "eastwest", "eu_position", 
        "eu_position_n", "eu_integration", "eu_integration_n", "position", "position_n",
        "imputed", "imputationmethod", "survey", "survey_date"
    ]
    data_loader.preprocess_data()

    processed_data = data_loader.party_data

    # 2. Dimensionality reduction step
    dim_reducer = DimensionalityReducer(mock_df, n_components=2)

    reduced_dim_data = dim_reducer.transform()

    # 3. Plot dim reduced data
    plt.figure()
    splot = plt.subplot()
    scatter_plot(
        reduced_dim_data,
        color="red",
        splot=splot,
        label="dim reduced data",
    )
    plt.title("2D Representation of Political Parties")
    plt.savefig(Path(__file__).parents[1].joinpath("plots", "dim_reduced_data.png"))

    # 4. Density estimation/distribution modelling step
    estimator = DensityEstimator(reduced_dim_data, dim_reducer, processed_data.columns)
    estimator.fit_kde()

    # 5. Plot density estimation results
    plt.figure()
    splot = plt.subplot()
    estimator.plot_density(splot=splot)
    plt.title("Density Estimation of Political Parties")
    plt.savefig(Path(__file__).parents[1].joinpath("plots", "density_estimation.png"))

    # 6. Plot randomly sampled left and right wing parties
    plt.figure()
    splot = plt.subplot()
    estimator.plot_left_right_parties(splot=splot)
    plt.title("Lefty/Righty Parties")
    plt.savefig(Path(__file__).parents[1].joinpath("plots", "left_right_parties.png"))

    # 7. Plot Finnish parties only
    plt.figure()
    splot = plt.subplot()
    estimator.plot_finnish_parties(
        splot=splot,
        original_data=processed_data,
        reduced_data=reduced_dim_data
    )
    plt.title("Finnish Parties in 2D Space")
    plt.savefig(Path(__file__).parents[1].joinpath("plots", "finnish_parties.png"))

    print("Analysis Complete")
