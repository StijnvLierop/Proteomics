import pandas as pd
from typing import Iterable
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from utils import columns_to_labels, fig2img


def run_TSNE_pure(proteins_per_pure_sample: pd.DataFrame,
                  sample_columns: Iterable[str]) -> Image:

    # Create feature matrix n_samples x n_proteins
    X = proteins_per_pure_sample[sample_columns].T

    # Create label vector of body fluids
    y = columns_to_labels(proteins_per_pure_sample[sample_columns].columns)

    # Run T-SNE
    X_embedded = TSNE(n_components=2).fit_transform(X)

    # Store results in dataframe
    df = pd.DataFrame(X_embedded, columns=['x', 'y'])
    df['body fluid'] = y

    # Visualize results
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.scatterplot(df, x='x', y='y', hue='body fluid', ax=ax)
    plt.title("T-SNE Projection of pure samples")

    # Write to temporary image
    image = fig2img(fig)

    return image