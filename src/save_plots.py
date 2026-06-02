# %%
import os
import matplotlib.pyplot as plt

# %%
def save_current_figure(filename):

    os.makedirs(
        "results/plots",
        exist_ok=True
    )

    plt.savefig(
        f"results/plots/{filename}",
        dpi=300,
        bbox_inches="tight"
    )

    print(
        f"Saved: results/plots/{filename}"
    )