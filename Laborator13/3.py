import arviz as az
import matplotlib.pyplot as plt


centered_data = az.load_arviz_data("centered_eight")
non_centered_data = az.load_arviz_data("non_centered_eight")

# contorizeaza numarul de divergente pentru fiecare model
divergences_centered = centered_data.sample_stats.diverging.sum()
divergences_non_centered = non_centered_data.sample_stats.diverging.sum()

print("\nNumărul de divergențe pentru modelul centrat:", divergences_centered)
print("Numărul de divergențe pentru modelul necentrat:", divergences_non_centered)

# Vizualizați divergențele în spațiul parametrilor mu și tau folosind az.plot_pair
_, ax = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(10, 5), constrained_layout=True)

for idx, tr in enumerate([centered_data, non_centered_data]):
    az.plot_pair(
        tr,
        var_names=['mu', 'tau'],
        kind='scatter',
        divergences=True,
        divergences_kwargs={'color': 'C1'},
        ax=ax[idx]
    )

    ax[idx].set_title(['Centrat', 'Necentrat'][idx])

plt.show()
