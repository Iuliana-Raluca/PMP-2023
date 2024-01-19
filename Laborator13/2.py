import arviz as az
import matplotlib.pyplot as plt


centered_data = az.load_arviz_data("centered_eight")
non_centered_data = az.load_arviz_data("non_centered_eight")

#  sinteza pentru parametrii mu si tau pentru modelul centrat
centered_summary = az.summary(centered_data, var_names=["mu", "tau"])

#  sinteza pentru parametrii mu si tau pentru modelul necentrat
non_centered_summary = az.summary(non_centered_data, var_names=["mu", "tau"])

print("\nSinteza pentru modelul centrat:")
print(centered_summary)

print("\nSinteza pentru modelul necentrat:")
print(non_centered_summary)

az.plot_autocorr(centered_data, var_names=["mu", "tau"])
plt.suptitle("Autocorelație pentru modelul centrat")
plt.show()

az.plot_autocorr(non_centered_data, var_names=["mu", "tau"])
plt.suptitle("Autocorelație pentru modelul necentrat")
plt.show()
