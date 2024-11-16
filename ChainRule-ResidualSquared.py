import numpy as np
import matplotlib.pyplot as plt

slope = 1
observed_height = 181
observed_weight = 69
graphWeight = [0, observed_weight]
datasetNum = 10

def _predicted_height(intercept, weight):
    return intercept + (slope * weight)

def residual(n, weight_val):
    return observed_height - _predicted_height(n, weight_val)

def residualSquared(n, weight_val):
    return residual(n, weight_val)**2

# dResidual**2 / dIntercept = (dResidual**2 / dResidual) * (dResidual / dIntercept)
# dResidual**2 / dIntercept = (Residual * 2) * (dResidual / dIntercept)

# Residual = Observed - (Intercept + (Weight * Slope))
# Residual = Observed - Intercept - (Weight * Slope)

# dIntercept = 1
# -dIntercept = -1

# dResidual / dIntercept = 0 + (-1) + 0
# dResidual / dIntercept = -1 = Slope of Residual vs. Intercept

# dResidual**2 / dIntercept = (Residual * 2) * (-1)
# dResidual**2 / dIntercept = -2 * Residual 
# dResidual**2 / dIntercept = -2 * (Observed - Intercept - (Weight * Slope)) ------ for this to be 0 to get the observed height...

# dResidual**2 / dIntercept = -2 * (Observed = 181 - Intercept? - (69 * 1)) = 0
# dResidual**2 / dIntercept = -2 * (112 - Intercept) = 0
# Intercept = 112

#Using the formulas...
def findWantedIntercept(observedHeight, observedWeight):
    wantedIntercept = observedHeight - (slope * observedWeight)
    return wantedIntercept

correct_intercept = findWantedIntercept(observed_height, observed_weight)
graphHeight = [correct_intercept, observed_height]

# Generate around 100 intercept values centered around the correct intercept
intercept_range = np.linspace(correct_intercept - 10, correct_intercept + 10, 100)

residuals = []
residuals_squared = []

for intercept in intercept_range:
    res = residual(intercept, observed_weight)
    res_squared = residualSquared(intercept, observed_weight)
    
    residuals.append(res)
    residuals_squared.append(res_squared)


# Create a figure with 3 subplots in a row
fig, axs = plt.subplots(1, 4, figsize=(17, 6))

# Set window title
fig.canvas.manager.set_window_title('Analysis of The Chain Rule: Residual Sum of Squares ')

# Plot Height vs. Weight according to correct_intercept
axs[0].plot(graphWeight, graphHeight, marker='o')
axs[0].set_title('Weight vs. Height (Observed)')
axs[0].set_xlabel('Weight')
axs[0].set_ylabel('Height')
axs[0].axhline(y=correct_intercept, color='r', linestyle='--', label = "Intercept Value")
axs[0].axhline(y=0.0, color='g', linestyle='--')
axs[0].axvline(x=0.0, color='g', linestyle='--')
axs[0].legend()

# Plot Residual vs. Intercept
axs[1].plot(intercept_range, residuals, marker='.')
axs[1].set_title('Residual vs. Intercept')
axs[1].set_xlabel('Intercept')
axs[1].set_ylabel('Residual')
axs[1].axhline(y=0.0, color='g', linestyle='--')
axs[1].axvline(x=correct_intercept, color='r', linestyle='--', label='Correct Intercept')
axs[1].legend()

# Plot Residual Squared vs. Intercept
axs[2].plot(residuals, residuals_squared, marker='o')
axs[2].set_title('Residual Squared vs. Residual')
axs[2].set_xlabel('Residual')
axs[2].set_ylabel('Residual Squared')
axs[2].axhline(y=0.0, color='g', linestyle='--')
axs[2].axvline(x=0.0, color='m', linestyle='--', label='Wanted Residual')
axs[2].legend()

# Plot Residual Squared vs. Intercept
axs[3].plot(intercept_range, residuals_squared, marker='o')
axs[3].set_title('Residual Squared vs. Intercept')
axs[3].set_xlabel('Intercept')
axs[3].set_ylabel('Residual Squared')
axs[3].axhline(y=0.0, color='g', linestyle='--')
axs[3].axvline(x=correct_intercept, color='r', linestyle='--', label='Correct Intercept')
axs[3].legend()


# Adjust layout
plt.tight_layout()
plt.show()
