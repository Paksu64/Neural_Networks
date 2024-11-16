import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import math


observed_height = [1.4, 1.9, 3.2]
observed_weight = [0.5, 2.3, 2.9]
learning_rate = 0.01
epoch = 1000


def _predicted_height(intercept, n):
    return intercept + temp_slope * observed_weight[n]

def _residual(n, intercept):
    return observed_height[n] - _predicted_height(intercept, n)

def _residual_squared(n, m):
    return _residual(n, m)**2

def _sum_of_squared_residuals(intercept, slope):
    sum = 0
    for i in range(0, len(observed_height)):
        sum += observed_height[i] - (intercept + slope * observed_weight[i])

#Formula of dSumOfSquaredResiduals / dIntercept
def _dsum_of_squares_dintercept(residualValue):
    return -2 * residualValue

#Formula of dSumOfSquaredResiduals / dSlope
def _dsum_of_squares_dslope(residualValue, n):
    return -2 * observed_weight[n] * residualValue

def _round_down(n, decimals = 0):
    multiplier = 10 ** decimals
    return math.floor(n * multiplier) / multiplier

#residual and residual-squared calculation
intercepts = []
intercepts_DEBUG = []

slopes = []
slopes_DEBUG = []

sum_of_squared_residuals_history = []
sum_of_squared_residuals_history_DEBUG = []

temp_intercept = 0.0  # Initial intercept
temp_slope = 1.0 # Initial slope

for _ in range(epoch):
    residuals = []  # Store residuals for *current* epoch
    dsum_of_squared_res_dintercept = 0
    dsum_of_squared_res_dslope = 0

    #Calculating the values according to current given residuals
    for i in range(len(observed_height)):
        residual_value = _residual(i, temp_intercept)
        residuals.append(residual_value)
        dsum_of_squared_res_dintercept += _dsum_of_squares_dintercept(residual_value)
        dsum_of_squared_res_dslope += _dsum_of_squares_dslope(residual_value, i)

    # Update intercept and slope using gradient descent
    temp_intercept -= learning_rate * dsum_of_squared_res_dintercept
    temp_slope -= learning_rate * dsum_of_squared_res_dslope

    # Store intercept and sum of squared residuals for plotting
    sum_of_squared_residuals = sum(r**2 for r in residuals)

    #Appending to lists...
    intercepts.append(temp_intercept)
    intercepts_DEBUG.append(_round_down(temp_intercept, 3))

    slopes.append(temp_slope)
    slopes_DEBUG.append(_round_down(temp_slope, 3))

    sum_of_squared_residuals_history.append(sum_of_squared_residuals)
    sum_of_squared_residuals_history_DEBUG.append(_round_down(sum_of_squared_residuals, 5))

print(intercepts_DEBUG)
print(slopes_DEBUG)
print(sum_of_squared_residuals_history_DEBUG)

#Ploting and adjusting graph size
figure, axs = plt.subplots(1, 3, figsize=(17, 6))

#Intercept vs Sum of Residual Squared
axs[0].plot(intercepts, sum_of_squared_residuals_history, marker = 'o')
axs[0].set_title('Intercept Updates vs. Sum of Squared Residuals')
axs[0].set_xlabel('Intercept Value')
axs[0].set_ylabel('Sum of Squared Residuals')
axs[0].grid(True)

#Slope vs Sum of Residual Squared
axs[1].plot(slopes, sum_of_squared_residuals_history, marker = 'o')
axs[1].set_title('Slope Updates vs. Sum of Squared Residuals')
axs[1].set_xlabel('Slope Value')
axs[1].set_ylabel('Sum of Squared Residuals')
axs[1].grid(True)

#Slope vs Intercept
axs[2].plot(intercepts, slopes)
axs[2].set_title('Intercept Updates vs. Slope Updates')
axs[2].set_xlabel('Intercept Value')
axs[2].set_ylabel('Slope Value')
axs[2].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()

#3d graph attempt
fig = plt.figure()
ax = plt.axes(projection ='3d')

tri = mtri.Triangulation(slopes, intercepts)

# Plot the surface.
ax.set_xlabel('intercept')
ax.set_ylabel('slope')
ax.plot_trisurf(intercepts,slopes,sum_of_squared_residuals_history, triangles=tri.triangles, cmap=plt.cm.Spectral)
ax.view_init(0, 30)
plt.show()