import numpy as np

weight1 = 3.34
weight2 = -1.22
weight3 = -3.53
weight4 = -2.30
bias1 = -1.43
bias2 = 0.57
bias3 = 0  # Unoptimized bias!
learning_rate = 0.1
epoch = 50

# Define dosage and efficiency (from video)
dosage = [0, 0.5, 1]
efficiency = [0, 1, 0]

# Softplus activation function
def Softplus(x):
    return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x, 0)

# Residual squared calculation
def Residual(observed_value, predicted_value):
    return observed_value - predicted_value

# Derivative of SSR(Sum of squared Residuals) to derivative of Predicted Values(Predicted efficiency) is *-2
# Derivative of predicted efficiency to derivative of bias3 is *1

def dSSR_dB3(residual):
    return -2 * residual

# Blue Squiggle
def Path1(dosage):
    return Softplus(dosage * weight1 + bias1) * weight2

# Orange squiggle
def Path2(dosage):
    return Softplus(dosage * weight3 + bias2) * weight4

# Green Squiggle = Blue Squiggle + Orange Squiggle + bias3
def Output(path1Output, path2Output):
    return path1Output + path2Output + bias3

# Calculate sum of squared residuals
sum_of_squared_residuals_Y = []
for _ in range(epoch):

    sum_of_squared_residuals = 0
    slope = 0

    for i, dose in enumerate(dosage):
        predicted_efficiency = Output(Path1(dose), Path2(dose))
        temp_residual = Residual(efficiency[i], predicted_efficiency)
        slope += dSSR_dB3(temp_residual)

        #For the graph
        residualSquared = temp_residual**2
        sum_of_squared_residuals += residualSquared
        
    step_size = slope * learning_rate
    bias3 -= step_size

    #For the graph
    sum_of_squared_residuals_Y.append(sum_of_squared_residuals)


print("Sum of Squared Residuals:", sum_of_squared_residuals)
print('Bias 3 =', bias3)