from mode_fitting import FCD
import utility
import utility_guesses
import numpy as np

#Set datasets
y=np.load(f"test_datasets/cryptocoin_tests/test18.npy")[:1000]
x=np.arange(len(y))
settings_args={"multi_scale": False, 'num_segments_single': 1}
#Initialize FCD runner
fcd = FCD(
    x_dataset=x, y_dataset=y,
    model=utility.model_sin6,
    initial_guesses_function=utility_guesses.initial_guess_sin6,
    parallel=True,
    verbose=1
)

# Execute fitting
params = fcd.run()

# Extract analytic insights
fcd.print_fitted_functions()
fitted_y_values=fcd.calculate_y_fit_modes()
derivatives = fcd.calculate_derivatives(order=1, print_derivative_formulas=True)
integrals = fcd.calculate_integrals(order=1, print_integral_formulas=True)