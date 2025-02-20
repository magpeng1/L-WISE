from scipy.stats import binom
import argparse

def calculate_dropout_probability(n_dropouts, n_condition_dropouts, n_conditions, p_condition=None):
    """
    Calculate the probability of observing n_condition_dropouts or more dropouts 
    in a specific condition, given total n_dropouts across all conditions,
    under the null hypothesis of equal dropout probability across conditions.
    
    Parameters:
    -----------
    n_dropouts : int
        Total number of dropouts across all conditions
    n_condition_dropouts : int
        Number of dropouts observed in the condition of interest
    n_conditions : int, optional
        Total number of conditions in the experiment
    p_condition : float, optional
        Probability of being assigned to the condition of interest.
        If None, assumes equal probability (1/n_conditions)
        
    Returns:
    --------
    float
        Probability of observing n_condition_dropouts or more in one condition
        under the null hypothesis
    """

    if p_condition is None:
      # Probability of being in any one condition under null hypothesis
      p_condition = 1 / n_conditions
    
    # Calculate P(X >= n_condition_dropouts) = 1 - P(X <= n_condition_dropouts-1)
    # where X follows Binomial(n_dropouts, 1/n_conditions)
    probability = 1 - binom.cdf(n_condition_dropouts - 1, n_dropouts, p_condition)
    
    return probability

def main():
    parser = argparse.ArgumentParser(description='Calculate probability of extreme dropout rates in experimental conditions')

    parser.add_argument('--total', '-t', type=int, required=True, help='Total number of dropouts across all conditions')
    parser.add_argument('--condition', '-c', type=int, required=True, help='Number of dropouts in the condition of interest')
    parser.add_argument('--n_conditions', '-n', type=int, default=6, help='Total number of conditions in the experiment (default: 6)')
    parser.add_argument('--p_condition', '-p', type=float, default=None, help='Probability of being assigned to the condition of interest')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print additional details')

    args = parser.parse_args()

    if args.condition > args.total:
        parser.error("Number of dropouts in condition cannot exceed total dropouts")
    
    if args.n_conditions < 2:
        parser.error("Number of conditions must be at least 2")

    if args.p_condition is not None:
        if not 0 < args.p_condition < 1:
            parser.error("Probability must be between 0 and 1")

    p_value = calculate_dropout_probability(
        args.total,
        args.condition,
        args.n_conditions,
        p_condition=args.p_condition
    )

    if args.verbose:
        print(f"\nAnalyzing dropout patterns:")
        print(f"- Total dropouts: {args.total}")
        print(f"- Dropouts in condition: {args.condition}")
        print(f"- Number of conditions: {args.n_conditions}")
        print(f"\nResults:")
        print(f"- P-value: {p_value:.4f}")
        print(f"- This corresponds to a {p_value*100:.2f}% chance under the null hypothesis")
        if p_value < 0.05:
            print("- This result is statistically significant at the 0.05 level")
    else:
        print(f"{p_value:.4f}")

if __name__ == "__main__":
    main()