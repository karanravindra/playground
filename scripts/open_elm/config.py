import argparse


def calculate_layer_params(
    d_model, N, alpha_min, alpha_max, beta_min, beta_max, layer_idx
):
    # Calculate alpha^i and beta^i for the given layer index
    alpha_i = alpha_min + (alpha_max - alpha_min) * layer_idx / (N - 1)
    beta_i = beta_min + (beta_max - beta_min) * layer_idx / (N - 1)

    # Assuming dh is given as dh = d_model / nh, where nh is not specified
    # We can only calculate nh^i and mi with given alpha_i and beta_i
    # nh^i is determined by alpha^i * (d_model / dh), assuming dh = d_model / nh, nh^i = alpha^i * nh
    # We set nh = d_model / dh for simplicity

    d_h = d_model / alpha_i  # Simplified assumption to calculate dh
    nh_i = alpha_i * (d_model / d_h)
    mi = beta_i

    return nh_i, mi


def main():
    defaults = {
        "d_model": 8,
        "N": 4,
        "alpha_min": 0.5,
        "alpha_max": 4,
        "beta_min": 1,
        "beta_max": 2,
    }

    parser = argparse.ArgumentParser(
        description="Calculate transformer layer parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--d_model",
        type=float,
        help="Dimensionality of the model input.",
        default=defaults["d_model"],
    )
    parser.add_argument(
        "--N", type=int, help="Number of transformer layers.", default=defaults["N"]
    )
    parser.add_argument(
        "--alpha_min",
        type=float,
        help="Minimum alpha value.",
        default=defaults["alpha_min"],
    )
    parser.add_argument(
        "--alpha_max",
        type=float,
        help="Maximum alpha value.",
        default=defaults["alpha_max"],
    )
    parser.add_argument(
        "--beta_min",
        type=float,
        help="Minimum beta value.",
        default=defaults["beta_min"],
    )
    parser.add_argument(
        "--beta_max",
        type=float,
        help="Maximum beta value.",
        default=defaults["beta_max"],
    )

    args = parser.parse_args()

    for i in range(args.N):
        nh_i, mi = calculate_layer_params(
            args.d_model,
            args.N,
            args.alpha_min,
            args.alpha_max,
            args.beta_min,
            args.beta_max,
            i,
        )
        print(f"Layer {i + 1:>2}: nh = {nh_i:.2f}, mi = {mi:.2f}")


if __name__ == "__main__":
    main()
