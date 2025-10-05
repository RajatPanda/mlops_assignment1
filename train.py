from sklearn.tree import DecisionTreeRegressor
from misc import run_complete_pipeline


def main():
    print("=" * 60)
    print("Decision Tree Regressor Training")
    print("=" * 60)
    model = DecisionTreeRegressor(
        random_state=42,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2
    )
    mse = run_complete_pipeline(model, "Decision Tree Regressor")
    print("=" * 60)
    print(f"Final MSE for Decision Tree Regressor: {mse:.4f}")
    print("=" * 60)
    return mse


if __name__ == "__main__":
    main()
