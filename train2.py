from sklearn.kernel_ridge import KernelRidge
from misc import run_complete_pipeline

def main():
    print("\n" + "=" * 60)
    print("Kernel Ridge Regression Training")
    print("=" * 60)
    model = KernelRidge(
        alpha=1.0,
        kernel='rbf',
        gamma=0.1
    )
    mse = run_complete_pipeline(model, "Kernel Ridge Regression")
    print("=" * 60)
    print(f"Final MSE for Kernel Ridge Regression: {mse:.4f}")
    print("=" * 60)
    return mse

if __name__ == "__main__":
    main()
