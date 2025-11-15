import subprocess
import argparse

def run_script(script_name, args=None):
    cmd = ["python", f"scripts/{script_name}"]
    if args:
        cmd += args
    print(f"🔧 Running: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Master pipeline for DIT5411 project")
    parser.add_argument("--model", type=str, default="cnn", help="Model type to train")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    args = parser.parse_args()

    print("🚀 Step 1: Prepare data")
    run_script("prepare_data.py")

    print("📂 Step 2: Split train/test")
    run_script("split_train_test.py")

    print("🧪 Step 3: Data augmentation")
    run_script("augment.py")

    print("📦 Step 4: Build data pipeline")
    run_script("data_pipeline.py")

    print(f"🎯 Step 5: Train model ({args.model})")
    run_script("train_model.py", ["--model", args.model, "--epochs", str(args.epochs), "--batch_size", str(args.batch_size)])

    print("📊 Step 6: Evaluate model")
    run_script("evaluate_model.py", ["--model", args.model])

if __name__ == "__main__":
    main()