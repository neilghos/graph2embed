import numpy as np
from train import train as train_convreader
from baseline_gin import train_baseline as train_gin

def main():
    # Only test on the requested datasets
    datasets = ["PTC_MR"]
    num_seeds = 10
    
    results = {}
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"EVALUATING DATASET: {dataset}")
        print(f"{'='*50}")
        
        convreader_accs = []
        gin_accs = []
        
        for seed in range(num_seeds):
            print(f"\n--- Seed {seed} ---")
            
            # Train LSTMReader
            print("Training LSTMReader V2...")
            acc_conv = train_convreader(dataset, seed=seed)
            convreader_accs.append(acc_conv)
            
            # Train Baseline GIN (Dormant)
            # print("Training Baseline GIN...")
            # acc_gin = train_gin(dataset, seed=seed)
            # gin_accs.append(acc_gin)
            
        results[dataset] = {
            'convreader_mean': np.mean(convreader_accs),
            'convreader_std': np.std(convreader_accs),
            # 'gin_mean': np.mean(gin_accs),
            # 'gin_std': np.std(gin_accs)
        }
        
    print("\n\n" + "="*50)
    print("FINAL 10-SEED BENCHMARK RESULTS")
    print("="*50)
    
    for dataset, res in results.items():
        print(f"\n{dataset}:")
        print(f"  ConvReader V2 : {res['convreader_mean']:.4f} ± {res['convreader_std']:.4f}")
        # print(f"  Standard GIN  : {res['gin_mean']:.4f} ± {res['gin_std']:.4f}")
        
        # if res['convreader_mean'] > res['gin_mean']:
        #     print(f"  -> ConvReader WINS by {(res['convreader_mean'] - res['gin_mean']):.4f}!")
        # else:
        #     print(f"  -> GIN WINS by {(res['gin_mean'] - res['convreader_mean']):.4f}!")

if __name__ == '__main__':
    main()
