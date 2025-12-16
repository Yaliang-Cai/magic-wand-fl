import json
import os
import random
import shutil

# ================= Core Configuration =================
# Input files (Ensure these match your actual filenames)
FILE_O_SOURCE = "wanddata_O3_original.json"
FILE_W_SOURCE = "wanddata_W_original.json"

# Output directory name
OUTPUT_DIR = "final_dataset_800"

# Target Allocation Strategy (Total: 800 High-Quality Samples)
# We prioritize a balanced Server set and Test set, while enforcing Bias on Clients.
TARGET_COUNTS = {
    # Server: 50% of total. Balanced distribution for training the Backbone.
    "server":   {"O": 100, "W": 100, "N": 200},
    
    # Client A: 20% of total. Biased: Knows 'O', doesn't know 'W'.
    "client_A": {"O": 80,  "W": 0,   "N": 80},
    
    # Client B: 20% of total. Biased: Knows 'W', doesn't know 'O'.
    "client_B": {"O": 0,   "W": 80,  "N": 80},
    
    # Test Set: 10% of total. Balanced and unseen by any training process.
    "test":     {"O": 20,  "W": 20,  "N": 40}
}
# ======================================================

def load_all_data():
    """
    Reads both source files and merges all strokes into a single pool.
    Handles label normalization (e.g., merging 'n', 'N' into 'N').
    """
    pool = {"O": [], "W": [], "N": []}
    files = [FILE_O_SOURCE, FILE_W_SOURCE]
    
    print(">>> Reading source files...")
    for fname in files:
        if not os.path.exists(fname):
            print(f"[Error] File not found: {fname}")
            continue
            
        with open(fname, 'r') as f:
            try:
                data = json.load(f)
                # Handle compatibility for JSONs with or without "strokes" key
                strokes = data if isinstance(data, list) else data.get("strokes", [])
                
                for s in strokes:
                    # 1. Normalize Label: Convert to Upper Case (handle 'o', 'w', 'n')
                    raw_label = s.get("label", "N").upper()
                    
                    # 2. Categorization
                    if raw_label == "O":
                        pool["O"].append(s)
                    elif raw_label == "W":
                        pool["W"].append(s)
                    else:
                        # Treat 'n', 'N', and any unknown labels as Noise (N)
                        pool["N"].append(s)
                        
            except Exception as e:
                print(f"[Error] Failed to read {fname}: {e}")
                
    return pool

def save_subset(strokes, folder, label):
    """
    Saves a list of strokes into the target folder.
    Each stroke is saved as an individual JSON file.
    Structure: folder/Label/Label_index.json
    """
    target_dir = os.path.join(folder, label)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    for i, s in enumerate(strokes):
        # Save as individual JSON file
        fname = f"{label}_{i}.json"
        with open(os.path.join(target_dir, fname), 'w') as f:
            json.dump({"strokes": [s]}, f)

def main():
    # 1. Clean up old output directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        
    # 2. Load and Clean Data
    pool = load_all_data()
    print("-" * 30)
    print(f"[Raw Inventory] O: {len(pool['O'])},  W: {len(pool['W'])},  N: {len(pool['N'])}")
    print(f"[Total Raw] {len(pool['O']) + len(pool['W']) + len(pool['N'])}")
    print("-" * 30)
    
    # 3. Shuffle Data (Crucial for Randomness)
    # Using a fixed seed ensures reproducibility of the experiment.
    random.seed(2025) 
    for k in pool:
        random.shuffle(pool[k])
        
    # 4. Slice and Distribute Data
    cursor = {"O": 0, "W": 0, "N": 0} # Pointer to track usage of the data pool
    
    for split_name, requirements in TARGET_COUNTS.items():
        print(f"Generating Dataset: [{split_name}] ...")
        split_folder = os.path.join(OUTPUT_DIR, split_name)
        
        for label in ["O", "W", "N"]:
            count = requirements[label]
            if count == 0: continue
            
            # Slice the required amount of data from the pool
            start = cursor[label]
            end = start + count
            subset = pool[label][start:end]
            
            # Validation: Check if we have enough data
            if len(subset) < count:
                print(f"  [Warning] Insufficient data for {label}! Needed {count}, got {len(subset)}")
            
            # Save to disk
            save_subset(subset, split_folder, label)
            
            # Update cursor position
            cursor[label] = end
            print(f"  - {label}: Extracted {len(subset)} samples")

    print("-" * 30)
    print(f"✅ Process Completed! Output Directory: {OUTPUT_DIR}")
    print(f"  Unused Spare Data: O:{len(pool['O'])-cursor['O']}, W:{len(pool['W'])-cursor['W']}, N:{len(pool['N'])-cursor['N']}")

if __name__ == "__main__":
    main()