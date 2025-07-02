
dirs=(
'path/to/data'
)

SAVE_DIR='clinical_neuro/preprocess_20x_768'

# Iterate over the directories
for dir in "${dirs[@]}"; do
    # extract paths and get the coordinate
    python3 gen_tiles.py \
    --source "$dir" \
    --save_dir "$SAVE_DIR" \
    --seg --patch --stitch \
    --preset 'bwh_biopsy.csv'

done