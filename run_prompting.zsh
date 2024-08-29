#!/bin/zsh

LOG_FILE="logs/facet_prop_generation/run1_llama38b_all_eval_taxonomies_1inc_prompting_zsh_script_log.txt"

# scripts=(
#     "python3 src/concept_facet_property_prompting.py --config configs/facet_prop_generation/2_llama3_5inc_food_facet_property.json"
#     "python3 src/concept_facet_property_prompting.py --config configs/facet_prop_generation/2_llama3_5inc_equipment_facet_property.json"
#     "python3 src/concept_facet_property_prompting.py --config configs/facet_prop_generation/2_llama3_5inc_science_facet_property.json"
#     "python3 src/concept_facet_property_prompting.py --config configs/facet_prop_generation/2_llama3_5inc_environment_facet_property.json"
# )

scripts=(
    "python3 src/concept_facet_property_prompting.py --config configs/facet_prop_generation/3_llama3_1inc_food_repeat_1.json"
    "python3 src/concept_facet_property_prompting.py --config configs/facet_prop_generation/3_llama3_1inc_equipment_repeat_1.json"
    "python3 src/concept_facet_property_prompting.py --config configs/facet_prop_generation/3_llama3_1inc_science_repeat_1.json"
    "python3 src/concept_facet_property_prompting.py --config configs/facet_prop_generation/3_llama3_1inc_environment_repeat_1.json"    
    "python3 src/concept_facet_property_prompting.py --config configs/facet_prop_generation/3_llama3_1inc_commonsense_repeat_1.json"
)

for cmd in "${scripts[@]}"; do
    echo "Executing $cmd..."
    
    eval "$cmd" >> "$LOG_FILE" 2>&1

    if [[ $? -eq 0 ]]; then
        echo "$cmd executed successfully." >> "$LOG_FILE"
    else
        echo "Error occurred while executing $cmd." >> "$LOG_FILE"
    fi
    
    echo "" >> "$LOG_FILE"
done

echo "All scripts have been executed. Check the log file at $LOG_FILE for details."
