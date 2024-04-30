# echo "Mamba 130M Zero-shot"
# python evals/lm_harness_eval.py --model hf --model_args pretrained=state-spaces/mamba-130m-hf --tasks hellaswag,arc_easy,arc_challenge --device cuda --batch_size 64

# echo "Pythia 160M Zero-shot"
# python evals/lm_harness_eval.py --model hf --model_args pretrained=EleutherAI/pythia-160m --tasks hellaswag,arc_easy,arc_challenge --device cuda --batch_size 64

# echo "Mamba 370M Zero-shot"
# python evals/lm_harness_eval.py --model hf --model_args pretrained=state-spaces/mamba-370m-hf --tasks hellaswag,arc_easy,arc_challenge --device cuda --batch_size 64

# echo "Pythia 410M Zero-shot"
# python evals/lm_harness_eval.py --model hf --model_args pretrained=EleutherAI/pythia-410m --tasks hellaswag,arc_easy,arc_challenge --device cuda --batch_size 64

# echo "Mamba 790M Zero-shot"
# python evals/lm_harness_eval.py --model hf --model_args pretrained=state-spaces/mamba-790m-hf --tasks hellaswag,arc_easy,arc_challenge --device cuda --batch_size 64

# echo "Pythia 1B Zero-shot"
# python evals/lm_harness_eval.py --model hf --model_args pretrained=EleutherAI/pythia-1b --tasks hellaswag,arc_easy,arc_challenge --device cuda --batch_size 64

# echo "Mamba 1.4B Zero-shot"
# python evals/lm_harness_eval.py --model hf --model_args pretrained=state-spaces/mamba-1.4b-hf --tasks hellaswag,arc_easy,arc_challenge --device cuda --batch_size 64

# echo "Pythia 1.4B Zero-shot"
# python evals/lm_harness_eval.py --model hf --model_args pretrained=EleutherAI/pythia-1.4b --tasks hellaswag,arc_easy,arc_challenge --device cuda --batch_size 64

# echo "Mamba 2.8B Zero-shot"
# python evals/lm_harness_eval.py --model hf --model_args pretrained=state-spaces/mamba-2.8b-hf --tasks hellaswag,arc_easy,arc_challenge --device cuda --batch_size 64

# echo "Pythia 2.8B Zero-shot"
# python evals/lm_harness_eval.py --model hf --model_args pretrained=EleutherAI/pythia-2.8b --tasks hellaswag,arc_easy,arc_challenge --device cuda --batch_size 64



echo "Mamba 130M Few-shot"
python evals/lm_harness_eval.py --model hf --model_args pretrained=state-spaces/mamba-130m-hf --tasks hellaswag,arc_easy,arc_challenge --device cuda --batch_size 64 --num_fewshot 5

echo "Pythia 160M Few-shot"
python evals/lm_harness_eval.py --model hf --model_args pretrained=EleutherAI/pythia-160m --tasks hellaswag,arc_easy,arc_challenge --device cuda --batch_size 64 --num_fewshot 5

echo "Mamba 370M Few-shot"
python evals/lm_harness_eval.py --model hf --model_args pretrained=state-spaces/mamba-370m-hf --tasks hellaswag,arc_easy,arc_challenge --device cuda --batch_size 64 --num_fewshot 5

echo "Pythia 410M Few-shot"
python evals/lm_harness_eval.py --model hf --model_args pretrained=EleutherAI/pythia-410m --tasks hellaswag,arc_easy,arc_challenge --device cuda --batch_size 64 --num_fewshot 5

echo "Mamba 790M Few-shot"
python evals/lm_harness_eval.py --model hf --model_args pretrained=state-spaces/mamba-790m-hf --tasks hellaswag,arc_easy,arc_challenge --device cuda --batch_size 64 --num_fewshot 5

echo "Pythia 1B Few-shot"
python evals/lm_harness_eval.py --model hf --model_args pretrained=EleutherAI/pythia-1b --tasks hellaswag,arc_easy,arc_challenge --device cuda --batch_size 64 --num_fewshot 5

echo "Mamba 1.4B Few-shot"
python evals/lm_harness_eval.py --model hf --model_args pretrained=state-spaces/mamba-1.4b-hf --tasks hellaswag,arc_easy,arc_challenge --device cuda --batch_size 64 --num_fewshot 5

echo "Pythia 1.4B Few-shot"
python evals/lm_harness_eval.py --model hf --model_args pretrained=EleutherAI/pythia-1.4b --tasks hellaswag,arc_easy,arc_challenge --device cuda --batch_size 64 --num_fewshot 5

echo "Mamba 2.8B Few-shot"
python evals/lm_harness_eval.py --model hf --model_args pretrained=state-spaces/mamba-2.8b-hf --tasks hellaswag,arc_easy,arc_challenge --device cuda --batch_size 64 --num_fewshot 5

echo "Pythia 2.8B Few-shot"
python evals/lm_harness_eval.py --model hf --model_args pretrained=EleutherAI/pythia-2.8b --tasks hellaswag,arc_easy,arc_challenge --device cuda --batch_size 64 --num_fewshot 5