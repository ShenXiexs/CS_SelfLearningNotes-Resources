# Stanford CS336: Language Models From Scratch

## Motivation
Build an end-to-end understanding of large language models, from data and tokenization through transformer design, optimization, and evaluation. The focus is on research-grade clarity and reproducibility rather than black-box usage.

## Course Background and Prerequisites
- Linear algebra, probability, and optimization (CS229-level)
- Deep learning fundamentals (CS230-level)
- NLP basics and sequence modeling (CS224N-level)
- Proficiency with Python and PyTorch
- Systems intuition for training at scale

## Core Concepts Covered
- Tokenization, vocabulary design, and data curation
- Autoregressive modeling and the language modeling objective
- Transformer architecture, attention mechanisms, and positional encodings
- Optimization, regularization, and stability techniques
- Scaling laws, compute/data tradeoffs, and evaluation
- Distributed training and systems considerations
- Safety, alignment, and model behavior diagnostics

## Concepts Learned and Key Takeaways
- The transformer is a flexible compute graph whose inductive biases are defined by data, tokenization, and optimization choices.
- Empirical scaling laws connect model size, data size, and compute to downstream performance.
- Systems decisions (parallelism, checkpointing, data pipelines) materially affect research velocity and model quality.

## How This Fits the CS Learning Pipeline
This module sits after classical ML and deep learning, bridging NLP foundations with modern LLM research. It synthesizes algorithmic understanding with systems tradeoffs, preparing for LLM systems and capstone research.

## Relation to Modern ML, LLM, and Systems Research
- Connects theoretical modeling choices to empirical scaling behavior.
- Emphasizes reproducible training pipelines and experimental rigor.
- Highlights systems constraints that shape current LLM capabilities.

## Repository Layout
- `src/`: modular implementations (tokenizer, transformer blocks, training loop)
- `notes/`: theory summaries, derivations, and reading notes
- `experiments/`: ablations, scaling-law fits, and evaluation studies
- `report/`: structured summaries and final write-ups

## Re-implementation and Reproduction Targets
- Re-implement a minimal GPT-style decoder-only transformer from scratch.
- Reproduce a small-scale scaling-law study (e.g., compute-optimal training at toy scale).
- Replicate a key assignment or paper result with controlled ablations.

## Suggested Projects or Experiments
- **Tokenizer study**: Compare BPE vs unigram LM on perplexity and downstream tasks.
- **Attention variants**: Evaluate Rotary vs ALiBi positional encodings at fixed compute.
- **Data quality**: Filter noisy web data and quantify impact on perplexity and toxicity metrics.
- **Efficiency**: Implement gradient checkpointing and compare memory/throughput tradeoffs.
- **Alignment mini-study**: Test simple preference optimization on small models with synthetic pairs.

## Reproducibility Standards
- Log hyperparameters, seeds, and dataset versions.
- Keep training scripts deterministic where possible.
- Store experiment configs alongside results.
