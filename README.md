# SEQUOR: A Multi-Turn Benchmark for Realistic Constraint Following

**SEQUOR** is a benchmark for evaluating LLMs on multi-turn realistic constraint following. It includes five test sets — *single*, *tuples*, *replace*, *add*, and *everything* — built from a fixed collection of user-turn sequences that differ in how constraints are introduced and modified across conversation turns. The repository includes the full pipeline for constraint collection and filtering, testset generation, model response generation, evaluation with LLM-as-a-judge, and result analysis.

📄 Paper: https://arxiv.org/abs/2605.06353


## Repository Structure

- `data/`: Constraints, tasks, and testsets that constitute SEQUOR.
- `pipeline/`: Constraint collection and filtering pipeline (extraction, deduplication, satisfiability, triviality, subjectivity, tuple creation) and evaluation with LLM-as-a-judge. See [`pipeline/README.md`](pipeline/README.md).
- `multi_if/`: Testset generation, model response generation, scoring, and plotting. See [`multi_if/README.md`](multi_if/README.md).


## Citation

If you use SEQUOR or this repository in your work, please cite our paper:
```bibtex
@misc{canaverde2026sequormultiturnbenchmarkrealistic,
      title={SEQUOR: A Multi-Turn Benchmark for Realistic Constraint Following}, 
      author={Beatriz Canaverde and Duarte M. Alves and José Pombal and Giuseppe Attanasio and André F. T. Martins},
      year={2026},
      eprint={2605.06353},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.06353}, 
}
```
