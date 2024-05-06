# Explainable Style Transfer

This repository contains code for the paper [*ICLEF: In-Context Learning with Expert Feedback for Explainable Style Transfer*](https://asaakyan.github.io/content/iclef.pdf).

## Obtaining the data

The e-GYAFC corpus is based on the GYAFC corpus, whcih was created using the Yahoo Answers corpus: L6 - Yahoo! Answers Comprehensive Questions and Answers version 1.0 . This Yahoo Answers corpus can be requested free of charge for research purposes. Once you have gained access to the L6 corpus, please forward the acknowledgment to Arkadiy Saakyan (a.saakyan@cs.columbia.edu), along with your affiliation and a short description of how you will be using the data, and we will provide access to the e-GYAFC corpus.

## Generating the data

Please see notebooks and prompts in the egyac-generation folder.

## Obtaining the models

We are in the process of uploading some of the fine-tuned models to huggingface.

## Fine-tuning the models

Please see instructions in fine-tune folder. Fine-tuning scripts are all taken from Stanford-Alpaca.

## Running inference

To run inference on instruction models or fine-tuned models, feel free to use notebooks in the inference folder.

## Running evaluation

To run evaluation on instruction models or fine-tuned models, feel free to use notebooks in the evaluation folder.

## Questions

Research-related questions can be directed to Arkadiy Saakyan (a.saakyan@cs.columbia.edu). For coding-related questions, please raise a request on GitHub.

## Citation

If you find this project useful, please cite us as follows:
```
@misc{saakyan2023iclef,
      title={ICLEF: In-Context Learning with Expert Feedback for Explainable Style Transfer}, 
      author={Arkadiy Saakyan and Smaranda Muresan},
      year={2023},
      eprint={2309.08583},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Acknowledgements
This research is supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via the HIATUS Program contract #2022-22072200005. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.
