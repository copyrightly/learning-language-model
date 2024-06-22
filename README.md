## Tokenization
- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) YouTurbe course by Andrej Karpathy
  
  The Tokenizer is a necessary and pervasive component of Large Language Models (LLMs), where it translates between strings and tokens (text chunks). Tokenizers are a completely separate stage of the LLM pipeline: they have their own training sets, training algorithms (Byte Pair Encoding), and after training implement two fundamental functions: encode() from strings to tokens, and decode() back from tokens to strings. In this lecture we build from scratch the Tokenizer used in the GPT series from OpenAI. In the process, we will see that a lot of weird behaviors and problems of LLMs actually trace back to tokenization. We'll go through a number of these issues, discuss why tokenization is at fault, and why someone out there ideally finds a way to delete this stage entirely.

- Original [notebook](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing#scrollTo=PeU63eDYOhve) for the tutorial
- [My notebook](https://colab.research.google.com/drive/1aVVwI5L0p9ISGbPDVWC7hysz1z72IMrZ#scrollTo=6vw02HpOhCfy)

- Minimal, clean code for the Byte Pair Encoding (BPE) algorithm commonly used in LLM tokenization: https://github.com/karpathy/minbpe/tree/master
    - exercise: https://github.com/karpathy/minbpe/blob/master/exercise.md (refer to `minbpe` and `tests` folder for solution)
    - my forked repo with solution to the exercise: https://github.com/copyrightly/minbpe/tree/luwei-branch
- Reference:
    - OpenAI's resources
        - Code for the GPT2 paper [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf): https://github.com/openai/gpt-2
        - Tiktoken: a fast BPE tokeniser for use with OpenAI's models: https://github.com/openai/tiktoken
        - [Efficient Training of Language Models to Fill in the Middle](https://arxiv.org/abs/2207.14255)
        - [Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/): Sora
        - OpenAI playground (raw model for text completion): https://platform.openai.com/playground/chat?models=gpt-3.5-turbo-16k
    - [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288)
    - [Tiktokenizer](https://tiktokenizer.vercel.app/)
    - [sentencepiece: Unsupervised text tokenizer for Neural Network-based text generation by Google](https://github.com/google/sentencepiece)
    - [Learning to Compress Prompts with Gist Tokens](https://arxiv.org/abs/2304.08467)
    - [Taming Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2012.09841)
      - https://compvis.github.io/taming-transformers/
    - [Integer tokenization is insane](https://www.beren.io/2023-02-04-Integer-tokenization-is-insane/)
    - [SolidGoldMagikarp (plus, prompt generation)](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation)

## GPT-2
- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) YouTurbe course by Andrej Karpathy
- [GPT-2 source code using PyTorch on Huggingface](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py) (OpenAI's original implementation was using tensorflow)
- GPT-3 paper [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) contains details of training, but source code was not released. GPT-2 paper is vague on model training and only inference code and model weights are released, no training code
- Training acceleration
    - For local training, use mps (GPU on Mac)
        - ```
          if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
              device = "mps"
          ```
    - Use [Automatic Mixed Precision](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html) to accelerate training
        - use `torch.set_float32_matmul_precision('high')`
        - focus on [`torch.autocast`](https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-torch-autocast), ignore gradient scalar
            - `FP32` --> On Tensor Core: `TF32`, `BFLOAT16`, `FP16`
        - use `with torch.autocast(device_type=device, dtype=torch.float16):` for inference and loss computation but not back-propagation
    - `model = torch.compile(model)`
        - [Introduction to `torch.compile`](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html): e.g. kernel fusion. Without `torch.compile`, there are a lot of round trip between GPU and HBM.
        - GPU HBM(high bandwidth memory), GPU SRAM, L1 cache, L2 cache, CPU DRAM
    - An operation not covered by `torch.compile`: [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) which uses kernel fusion to compute attention, more FLOPS but less I/O with HBM
        - Use `torch.nn.functional.scaled_dot_product_attention` to replace the original 4-step attention computation as below
            - ```
              att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
              att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
              att = F.softmax(att, dim=-1)
              y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
              ```
              replaced the above with `y = F.scaled_dot_product_attention(q, k, v, is_causal=True)`
        - [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
        - [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)
    - Use power of 2 (e.g. 1024, 512, ...) for the model's parameters whenever you can
        - increase the "ugly" number to the nearest "good" number, e.g. 50257 (`vocab_size`) --> 50304
    - When using `grad_accum`, note that we need to [do normalization manually](https://github.com/karpathy/build-nanogpt/blob/master/train_gpt2.py#L498)
    - 
