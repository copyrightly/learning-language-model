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
