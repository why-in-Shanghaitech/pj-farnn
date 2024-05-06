# CS274A Natural Language Processing Spring 2024 Project

<div align="center">
<img width="450" src="https://github.com/why-in-Shanghaitech/pj-farnn/assets/43395692/86981d69-6e01-4333-99ca-70084d307aed" />
<p>
  A regular expression system
  <br>
  is equivalent to a recurrent neural network
  <br>
  combining strong interpretability and ability of learning.
</p>
</div>

Since the release of ChatGPT, large language models (LLMs) have dominated the field of natural language processing (NLP). Even though LLMs have achieved remarkable performance in various tasks, they still struggle to solve simple arithmetic tasks. For example, try to ask a popular LLM: `Is the number 654661234561234566543216 divisible by 6?` (you may use [hf-chat](https://huggingface.co/chat)/[mirror](https://hf-mirror.com/chat)) Without external tools, it is hard for the LLMs to answer such a question with high accuracy. In this project, you will train a Finite Automaton Recurrent Neural Network (FARNN) within 5 minutes, which will effectively solve this problem.

This project is designed based on the paper [Cold-start and Interpretability: Turning Regular Expressions into Trainable Recurrent Neural Networks (Jiang, C., et al., 2020)](http://faculty.sist.shanghaitech.edu.cn/faculty/tukw/emnlp20reg.pdf). The authors successfully convert regular expressions to a type of recurrent neural networks (RNNs). You may refer to the [GitHub repository](https://github.com/jeffchy/RE2RNN) of the paper for more details. Don't forget to give the authors a star if you find the repo helpful!

<sub>* The grader of this project was originally modified from the pacman project from CS188 in Berkeley, developed in spring 2019. Problems are designed for CS274A Natural Language Processing course project.</sub>

<details>
<summary>The Map of AI Approaches</summary>
<div align="center">
<img width="400" src="https://github.com/why-in-Shanghaitech/pj-farnn/assets/43395692/f62df3cf-49a4-4fd0-b315-da198f0d4cc7" />
</div>
</details>


## Setup

You need to install some dependencies to finish this project, including `tokenizers`, `numpy`, .etc. To prepare the environment, please execute

```sh
pip install -r requirements.txt
```

This project is developed under python version 3.10.11. We recommend using python with version >= 3.9.0 for this project.

## Usage

This project includes an autograder for you to grade your answers on your machine. If you're familiar with the [pacman project](https://inst.eecs.berkeley.edu/~cs188), you may skip this section since the usage is almost the same. If you've never tried the pacman project, here are some commands that might be helpful:

If you want to get help, run
```sh
    python autograder.py -h
```

If you want to test your code, run
```sh
    python autograder.py
```

If you want to test your code for a specified question, for example, question 1, run
```sh
    python autograder.py -q q1
```

If you want to mute outputs, run
```sh
    python autograder.py -m
```

If you want to test your code for a specified testcase, for example, q1/test_1.test, run
```sh
    python autograder.py -t test_cases/q1/test_1
```

If you want to show the input parameter of your failed cases in question 1, run
```sh
    python autograder.py -q q1 -p
```


## Question 1 (2 points): The BPE Tokenizer

In class, we have learnt about the BPE tokenizer. In this project, we will use an off-the-shelf tokenizer which was originally proposed in [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) by OpenAI. It uses a byte-level version of BPE, which is slightly different from the BPE we learnt in class. It maps all the bytes to an initial alphabet with the size as small as 256 characters (as opposed to the 130,000+ Unicode characters). In addition, this tokenizer adds a special character `Ġ` in front of each token (except the first one), instead of adding a special character `_` at the end of each token.

In this question, your task is to manipulate a pre-trained tokenizer to prevent it from tokenizing numbers. We will use the most popular [huggingface](https://github.com/huggingface/tokenizers) library. Let's first take a look at an example:

```python
>>> import tokenizers
>>> sentence = "Is 1029310928407 a multiple of 3?"
>>> tokenizer.encode(sentence).tokens
['Is', 'Ġ10', '293', '109', '28', '407', 'Ġa', 'Ġmultiple', 'Ġof', 'Ġ3', '?']
```

The tokenizer decides to tokenize the number `1029310928407` into several components. However, under some circumstances, especially the low-resource scenario, it would lead to the data sparsity problem. Each component with separate embeddings only appears a few times in the training corpus, but they do not carry too much difference in meanings.

The solution is very simple: tokenize each digit as an individual token.

```python
>>> sentence = "Is 1029310928407 a multiple of 3?"
>>> tokenizer.encode(sentence).tokens
['Is', 'Ġ1', '0', '2', '9', '3', '1', '0', '9', '2', '8', '4', '0', '7', 'Ġa', 'Ġmultiple', 'Ġof', 'Ġ3', '?']
```

To do this, you need to make a small manipulation on the BPE data: the vocabulary and the merges. Implement the function `clean_vocab` in `tokenizer.py`. You need to remove some subtokens from the vocabulary and remove some merges in place, such that the tokenizer won't merge the digits.

To run the examples above:

```bash
python tokenizer.py
```

To run the autograder for this question:

```bash
python autograder.py -q q1
```

*HINT: GPT-2 tokenizer will prevent BPE from merging across character categories for any byte sequence. That is, there won't be any merges that try to merge digits and other non-digit characters. See the original tokenization file [here](https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json).*


## Question 2 (4 points): Multiple of N

Our journey starts from regex, or regular expressions. You might think today's world with LLMs does not require regular expressions anymore. The fact is that regular expressions are often the first model for many text-processing tasks and they are still widely used in complex NLP tasks, especially in industry where we need highly interpretable and reliable results. People always underestimate the expressiveness of regular expressions. And this time, you will be implementing a regular expression generator that accepts multiples of N.

One thing that is different here is that everything is a token instead of a character. So a regular expression is a sequence of token strings instead of a single string. For example:

```python
>>> from regex import Regex
>>> vocab = ['The', 'Ġmovie', 'Ġis', 'Ġgreat', 'Ġexcellent', 'Ġ!']
>>> regex = Regex(['The', 'Ġmovie', 'Ġis', '[', 'Ġgreat', 'Ġexcellent', ']', 'Ġ!'], vocab = vocab)
>>> regex.match(['The', 'Ġmovie', 'Ġis', 'Ġgreat', 'Ġ!'])
True
>>> regex = Regex(['6', '+'])
>>> regex.match(['6', '6', '6'])
True
```

In this project, you may only use a subset of the regular expression grammar (well, since we only implemented these symbols...). You may use `|`, `()`, `[]`, `.`, `*`, `+`, `?`, `^` which has almost the same meaning as `re` module in python. Some differences (may not cover all the cases):
 - quantifiers (`?`, `*`, `+`) cannot come one after another. That is, `['a', '+', '?']` is invalid (but `a+?` is valid in `re`).
 - the match function will always try to match the whole input. It returns `True` if and only if the pattern accepts the whole input sequence.
 - `^` only serves as negation. We always match the whole sentence, so `^` and `$` are useless.

Implement the function `multiple_of_n` in `regexps.py`. The function has two arguments: `n` and `special_char`. The generated regular expression should accept all the multipliers of `n` including 0 and reject other numbers. `special_char` is to suit the tokenization: Remember in the previous question that when a number is tokenized, the tokenizer may add a special character `Ġ` in front of the first token, depending on where the number appears in the sentence. `special_char` tells you whether `Ġ` will exist. That is, if the number to match is `12` and `special_char` is `False`, then the input sequence is `['1', '2']`. If `special_char` is True, then the input sequence is `['Ġ1', '2']`. The number to match is always non-negative.

You do not need to worry about leading zeros, but empty strings should be rejected. In this question, we guarantee that `2<=n<=6`.

To run the example:

```bash
python regexps.py
```

To run the autograder for this question:

```bash
python autograder.py -q q2
```

Grading: You need to handle `n` and `special_char` properly to receive points. Depending on your response, you will be graded:

| `n`     | `special_char` | **Grade** |
| ------- | -------------- | --------- |
| 2, 5    | False          | +1        |
| 3       | False          | +1        |
| 2, 3, 5 | True           | +1        |
| 4, 6    | False, True    | +1        |

These tests do not depend on each other. That is, you may solve `n=3`, `special_char=False` and receive one point without implementing `n=5`.

We will also calculate the length of your regular expressions. The shorter, the better. If the output is too long, you will fail immediately without receiving any points. Depending on the total length of all possible regular expressions, you will be graded:

| **Total length of regular expressions** | **Grade** |
| --------------------------------------- | --------- |
| > 40,000                                | FAIL      |
| <= 40,000                               | +0        |

That is, you may only receive the points if the total length of all possible regular expressions is less than 40,000. This is easy to achieve if you carefully deal with the brackets. Our reference implementation has a total length of 1,209.

*HINT: This question is intended to let you get familiar with regex and finite automata. [Here](https://math.stackexchange.com/questions/140283/why-does-this-fsm-accept-binary-numbers-divisible-by-three) is a related question. Remember our numbers are decimal, not binary.*


## Question 3 (3 points): The Cold Start

Now we have the regular expressions! The next step is to use these regular expressions to initialize a neural network: FARNN. In section 2.3, the original paper mentioned that REs can be automatically converted to a m-DFA. We have implemented this for you, so now your job is to use these m-DFAs (just finite automata) to initialize the FARNN.

Complete `FARNN.initialize` in `models.py`. The (unbatched version) forward algorithm has been implemented for you, all you need to do is to initialize `self.alpha_0`, `self.ER`, `self.D1`, `self.D2` and `self.mlp`. `self.alpha_0` is the initial state. `self.ER`, `self.D1`, `self.D2` are the decompositions of the transition tensor `T`, which you may directly call function `decompose` to do decomposition. The most tricky part lies in `self.mlp`, which involves the aggregation of multiple matching results.

In section 3.5 of the original paper, the author only simply introduces soft logic when aggregating 2 symbols. We say that it is simple to extend to more symbols. Take `A | B` as an example. Consider `A | B | C`, we have `A | B | C = (A | B) | C`, thus the soft logic value is `min(1, min(1, a+b)+c) = min(1, 1+c, a+b+c)`. Since `c` is either 0 or 1, we know that `c+1` is no less than `c`, thus the soft logic can be written as `min(1, a+b+c)`.

The rules passed in are already in the form of CNF. To get its components, feel free to make use of the functions `conjuncts` and `disjuncts`. Each automaton has a member variable representing its corresponding logic symbol.

After initialization, the neural network should already have the ability to do prediction! Given the input sequence, the network should output a vector of logits over the label set. The correct label should have a logit around 1.

To run the example:

```bash
python models.py
```

To run the autograder for this question:

```bash
python autograder.py -q q3
```

*THINK IT OVER: In the original paper, the author proposes to integrate pretrained word embeddings like GloVE. But here we don't do this. Why?*


## Question 4 (3 points): Training with a Budget

All the things we have done are the preparation of training a powerful model. And this time, we will be training a simple arithmetic QA system. The system should be able to answer yes-no questions about divisible numbers (actually, a binary classifier). The dataset is provided in `data/train.json`. Train a FARNN to make the correct prediction.

We intend to give you full freedom to solve this problem, as long as you achieve an accuracy of 92% within 5 minutes. You may
1. write complex regular expressions to directly initialize the network, without any training. This requires extra effort in regular expression writing. In addition, building the m-DFA and doing tensor decomposition may exceed the time limit.
2. train a FARNN from scratch with random initialization. However, arithmetic problems are known to be difficult for neural models. It might be hard to converge within the time limit.
3. write simple regular expressions, then train the network properly. This is the most promising solution. Initializing a FARNN with simple regular expressions only requires a little time, but it may boost the process of training.

Here are the functions that you might want to implement/modify:
- `get_farnn_kwargs` in `trainer.py`: Hyperparameters to initialize a FARNN.
- `Trainer.__post_init_` in `trainer.py`: Initialize the model.
- `Trainer.train` in `trainer.py`: Implement the train loop.
- `FARNN.forward` in `models.py`: Forward pass of FARNN. You might want to implement a batched version of the forward algorithm.
- `multiple_of_n` in `regexps.py`: Optimize the implementation to initialize the model faster if you decide to use it.

Though it is not required, you'd better finish the previous questions first. Correct implementation of question 1 is mandatory.

To interact with your trained model, run:

```bash
python trainer.py
```

To save the computation, we will only use a small subset of the vocabulary that only contains subtokens appearing in the training set. If you encounter errors like `Some tokens in the sentence are not in the specially-designed vocabulary. Please try again`, please try to input a different sentence.

To run the autograder for this question:

```bash
python autograder.py -q q4
```

To pass this test, you are required to reach an accuracy of 92% within 5 minutes.

*NOTE: Please do not try to hack the tests since we will use a different IID dataset when you submit the project.*


## Submission

In order to submit your project, run 

```bash
python submit.py
```

It will generate a tar file farnn.tar in your project folder. Submit this file to [Autolab](http://10.19.136.45/). You may submit unlimited times before the deadline. The final score will be the last score you get in all your submissions. The deadline is **23:59 on June 8th, 2024**. Late submissions will not be accepted.
