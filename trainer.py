# trainer.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
# 
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)


import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from logic import Expr, expr
from models import FARNN
from regexps import multiple_of_n


def get_farnn_kwargs():
    """
    Question:
        This function will be used to initalize your FARNN. Feel free to tune the model hyperparameters.
        
        Please be aware that modification to current implementation is optional. You may implement other
        functions while leave this function untouched. We have provided starter codes, you may choose to
        or not to use it.

    >>> vocab = ...
    >>> labels = ['Yes', 'No']
    >>> kwargs = get_farnn_kwargs()
    >>> model = FARNN(vocab, labels, **kwargs)
    """
    return {
        "K": 256,
        "r": 32,
        "h1": 100,
        "h2": 100,
        "amp": 1.0
    }


class Trainer:
    def __init__(self, model: FARNN) -> None:
        """
        Initialize a trainer from FARNN.
        """
        self.model = model
        self.__post_init_()

    def __post_init_(self):
        """
        Question:
            Initialize the model as you like. The dataset is provided in 'data/train.json'.

            HINT: Use `model.init_with_re` to initialize the model with regular expressions.
            HINT: You might want to add some noise to the model parameters after initialization.
            HINT: Sometimes regular expressions that do not appear in rules also help the model converge.
                  If you feel puzzled, just write some regular expressions that are potentially useful.

            Please be aware that modification to current implementation is optional. You may implement
            other functions while leave this function untouched. We have provided starter codes, you
            may choose to or not to use it.
        """
        self.model.reset_parameters()

    def train(self, dataset: Dataset):
        """
        Question:
            Implement the train loop. You may choose the batch size, learning rate, etc. as you like.
            After calling this function, we will test the model on the test set. Remember that we have
            a time limit, so that you should balance the time used to initialize and train the network.

            HINT: You might want to use torch.utils.dataDataLoader to load the dataset.
            HINT: You might want to use a suitable optimizer to train the network.
            HINT: You might want to use a learning rate scheduler to adjust the learning rate.

            Please be aware that modification to current implementation is optional. You may implement
            other functions while leave this function untouched. We have provided starter codes, you
            may choose to or not to use it.
        """
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-5)
        
        def train():
            self.model.train()
            for i, (seq, label) in enumerate(dataset):
                # Compute prediction and loss
                logit = self.model.forward_unbatched(torch.tensor(seq))
                loss = self.model.loss(logit, torch.tensor(label))

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if i % 200 == 199:
                    loss, current = loss.item(), (i + 1)
                    print(f"loss: {loss:>7f}  [{current:>5d}/{len(dataset):>5d}]")
        
        for epoch in range(5):
            print(f"Epoch {epoch + 1}\n-------------------------------")
            train()


if __name__ == "__main__":

    from tokenizers import Tokenizer
    import tokenizers.models
    import tokenizers.pre_tokenizers
    import tokenizers.decoders

    import tempfile
    import json
    from tokenizer import clean_vocab

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, path: str, tokenize) -> None:
            super().__init__()
            with open(path, 'r') as f:
                self.data = json.load(f)
            self.tokenizer = tokenize
        
        def __len__(self) -> int:
            return len(self.data)
        
        def __getitem__(self, index):
            seq = self.tokenizer(self.data[index]["question"])
            label = 0 if self.data[index]["truth"] else 1
            return seq, label

    def build_tokenizer(path_to_train):
        """Prepare the tokenizer."""
        ## load an empty tokenizer
        tokenizer = Tokenizer(tokenizers.models.BPE())
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space = False)
        tokenizer.decoder = tokenizers.decoders.ByteLevel()

        ## load the train set and train the tokenizer
        with open(path_to_train, 'r') as f:
            data = json.load(f)
        samples = [item["question"] for item in data]
        tokenizer.train_from_iterator(samples)

        ## clean the vocab
        vocab = tokenizer.get_vocab()
        with tempfile.NamedTemporaryFile() as tmp:
            tokenizer.save(tmp.name)
            tmp.seek(0)
            data = json.load(tmp)
            merges = [tuple(s.split(" ")) for s in data["model"]["merges"]]
        clean_vocab(vocab, merges)

        ## rebuild the tokenizer
        tokenizer = Tokenizer(tokenizers.models.BPE(vocab, merges))
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space = False)
        tokenizer.decoder = tokenizers.decoders.ByteLevel()

        ## shrink the vocab
        tokens = set()
        for sample in samples:
            for token in tokenizer.encode(sample).tokens:
                tokens.add(token)
        vocab_ = [tokenizer.id_to_token(i) for i in range(len(vocab)) if tokenizer.id_to_token(i) in tokens]
        vocab = {token: i for i, token in enumerate(set(vocab_))}

        ## define the function
        def tokenize(text: str):
            return [vocab[token] for token in tokenizer.encode(text).tokens]
        
        return tokenize, vocab
    
    train_path = "data/train.json"
    
    # tokenize
    print("Tokenizer training...")
    tokenize, vocab = build_tokenizer(train_path)
    train_dataset = CustomDataset(train_path, tokenize)
    labels = ['Yes', 'No']

    # training
    model = FARNN(vocab, labels, **get_farnn_kwargs())
    print("Model initializing...")
    trainer = Trainer(model)
    print("Model training...")
    trainer.train(train_dataset)

    # interacting
    print("Model training completed. Now you may ask it questions! Try to start from: Is the number 165120 divisible by 2?")
    while True:
        sentence = input("Enter your question (empty to quit): ")
        if not sentence:
            break
        try:
            sentence = tokenize(sentence)
            answer = model.predict([sentence])[0]
            print("Answer:", answer)
        except KeyError as e:
            print("Error:", e)
            print("Some tokens in the sentence are not in the specially-designed vocabulary. Please try again.")
            continue
        except Exception as e:
            print("Error:", e)
            print("Some unexpected exceptions raised. Please try again.")
            continue