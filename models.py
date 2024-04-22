# models.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
# 
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)


from typing import List, Tuple, Dict
from regex import Regex
from logic import (
    Expr, 
    is_valid_cnf,
    conjuncts,
    disjuncts
)
import util

import torch
import torch.nn as nn

import tensorly
from tensorly.decomposition import parafac
tensorly.set_backend('pytorch')


def decompose(tensor: torch.Tensor, rank: int) -> List[torch.Tensor]:
    """
    Given a tensor, return its decomposition.

    Return:
        factors (:obj:`List[torch.Tensor]`): 
            List of factors of the CP decomposition element i is of shape (tensor.shape[i], rank)
    """
    difficulty = tensor.numel()
    iteration = 20 if difficulty > 1e6 else 100
    _, factors = parafac(tensor, rank=rank, init='svd', random_state=42, n_iter_max=iteration)
    return factors


class FiniteAutomaton:
    def __init__(self, symbol: Expr, start_state: int, jump_table: List[Dict[str, int]], vocab: Dict[str, int]):
        """
        Initialize a Finite Autotmaton.

        Args:
            symbol (:obj:`Expr`):
                A logic symbol that represents the finite automaton.
            start_state (:obj:`int`):
                The index of the start state.
            jump_table (:obj:`List[Dict[str, int]]`):
                Transition of the finite automaton. Each element is a state, a dict that
                represents the transition. The key is the token and the value is the index
                of next state.
            vocab (:obj:`Dict[str, int]`):
                A dictionnary of string keys and their ids, e.g. `{"am": 0,...}`
        """
        self.symbol = symbol
        self.start_state = start_state
        self.jump_table = jump_table
        self.vocab = vocab

    def match(self, input_sequence: List[str]) -> bool:
        """
        Try to match a sequence. Return True if matched, False otherwise. 
        """
        cur_status = self.start_state
        for c in input_sequence:
            jump_dict = self.jump_table[cur_status]
            js = jump_dict.get(c)
            if js is None:
                return False
            else:
                cur_status = js
        return self.jump_table[cur_status].get('accepted', False)

    def parameterize(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameterize a finite automaton. Convert a finite automaton to a weighted finite automaton.

        Return:
            alpha_0 (:obj:`torch.Tensor`):
                Initial weights of shape `(K)`.
            alpha_oo (:obj:`torch.Tensor`):
                Final weights of shape `(K)`.
            T (:obj:`torch.Tensor`):
                The transition weight tensor T of shape `(V, K, K)`.
        """
        K = len(self.jump_table)
        V = len(self.vocab)

        alpha_0 = torch.zeros(K)
        alpha_0[self.start_state] = 1

        alpha_oo = torch.zeros(K)
        for i, trans in enumerate(self.jump_table):
            if trans.get("accepted", False):
                alpha_oo[i] = 1
        
        T = torch.zeros(V, K, K)
        for i, trans in enumerate(self.jump_table):
            for k, v in trans.items():
                if k == 'accepted':
                    continue
                T[self.vocab[k], i, v] = 1

        return (alpha_0, alpha_oo, T)
    

class System:
    """
    Base class of the inference system. This class outlines the structure of a System,
    but doesn't implement any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """
    def __init__(self,
            vocab: Dict[str, int], 
            labels: List[str]
    ):
        """
        Initialize a RE system.
        """
        super(System, self).__init__()

        # Initialize
        self.vocab = vocab
        self.labels = labels
    
    def init_with_re(self, regexps: Dict[Expr, List[str]], rules: List[Expr]):
        """
        Initialize the system with the given regexps.
        """
        # build vocab
        vocab = ["" for i in range(len(self.vocab))]
        for k, v in self.vocab.items():
            vocab[v] = k
        
        # generate automata
        automata = []
        for k, v in regexps.items():
            regex = Regex(v, vocab)
            automata.append(
                FiniteAutomaton(k, regex.start_state, regex.jump_table, self.vocab)
            )
        
        self.initialize(automata, rules)
    
    def initialize(self, automata: List[FiniteAutomaton], rules: List[Expr]):
        """
        Initialize the system with the given automata.
        """
        util.raiseNotDefined()
    
    def predict(self, sentences: List[List[int]]) -> List[str]:
        """
        Given a list of sentence, return the prediction results.
        """
        util.raiseNotDefined()


class RE(System):
    def __init__(self,
            vocab: Dict[str, int], 
            labels: List[str]
    ):
        """
        Initialize a RE system.
        """
        super(RE, self).__init__(vocab, labels)

    def initialize(self, automata: List[FiniteAutomaton], rules: List[Expr]):
        self.automata = automata
        self.rules = rules

    def predict(self, sentences: List[List[int]]) -> List[str]:
        random = util.FixedRandom().random

        # build vocab
        vocab = ["" for i in range(len(self.vocab))]
        for k, v in self.vocab.items():
            vocab[v] = k
        
        # matching
        results = []
        for sentence in sentences:
            sentence = [vocab[token] for token in sentence]
            matchings = {
                automaton.symbol: automaton.match(sentence)
                for automaton in self.automata
            }
            candidates = []
            for i, rule in enumerate(self.rules):
                # eval a rule
                value = True
                for subrule in conjuncts(rule):
                    value_ = False
                    for literal in disjuncts(subrule):
                        if literal.op == '~':
                            value__ = not matchings[literal.args[0]]
                        else:
                            value__ = matchings[literal]
                        value_ |= value__
                    value &= value_
                if value:
                    candidates.append(self.labels[i])
            # randomly select a valid label. If all rejected, randomly select
            # one from the label set.
            if not candidates: candidates = self.labels
            results.append(random.choice(candidates))
        return results


class FARNN(nn.Module, System):
    def __init__(self, 
            vocab: Dict[str, int], 
            labels: List[str], 
            K: int = 256, 
            r: int = 32,
            h1: int = 100,
            h2: int = 100,
            amp: float = 1.0
    ):
        """
        Initialize a FARNN.

        Args:
            vocab (:obj:`Dict[str, int]`):
                A dictionnary of string keys and their ids, e.g. `{"am": 0,...}`
            labels (:obj:`List[str]`):
                A list of the label set.
                
            K (:obj:`int`):
                total number of states
            r (:obj:`int`):
                rank of the transition tensor
            h1 (:obj:`int`):
                size of the 1st hidden layer of the mlp
            h2 (:obj:`int`):
                size of the 2nd hidden layer of the mlp
            amp (:obj:`float`):
                amplifier of the loss function, the logits will multiply by this value
        """
        nn.Module.__init__(self)
        System.__init__(self, vocab, labels)

        # Initialize hyperparameters
        self.K = K
        self.r = r
        self.h1 = h1
        self.h2 = h2
        self.amp = amp

        # network parameters
        self.alpha_0 = nn.Parameter(torch.Tensor(K))
        self.ER = nn.Embedding(len(vocab), r)
        self.D1 = nn.Linear(K, r, bias=False)
        self.D2 = nn.Linear(r, K, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(K, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, len(labels))
        )

        # loss function
        self.loss = nn.CrossEntropyLoss()
    

    def reset_parameters(self):
        """
        Reset the parameters of the model.
        """
        nn.init.normal_(self.alpha_0)
        self.ER.reset_parameters()
        self.D1.reset_parameters()
        self.D2.reset_parameters()
        self.mlp[0].reset_parameters()
        self.mlp[2].reset_parameters()
        self.mlp[4].reset_parameters()
    

    def add_noise(self, scale: float = 0.1):
        """
        Add noise to the parameters of the model.
        """
        self.alpha_0.data += torch.randn_like(self.alpha_0) * scale
        self.ER.weight.data += torch.randn_like(self.ER.weight) * scale
        self.D1.weight.data += torch.randn_like(self.D1.weight) * scale
        self.D2.weight.data += torch.randn_like(self.D2.weight) * scale
        self.mlp[0].weight.data += torch.randn_like(self.mlp[0].weight) * scale
        self.mlp[2].weight.data += torch.randn_like(self.mlp[2].weight) * scale
        self.mlp[4].weight.data += torch.randn_like(self.mlp[4].weight) * scale
        self.mlp[0].bias.data += torch.randn_like(self.mlp[0].bias) * scale
        self.mlp[2].bias.data += torch.randn_like(self.mlp[2].bias) * scale
        self.mlp[4].bias.data += torch.randn_like(self.mlp[4].bias) * scale


    def initialize(self, automata: List[FiniteAutomaton], rules: List[Expr]):
        """
        Question:
            Initialize the FARNN with the given automata. The (unbatched version)
            forward algorithm has been implemented for you, all you need to do is to
            initialize `self.alpha_0`, `self.ER`, `self.D1`, `self.D2` and `self.mlp`.

            See section 3.2, 3.5 of the original paper for more details.

            Think it over: how to initialize the MLP layer? You might want to do some
            paper work before you start coding.

            Attention: this function might be called mutliple times after a FARNN is
            initialized. Each function call should overwrite all the parameters.

            HINT: You might be interested in some useful functions: `decompose`,
            `conjuncts`, `disjuncts`.
            HINT: Use `automata[i].symbol` to get the symbol of the i-th automaton.
            HINT: You may want to refer to the `RE` class for the usage of `automata`
            and `rules`.

        Args:
            automata (:obj:`List[FiniteAutomaton]`):
                A list of automata that is used to initialized the FARNN.
            rules (:obj:`List[Expr]`):
                A list of rules (in form of CNF) for the label set. Each rule is
                a logic expression of matching results (the symbols of automantons)
                that implies a specific label. See section 2.2 of the original paper.
        """
        
        # constants
        V, K = len(self.vocab), sum(len(fa.jump_table) for fa in automata)
        total_clauses = sum(len(conjuncts(rule)) for rule in rules)

        # basic checks
        assert len(rules) == len(self.labels), f"The number of rules ({len(rules)}) must be consistent with the size of the label set ({len(self.labels)})."
        assert all(is_valid_cnf(rule) for rule in rules), "The rules must be in form of CNF."
        assert K <= self.K, f"Too many states. You request {K} states but K={self.K} when creating the FARNN. Either reduce automata or use a larger K."
        assert total_clauses <= self.h1, f"Too many clauses ({total_clauses}) in rules. Either reduce rules or use a larger h1 ({self.h1})."
        assert len(self.labels) <= self.h2, f"To initialize from FAs, h2 ({self.h2}) must be no smaller than the size of the label set ({len(self.labels)}."


        """YOUR CODE HERE"""
        self.reset_parameters() # remove this line to start your implementation


    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Question:
            Core of FARNN. The current version uses a for loop to calculate the logits. You might want
            to modify this function to make it more efficient.

            Please be aware that modification to current implementation is optional. You may implement
            other functions while leave this function untouched. We have provided starter codes, you
            may choose to or not to use it.

        Args:
            x (:obj:`torch.Tensor`):
                The input tokens of shape `(n, l)`, where `n` is the batch size, `l` is 
                the length of the sequence. Padding might be random numbers.
            lengths (:obj:`torch.Tensor`):
                The lengths of each sentence in the input batch. Shape: `(n)`.
        
        Return:
            logits (:obj:`torch.Tensor`):
                The logit over the label set of shape `(n, L)`, where `n` is the batch
                size, `L` is the size of the label set.
        """
        return torch.stack(
            [ self.forward_unbatched(x_[:l]) for x_, l in zip(x, lengths) ],
            dim = 0
        )


    def forward_unbatched(self, x: torch.Tensor) -> torch.Tensor:
        """
        Core of FARNN. Unbatched version of forward. Do NOT change anything in this function.

        Args:
            x (:obj:`torch.Tensor`):
                The input tokens of shape `(l)`, where `l` is the length of the sequence.
        
        Return:
            logit (:obj:`torch.Tensor`):
                The logit over the label set of shape `(L)`, where `L` is the size of the 
                label set.
        """
        # get embddings
        embeds = self.ER(x)

        # initialize
        hidden = self.alpha_0

        # recurrent neural network
        for embed in embeds:

            # Equation 5
            a = self.D1(hidden) * embed
            hidden = self.D2(a)

        return self.mlp(hidden)
    

    def calculate_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Return the loss given the logits and labels.
        
        Args:
            logits (:obj:`torch.Tensor`):
                The logit over the label set of shape `(n, L)`, where `n` is the batch
                size, `L` is the size of the label set.
            labels (:obj:`torch.Tensor`):
                The labels of shape `(n)`, where `n` is the batch size.
        """
        return self.loss(logits*self.amp, labels)


    def predict(self, sentences: List[List[int]]) -> List[str]:
        self.eval()
        x = nn.utils.rnn.pad_sequence([
            torch.tensor(sentence)
            for sentence in sentences
        ], batch_first=True)
        lengths = torch.tensor([len(sentence) for sentence in sentences])
        logits = self.forward(x, lengths)
        results = [self.labels[idx.item()] for idx in logits.argmax(dim=1)]
        return results
        


if __name__ == '__main__':

    from regexps import multiple_of_n
    import tokenizers

    print("Running models.py ...")

    sentences = [
        "1029310928407", # False
        "102931092840222", # True
    ]

    vocab = {str(i):i for i in range(10)}
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(vocab, []))
    labels = ['True', 'False']

    sentences = [
        tokenizer.encode(sentence).ids
        for sentence in sentences
    ]

    # The RE system
    system = RE(vocab, labels)
    system.init_with_re({
        Expr('A'): multiple_of_n(3, False)
    }, [
        Expr('A'), ~Expr('A')
    ])
    print("RE system:", system.predict(sentences))

    # The FA-RNN system
    system = FARNN(vocab, labels, r=16)
    system.init_with_re({
        Expr('A'): multiple_of_n(3, False)
    }, [
        Expr('A'), ~Expr('A')
    ])
    print("FARNN system:", system.predict(sentences))
