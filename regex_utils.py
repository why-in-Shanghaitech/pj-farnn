# regen_utils.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
# 
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)


"""
This implements a minimum syntax of regular expressions, but accepts any iterable inputs.
It first builds the NFA, then convert to DFA and finally do minimization. 
The implementation is adapted from https://github.com/dejavudwh/Regex.
"""


from typing import List, Dict, Union
from enum import Enum
from collections import defaultdict
import operator
from functools import reduce


# epsilon edge
EPSILON = -1
# set of characters
CCL = -2

class Token(Enum):
    EOS = 0
    ANY = 1
    AT_BOL = 2
    AT_EOL = 3
    CCL_END = 4
    CCL_START = 5
    CLOSE_CURLY = 6
    CLOSE_PAREN = 7
    CLOSURE = 8
    DASH = 9
    END_OF_INPUT = 10
    L = 11
    OPEN_CURLY = 12
    OPEN_PAREN = 13
    OPTIONAL = 14
    OR = 15
    PLUS_CLOSE = 16


Tokens = {
    '.': Token.ANY,
    '^': Token.AT_BOL,
    '$': Token.AT_EOL,
    ']': Token.CCL_END,
    '[': Token.CCL_START,
    '}': Token.CLOSE_CURLY,
    ')': Token.CLOSE_PAREN,
    '*': Token.CLOSURE,
    '-': Token.DASH,
    '{': Token.OPEN_CURLY,
    '(': Token.OPEN_PAREN,
    '?': Token.OPTIONAL,
    '|': Token.OR,
    '+': Token.PLUS_CLOSE,
}



class Lexer(object):
    def __init__(self, pattern):
        self.pattern = pattern
        self.lexeme = ''
        self.pos = 0
        self.isescape = False
        self.current_token = None

    def advance(self):
        pos = self.pos
        pattern = self.pattern
        if pos > len(pattern) - 1:
            self.current_token = Token.EOS
            return Token.EOS

        text = self.lexeme = pattern[pos]
        if text == '\\':
            self.isescape = not self.isescape
            self.pos = self.pos + 1
            self.current_token = self.handle_escape()
        else:
            self.current_token = self.handle_semantic_l(text)

        return self.current_token

    def handle_escape(self):
        expr = self.pattern.lower()
        pos = self.pos
        ev = {
            '\0': '\\',
            'b': '\b',
            'f': '\f',
            'n': '\n',
            's': ' ',
            't': '\t',
            'e': '\033',
        }
        rval = ev.get(expr[pos])
        if rval is None:
            if expr[pos] == '^':
                rval = self.handle_tip()
            elif expr[pos] == 'O':
                rval = self.handle_oct()
            elif expr[pos] == 'X':
                rval = self.handle_hex()
            else:
                rval = expr[pos]
        self.pos = self.pos + 1
        self.lexeme = rval
        return Token.L

    def handle_semantic_l(self, text):
        self.pos = self.pos + 1
        return Tokens.get(text, Token.L)

    def handle_tip(self):
        self.pos = self.pos + 1
        return self.pattern[self.pos] - '@'

    def handle_oct(self):
        return 1

    def handle_hex(self):
        return 1

    def match(self, token):
        return self.current_token == token


class Nfa(object):

    __slots__ = ('edge', 'next_1', 'next_2', 'input_set', 'status_num')

    def __init__(self, status_num = 0):
        self.edge = EPSILON
        self.next_1: Nfa = None
        self.next_2: Nfa = None
        self.input_set = set()
        self.status_num = status_num

    def set_input_set(self, vocab):
        self.input_set = set()
        for c in vocab:
            self.input_set.add(c)


class NfaPair(object):

    __slots__ = ('start_node', 'end_node')

    def __init__(self):
        self.start_node: Nfa = None
        self.end_node: Nfa = None


class Dfa(object):

    __slots__ = ('nfa_sets', 'accepted', 'status_num')

    def __init__(self, status_num: int = 0):
        self.nfa_sets: List[Nfa] = []
        self.accepted = False
        self.status_num = status_num


class DfaGroup(object):

    __slots__ = ('group_num', 'group', 'status_nums')

    def __init__(self, group_num: int = 0):
        self.group_num = group_num
        self.group: List[Dfa] = []
        self.status_nums: Dict[int, List[Dfa]] = defaultdict(list)

    def remove(self, element: Union[Dfa, List[Dfa]]):
        if isinstance(element, list):
            for e in element:
                self.remove(e)
        else:
            self.group.remove(element)
            self.status_nums[element.status_num].remove(element)
            if not self.status_nums[element.status_num]:
                self.status_nums.pop(element.status_num)

    def add(self, element: Union[Dfa, List[Dfa]]):
        if isinstance(element, list):
            for e in element:
                self.add(e)
        else:
            self.group.append(element)
            self.status_nums[element.status_num].append(element)

    def __getitem__(self, idx) -> Dfa:
        return self.group[idx]

    def __len__(self) -> int:
        return len(self.group)


class defaultlist(list):
    """ref: https://stackoverflow.com/questions/8749542"""

    def __init__(self, fx):
        self._fx = fx
    
    def _fill(self, index):
        while len(self) <= index:
            self.append(self._fx())
    
    def __setitem__(self, index, value):
        if isinstance(index, int): self._fill(index)
        list.__setitem__(self, index, value)
    
    def __getitem__(self, index):
        if isinstance(index, int): self._fill(index)
        return list.__getitem__(self, index)



class Pattern:
    """
    Compile the pattern string into an nfa.

    group       ::= expr
    expr        ::= factor_conn
                  | factor_conn "|" expr
    factor_conn ::= factor 
                  | factor factor_conn
    factor      ::= term 
                  | term ("*" | "+" | "?")
    term        ::= char 
                  | "[" char "-" char "]" 
                  | .
                  | "(" expr ")"
    """
    def __init__(self, pattern: str, vocab: List[str]) -> None:
        self.lexer = Lexer(pattern)
        self.lexer.advance()
        self.vocab = vocab
        self.nfa_status = 0

    def create_nfa(self):
        nfa = Nfa(self.nfa_status)
        self.nfa_status += 1
        return nfa

    # terms, including single char, ., [] and ()
    def term(self, pair_out):
        if self.lexer.match(Token.L):
            self.nfa_single_char(pair_out)
        elif self.lexer.match(Token.ANY):
            self.nfa_dot_char(pair_out)
        elif self.lexer.match(Token.CCL_START):
            self.nfa_set_char(pair_out)
        elif self.lexer.match(Token.OPEN_PAREN):
            self.nfa_parenthese(pair_out)


    # single char
    def nfa_single_char(self, pair_out: NfaPair):
        if not self.lexer.match(Token.L):
            return False

        start = pair_out.start_node = self.create_nfa()
        pair_out.end_node = pair_out.start_node.next_1 = self.create_nfa()
        start.edge = self.lexer.lexeme
        self.lexer.advance()
        return True


    # . any char
    def nfa_dot_char(self, pair_out: NfaPair):
        if not self.lexer.match(Token.ANY):
            return False

        start = pair_out.start_node = self.create_nfa()
        pair_out.end_node = pair_out.start_node.next_1 = self.create_nfa()
        start.edge = CCL
        start.set_input_set(self.vocab)

        self.lexer.advance()
        return False


    # [] a set of chars
    def nfa_set_char(self, pair_out: NfaPair):
        if not self.lexer.match(Token.CCL_START):
            return False
        
        neagtion = False
        self.lexer.advance()
        if self.lexer.match(Token.AT_BOL):
            neagtion = True
            self.lexer.advance()
        
        start = pair_out.start_node = self.create_nfa()
        start.next_1 = pair_out.end_node = self.create_nfa()
        start.edge = CCL
        self.dodash(start.input_set)

        if neagtion:
            self.char_set_inversion(start.input_set)

        self.lexer.advance()
        return True


    def char_set_inversion(self, input_set: set):
        origin = set(input_set)
        for c in self.vocab:
            if c not in input_set:
                input_set.add(c)
        for c in origin:
            input_set.remove(c)


    def dodash(self, input_set: set):
        first = ''
        while not self.lexer.match(Token.CCL_END):
            if not self.lexer.match(Token.DASH):
                first = self.lexer.lexeme
                input_set.add(first)
            else:
                self.lexer.advance()
                for c in range(ord(first), ord(self.lexer.lexeme) + 1):
                    input_set.add(chr(c))
            self.lexer.advance()

    
    # parse parenthese
    def nfa_parenthese(self, pair_out: NfaPair):
        if not self.lexer.match(Token.OPEN_PAREN):
            return False
        
        self.lexer.advance()
        self.expr(pair_out)
        
        assert self.lexer.match(Token.CLOSE_PAREN), "missing ), unterminated subpattern"
        self.lexer.advance()

        return True


    # factor connect
    def factor_conn(self, pair_out: NfaPair):
        if self.is_conn(self.lexer.current_token):
            self.factor(pair_out)
        
        while self.is_conn(self.lexer.current_token):
            pair = NfaPair()
            self.factor(pair)
            pair_out.end_node.next_1 = pair.start_node
            pair_out.end_node = pair.end_node

        return True


    def is_conn(self, token: str):
        c = [
            Token.L,
            Token.ANY,
            Token.CCL_START,
            Token.OPEN_PAREN
        ]
        return token in c


    # factor * + ? closure
    def factor(self, pair_out: NfaPair):
        self.term(pair_out)
        if self.lexer.match(Token.CLOSURE):
            self.nfa_star_closure(pair_out)
        elif self.lexer.match(Token.PLUS_CLOSE):
            self.nfa_plus_closure(pair_out)
        elif self.lexer.match(Token.OPTIONAL):
            self.nfa_option_closure(pair_out)


    # * closure
    def nfa_star_closure(self, pair_out: NfaPair):
        if not self.lexer.match(Token.CLOSURE):
            return False
        start = self.create_nfa()
        end = self.create_nfa()
        start.next_1 = pair_out.start_node
        start.next_2 = end

        pair_out.end_node.next_1 = pair_out.start_node
        pair_out.end_node.next_2 = end

        pair_out.start_node = start
        pair_out.end_node = end

        self.lexer.advance()
        return True


    # + plus closure
    def nfa_plus_closure(self, pair_out: NfaPair):
        if not self.lexer.match(Token.PLUS_CLOSE):
            return False
        start = self.create_nfa()
        end = self.create_nfa()
        start.next_1 = pair_out.start_node

        pair_out.end_node.next_1 = pair_out.start_node
        pair_out.end_node.next_2 = end

        pair_out.start_node = start
        pair_out.end_node = end

        self.lexer.advance()
        return True


    # ?
    def nfa_option_closure(self, pair_out: NfaPair):
        if not self.lexer.match(Token.OPTIONAL):
            return False
        start = self.create_nfa()
        end = self.create_nfa()

        start.next_1 = pair_out.start_node
        start.next_2 = end
        pair_out.end_node.next_1 = end

        pair_out.start_node = start
        pair_out.end_node = end

        self.lexer.advance()
        return True


    def expr(self, pair_out: NfaPair = None):
        if pair_out is None:
            pair_out = NfaPair()
        self.factor_conn(pair_out)

        while self.lexer.match(Token.OR):
            self.lexer.advance()
            pair = NfaPair()
            self.factor_conn(pair)
            
            start = self.create_nfa()
            start.next_1 = pair.start_node
            start.next_2 = pair_out.start_node
            pair_out.start_node = start

            end = self.create_nfa()
            pair.end_node.next_1 = end
            pair_out.end_node.next_2 = end
            pair_out.end_node = end

        return pair_out


    def group(self, pair_out: NfaPair = None):
        pair_out = self.expr(pair_out)
        assert self.lexer.match(Token.EOS), "Unexpected Error."
        return pair_out



class Closure:
    def __init__(self, vocab: List[str]) -> None:
        self.vocab = vocab
        self.status_num = 0
        self.dfa_list: List[Dfa] = []
        self._dfa_dict: Dict[int, Dfa] = {}
    
    def convert_to_dfa(self, nfa_start_node: Nfa):
        """
        Convert a given NFA to a DFA.
        """
        self.status_num = 0
        self.dfa_list: List[Dfa] = []
        self._dfa_dict: Dict[int, Dfa] = {}

        # initialization
        jump_table = defaultlist(dict)
        ns = [nfa_start_node]
        n_closure = Closure.closure(ns)
        dfa = self.create_dfa(n_closure)

        # walk through the NFA to find closures
        dfa_index = 0
        while dfa_index < len(self.dfa_list):
            dfa = self.dfa_list[dfa_index]

            transit = self.transit(dfa.nfa_sets)
            for c, nfa_move in transit.items():
                nfa_closure = Closure.closure(nfa_move)
                if nfa_closure is None:
                    continue
                new_dfa = self.convert_completed(nfa_closure)
                next_state = new_dfa.status_num
                jump_table[dfa.status_num][c] = next_state
            
            dfa_index += 1

        # final states
        for dfa in self.dfa_list:
            if dfa.accepted:
                jump_table[dfa.status_num]['accepted'] = True
        
        return (self.dfa_list, jump_table)
    
    @staticmethod
    def hash_of_nfas(nfas: List[Nfa]) -> int:
        # XXX: direct hash(nfas) will cause heavy collision. any better idea?
        return reduce(operator.xor, [hash((nfa, id(nfa), nfa.edge, nfa.status_num, 'cs274a')) for nfa in nfas], 0)
    
    def create_dfa(self, nfas: List[Nfa]) -> Dfa:
        """
        Create a new DFA state.
        """
        dfa = Dfa(self.status_num)
        self.status_num += 1
        for n in nfas:
            dfa.nfa_sets.append(n)
            if n.next_1 is None and n.next_2 is None:
                dfa.accepted = True
        self.dfa_list.append(dfa)
        self._dfa_dict[Closure.hash_of_nfas(dfa.nfa_sets)] = dfa
        return dfa

    def convert_completed(self, closure: List[Nfa]) -> Dfa:
        """
        Given the closure, find the DFA state. If it does not exist, create a new one.
        """
        key = Closure.hash_of_nfas(closure)
        if key in self._dfa_dict:
            return self._dfa_dict[key]
        return self.create_dfa(closure)

    @staticmethod
    def closure(input_set: List[Nfa]) -> List[Nfa]:
        """
        Find the closure of a set of NFA states. Do in-place modification.
        """
        if len(input_set) <= 0:
            return None

        nfa_stack = []
        for i in input_set:
            nfa_stack.append(i)

        while len(nfa_stack) > 0:
            nfa = nfa_stack.pop()
            next1 = nfa.next_1
            next2 = nfa.next_2
            if next1 is not None and nfa.edge == EPSILON:
                if next1 not in input_set:
                    input_set.append(next1)
                    nfa_stack.append(next1)

            if next2 is not None and nfa.edge == EPSILON:
                if next2 not in input_set:
                    input_set.append(next2)
                    nfa_stack.append(next2)
            
        return input_set

    @staticmethod
    def move(input_set: List[Nfa], ch: str) -> List[Nfa]:
        """
        Given an input character, find the next set of NFA states given a set of NFA states.
        """
        out_set = []
        for nfa in input_set:
            if nfa.edge == ch or (nfa.edge == CCL and ch in nfa.input_set):
                out_set.append(nfa.next_1)

        return out_set
    
    def transit(self, input_set: List[Nfa]) -> Dict[str, List[Nfa]]:
        """
        Given a set of NFA states, return a dict. The key is the input character, the value
        is the next set of NFA states.
        """
        output = defaultdict(list)
        for nfa in input_set:
            if nfa.edge == CCL:
                for c in nfa.input_set:
                    output[c].append(nfa.next_1)
            elif nfa.edge in self.vocab:
                output[nfa.edge].append(nfa.next_1)
        return output



class Partition:
    """
    Minimize a DFA through partitioning.
    ref: https://www.geeksforgeeks.org/minimization-of-dfa/
    """

    def __init__(self, dfa_list: List[Dfa], vocab: List[str]) -> None:
        self.dfa_list: List[Dfa] = dfa_list
        self.group_list: List[DfaGroup] = []
        self.on_partition = True
        self.vocab = vocab
        self.group_num = 0


    def minimize_dfa(self, jump_table: List[Dict[str, int]]):
        """
        Iteratively do partitioning till nothing changes.
        """
        self.partition_accepted()

        while self.on_partition:
            self.on_partition = False
            self.partition(jump_table)

        return self.create_mindfa_table(jump_table)


    def append_group(self, group_to_append: List[Dfa]):
        """
        Create a new DFA group given a set of DFA states.
        """
        group = DfaGroup(self.group_num)
        self.group_num += 1
        group.add(group_to_append)
        self.group_list.append(group)


    def partition_accepted(self):
        """
        Split the states into 2 partitions:
            - final states
            - non-final states
        """
        group_reject = []
        group_accept = []
        for dfa in self.dfa_list:
            if dfa.accepted:
                group_accept.append(dfa)
            else:
                group_reject.append(dfa)
        
        if len(group_accept) > 0:
            self.append_group(group_accept)
        if len(group_reject) > 0:
            self.append_group(group_reject)


    def is_distinguishable(self, jump_table: List[Dict[str, int]], dfa1: Dfa, dfa2: Dfa) -> bool:
        """If two DFAs are distinguishable."""
        for token in self.vocab:
            goto1 = jump_table[dfa1.status_num].get(token)
            goto2 = jump_table[dfa2.status_num].get(token)

            if self.dfa_in_group(goto1) != self.dfa_in_group(goto2):
                return True
        return False


    def partition(self, jump_table: List[Dict[str, int]]):
        """
        If any two states in a partition has different outcome to the same input, split
        them into two partitions.
        """
        for group in self.group_list:
            while True:
                dfa_first = group[0]

                # distinguish dfa_first by finding all undistinguishable states
                undist: List[Dfa] = [dfa_first]
                for i in range(1, len(group)):
                    if not self.is_distinguishable(jump_table, dfa_first, group[i]):
                        undist.append(group[i])
                
                # whether to create a new partition
                if len(undist) != len(group):
                    self.append_group(undist)
                    group.remove(undist)
                    self.on_partition = True
                else:
                    break


    def dfa_in_group(self, status_num: int):
        """
        Find the DFA group given the DFA status number.
        """
        for group in self.group_list:
            if status_num in group.status_nums:
                return group
        return None


    def create_mindfa_table(self, jump_table: List[Dict[str, int]]):
        """
        Create the jump table based on the DFA groups.
        """
        trans_table = defaultlist(dict)
        for dfa in self.dfa_list:
            from_dfa = dfa.status_num
            for ch in self.vocab:
                to_dfa = jump_table[from_dfa].get(ch)
                if to_dfa:
                    from_group = self.dfa_in_group(from_dfa)
                    to_group = self.dfa_in_group(to_dfa)
                    trans_table[from_group.group_num][ch] = to_group.group_num
            if dfa.accepted:
                from_group = self.dfa_in_group(from_dfa)
                trans_table[from_group.group_num]['accepted'] = True

        return trans_table
    

## RE -> NFA
def compile_to_nfa(pattern_sequence, vocab):
    pattern = Pattern(pattern_sequence, vocab)
    return pattern.group().start_node

## NFA -> DFA
def convert_to_dfa(nfa_start_node: Nfa, vocab: List[str]):
    """
    Convert a given NFA to a DFA.
    """
    closure = Closure(vocab)
    return closure.convert_to_dfa(nfa_start_node)

## DFA -> MIN-DFA
def minimize_dfa(dfa_list: List[Dfa], jump_table: List[Dict[str, int]], vocab: List[str]):
    """
    Minimize a DFA.
    """
    partition = Partition(dfa_list, vocab)
    jump_table = partition.minimize_dfa(jump_table)
    start_state = partition.dfa_in_group(0).group_num
    return (start_state, jump_table)


def dot(jump_table: List[Dict[str, int]]) -> str:
    """
    Convert the jump table to a dot graph. Return the dot graph code.
    """
    final_states = [f"s{i}" for i, trans in enumerate(jump_table) if trans.get('accepted', False)]
    edges = []
    for i, trans in enumerate(jump_table):
        invert = defaultdict(list)
        for k, v in trans.items():
            if k == 'accepted':
                continue
            invert[v].append(k)
        for k, v in invert.items():
            edges.append(f'\n    s{i} -> s{k} [ label = "{",".join(v)}" ];')
    code = f"""
digraph regex {{
    rankdir=LR;
    size="8,5"
    node [shape = doublecircle]; {' '.join(final_states)};
    node [shape = circle];{''.join(edges)}
}}
"""
    return code