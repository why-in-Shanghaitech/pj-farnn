# regex.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
# 
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)


from typing import List, Union, Iterable
from regex_utils import convert_to_dfa, compile_to_nfa, minimize_dfa


class Regex(object):
    def __init__(self, pattern: Union[str, Iterable[str]], vocab: Iterable[str]=None):
        """
        Customized Regex.

        This implements a minimum syntax of regular expressions, but accepts any iterable inputs.
        It first builds the NFA, then convert to DFA and finally do minimization. 
        The implementation is adapted from https://github.com/dejavudwh/Regex.

        Intended to suit the paper `Cold-start and Interpretability: Turning Regular Expressions 
        into Trainable Recurrent Neural Networks` by Chengyue Jiang, et al.

        Grammar (BNF):
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

        Arguments:
            pattern: Any iterable of strings. Regular expression pattern.
            vocab: The vocabulary. All tokens that are possible to appear in the pattern 
                or input sequence. By default, the vocabulary is all the characters with
                ASCII 0 to 127.
        
        >>> pattern = Regex('[^c]+')
        >>> pattern.match('abbab')
        True
        >>> pattern = Regex(['ha', '+', '.'], vocab=['ha', '!'])
        >>> pattern.match(['ha', 'ha', 'ha', '!'])
        True
        >>> pattern.match(['!'])
        False
        """
        self.pattern = pattern
        self.vocab = list(map(chr, range(127))) if vocab is None else list(vocab)
        
        # prepare the jump table
        nfa_start_node = compile_to_nfa(self.pattern, self.vocab)
        dfa_list, jump_table = convert_to_dfa(nfa_start_node, self.vocab)
        self.start_state, self.jump_table = minimize_dfa(dfa_list, jump_table, self.vocab)

    def match(self, input_sequence: Union[str, List[str]]) -> bool:
        """
        Try to apply the pattern to the whole sequence. Return True if matched, False otherwise. 
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

