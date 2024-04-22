# SymbolicTestClasses.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
# 
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)


import testClasses
import util

from regex_utils import Tokens, convert_to_dfa, compile_to_nfa, minimize_dfa, Closure, dot
from logic import expr, conjuncts, disjuncts

from tokenizers import Tokenizer
import tokenizers.models
import tokenizers.pre_tokenizers
import tokenizers.decoders

import urllib.parse
import torch
import json
import tempfile


# Simple test case which evals an arbitrary piece of python code.
# The test is correct if the output of the code given the student's
# solution matches that of the instructor's.
class EvalTest(testClasses.TestCase):

    def __init__(self, name, question, testDict):
        super(EvalTest, self).__init__(name, question, testDict)
        self.preamble = compile(testDict.get('preamble', ""), "%s.preamble" % self.getPath(), 'exec')
        self.test = compile(testDict['test'], "%s.test" % self.getPath(), 'eval')
        self.success = testDict['success']
        self.failure = testDict['failure']

    def evalCode(self, moduleDict):
        bindings = dict(moduleDict)
        # exec self.preamble in bindings
        exec(self.preamble, bindings)
        return str(eval(self.test, bindings))

    def execute(self, grades, moduleDict, solutionDict):
        result = self.evalCode(moduleDict)
        if result == solutionDict['result']:
            grades.addMessage('PASS: %s' % self.path)
            grades.addMessage('\t%s' % self.success)
            return True
        else:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\tstudent result: "%s"' % result)
            grades.addMessage('\tcorrect result: "%s"' % solutionDict['result'])

        return False

    def writeSolution(self, moduleDict, filePath):
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('# The result of evaluating the test must equal the below when cast to a string.\n')

            handle.write('result: "%s"\n' % self.evalCode(moduleDict))
        return True


# Hidden test case checks the md5 of the result. Student can view
# the test case but not the plain text of the solution.
class HiddenTest(EvalTest):

    def evalCode(self, moduleDict):
        bindings = dict(moduleDict)
        # exec self.preamble in bindings
        exec(self.preamble, bindings)
        return util.md5(eval(self.test, bindings))

    def writeSolution(self, moduleDict, filePath):
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('# The hash of the result of evaluating the test must equal the below.\n')

            handle.write('result: "%s"\n' % self.evalCode(moduleDict))
        return True


# Test case that requires student to raise an exception.
class ExceptionTest(EvalTest):

    def execute(self, grades, moduleDict, solutionDict):
        try:
            result = self.evalCode(moduleDict)
        except Exception as inst:
            if str(type(inst)) == solutionDict['result']:
                grades.addMessage('PASS: %s' % self.path)
                grades.addMessage('\t%s' % self.success)
                return True
            raise inst
        
        grades.addMessage('FAIL: %s' % self.path)
        grades.addMessage('\t%s' % self.failure)
        grades.addMessage('\tstudent result: "%s"' % result)
        grades.addMessage('\tcorrect result: "%s"' % solutionDict['result'])

        return False

    def writeSolution(self, moduleDict, filePath):
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('# The result of evaluating the test must raise the following exception.\n')

            try:
                result = self.evalCode(moduleDict)
            except Exception as inst:
                result = str(type(inst))
            else:
                raise RuntimeError('Use ExceptionTest but no exception raised.')

            handle.write('result: "%s"\n' % result)
        return True
    


class VocabTest(testClasses.TestCase):

    def __init__(self, name, question, testDict):
        super(VocabTest, self).__init__(name, question, testDict)
        self.vocab = compile(testDict['vocab'], "%s.test" % self.getPath(), 'eval')
        self.merges = compile(testDict['merges'], "%s.test" % self.getPath(), 'eval')
        self.sentences = compile(testDict['sentences'], "%s.test" % self.getPath(), 'eval')
        self.success = testDict['success']
        self.failure = testDict['failure']

    def getEncoder(self, moduleDict):
        bindings = dict(moduleDict)
        clean_vocab = bindings["tokenizer"].clean_vocab

        vocab = eval(self.vocab, bindings)
        merges = eval(self.merges, bindings)
        clean_vocab(vocab, merges)

        tokenizer = Tokenizer(tokenizers.models.BPE(vocab, merges))
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space = False)
        tokenizer.decoder = tokenizers.decoders.ByteLevel()

        return tokenizer.encode
    
    def sanity_check(self, grades, moduleDict):
        bindings = dict(moduleDict)
        clean_vocab = bindings["tokenizer"].clean_vocab

        old_vocab = eval(self.vocab, bindings)
        old_merges = eval(self.merges, bindings)

        vocab = eval(self.vocab, bindings)
        merges = eval(self.merges, bindings)
        clean_vocab(vocab, merges)

        # vocab items must be sequential
        vocab_items = sorted(vocab.items(), key=lambda x: x[1])
        j = -1
        for i, (k, v) in enumerate(vocab_items):
            if i != v:
                grades.addMessage('FAIL: %s' % self.path)
                grades.addMessage('\tVocab index must be sequential.')
                grades.addMessage('\tThe index of %s should be %d, but get %d instead.' % (k, i, v))
                return False
            if k not in old_vocab:
                grades.addMessage('FAIL: %s' % self.path)
                grades.addMessage('\tCleaned vocab must be a subset of the original vocab.')
                grades.addMessage('\tToken %s does not exist in the original vocab.' % (k))
                return False
            if old_vocab[k] <= j:
                grades.addMessage('FAIL: %s' % self.path)
                grades.addMessage('\tCleaned vocab must follow the order of the original vocab.')
                grades.addMessage('\tToken %s has index %d in the original vocab, which is no larger than the previous one %d.' % (k, old_vocab[k], j))
                return False
            j = old_vocab[k]

        # merges should not repeat
        counter = util.Counter()
        j = -1
        for merge in merges:
            counter[merge] += 1
            if counter[merge] > 1:
                grades.addMessage('FAIL: %s' % self.path)
                grades.addMessage('\tMerge %s appears multiple times.' % (str(merge)))
                return False
            if merge not in old_merges:
                grades.addMessage('FAIL: %s' % self.path)
                grades.addMessage('\tMerge %s does not appear in orginal merges.' % (str(merge)))
                return False
            idx = old_merges.index(merge)
            if idx <= j:
                grades.addMessage('FAIL: %s' % self.path)
                grades.addMessage('\tMerge %s has index %d in the original merges, which does not follow the original order.' % (str(merge), idx))
                return False
            j = idx
        
        return True

    def execute(self, grades, moduleDict, solutionDict):
        if not self.sanity_check(grades, moduleDict):
            return False

        encoder = self.getEncoder(moduleDict)
        sentences = eval(self.sentences)
        results = eval(solutionDict['results'])

        for sentence, result in zip(sentences, results):

            output = ' '.join(encoder(sentence).tokens)

            if output != result:
                grades.addMessage('FAIL: %s' % self.path)
                grades.addMessage('\t%s' % self.failure)
                grades.addMessage('\t      sentence: "%s"' % sentence)
                grades.addMessage('\tstudent result: "%s"' % output)
                grades.addMessage('\tcorrect result: "%s"' % result)
                return False
            
        grades.addMessage('PASS: %s' % self.path)
        grades.addMessage('\t%s' % self.success)
        return True

    def writeSolution(self, moduleDict, filePath):
        encoder = self.getEncoder(moduleDict)
        sentences = eval(self.sentences)
        results = [' '.join(encoder(sentence).tokens) for sentence in sentences]

        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('# The tokenization of each sentence must equal the below.\n')

            handle.write('results: """\n%s\n"""\n' % str(results))
        return True



class Regex(object):
    def __init__(self, pattern, vocab=None, use_dfa=True):
        """
        regex for testing. convert to dfa might be extremely time-consuming under thread protection.
        """
        self.pattern = pattern
        self.vocab = list(map(chr, range(127))) if vocab is None else list(vocab)
        self.use_dfa = use_dfa
        
        # prepare the jump table
        nfa_start_node = compile_to_nfa(self.pattern, self.vocab)
        if use_dfa:
            dfa_list, jump_table = convert_to_dfa(nfa_start_node, self.vocab)
            self.start_state, self.jump_table = minimize_dfa(dfa_list, jump_table, self.vocab)
        else:
            self.nfa_start_node = nfa_start_node

    def match(self, input_sequence) -> bool:
        """
        Try to apply the pattern to the whole sequence. Return True if matched, False otherwise. 
        """
        if self.use_dfa:

            cur_status = self.start_state
            for c in input_sequence:
                jump_dict = self.jump_table[cur_status]
                if jump_dict:
                    js = jump_dict.get(c)
                    if js is None:
                        return False
                    else:
                        cur_status = js
            return self.jump_table[cur_status].get('accepted', False)
        
        else:

            start_node = self.nfa_start_node

            current_nfa_set = [start_node]
            next_nfa_set = Closure.closure(current_nfa_set)

            for i, ch in enumerate(input_sequence):
                current_nfa_set = Closure.move(next_nfa_set, ch)
                next_nfa_set = Closure.closure(current_nfa_set)

                if next_nfa_set is None:
                    return False
                
            for nfa in next_nfa_set:
                if nfa.next_1 is None and nfa.next_2 is None:
                    return True

            return False


class RegexTest(testClasses.TestCase):

    def __init__(self, name, question, testDict):
        super(RegexTest, self).__init__(name, question, testDict)
        self.ns = compile(testDict['ns'], "%s.test" % self.getPath(), 'eval')
        self.special = compile(testDict['special'], "%s.test" % self.getPath(), 'eval')
        self.success = testDict['success']
        self.failure = testDict['failure']

        # initialize random (to prevent hacking)
        self.random = util.random.Random()
        self.random.seed()

    def tokenize(self, n: int, special_char: bool):
        """a dummy tokenizer."""
        result = list(str(n))
        if special_char: result[0] = 'Ġ' + result[0]
        return result

    def execute(self, grades, moduleDict, solutionDict):
        # preparation
        bindings = dict(moduleDict)
        multiple_of_n = bindings["regexps"].multiple_of_n
        ns = eval(self.ns, bindings)
        special = eval(self.special, bindings)
        vocab = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Ġ0', 'Ġ1', 'Ġ2', 'Ġ3', 'Ġ4', 'Ġ5', 'Ġ6', 'Ġ7', 'Ġ8', 'Ġ9'] + list(Tokens.keys())
        weird = {
            True: [ [], ['1', '2'], ['Ġ1', 'Ġ2'], ['1', 'Ġ2'], ['Ġ1', 'Ġ2', '0'], ['Is', '2'], ['Ġ2', 'Ġof'], ['Ġ2', '1', 'Ġof', '6'], ['0'] ],
            False: [ [], ['Ġ1', '2'], ['Ġ1', 'Ġ2'], ['Ġ1', 'Ġ2', '0'], ['1', 'Ġ2'], ['Is', '2'], ['2', 'Ġof'], ['2', '1', 'Ġof', '6'], ['Ġ0'] ]
        }[special]
        dfa_nums = { (2, False): 2, (3, False): 4, (4, False): 3, (5, False): 2, (6, False): 4, (7, False): 8, (2, True): 3, (3, True): 4, (4, True): 4, (5, True): 3 }

        for n in ns:
            pattern = multiple_of_n(n, special)

            # sanity check
            if not isinstance(pattern, list) or \
                not all(isinstance(item, str) for item in pattern):
                grades.addMessage('FAIL: %s' % self.path)
                grades.addMessage(f'\tThe function `multiple_of_n` must return a list of strings. Arguments: {n}, {special}')
                return False
            if not all((item in vocab) for item in pattern):
                idx = [(item in vocab) for item in pattern].index(False)
                grades.addMessage('FAIL: %s' % self.path)
                grades.addMessage(f'\tThe function `multiple_of_n` must use these tokens: {vocab}, but {pattern[idx]} found. Arguments: {n}, {special}')
                return False

            # do tests
            use_dfa = (n <= 5 or not special)
            regex = Regex(pattern, vocab, use_dfa=use_dfa)

            # check 0 to 1000 and larger random numbers
            numbers = list(range(1000)) + sorted([self.random.randint(10**i-1, 10**(i+1)-1) for _ in range(15) for i in range(4, 12)])
    
            for number in numbers:
                result = regex.match(self.tokenize(number, special))
                expected = number % n == 0

                if result != expected:
                    grades.addMessage('FAIL: %s' % self.path)
                    grades.addMessage('\t%s' % self.failure)
                    grades.addMessage(f'\targuments: n = {n}, special_char = {special}, input = {self.tokenize(number, special)}')
                    grades.addMessage('\tstudent result: "%s"' % result)
                    grades.addMessage('\tcorrect result: "%s"' % expected)
                    return False
            
            # weird tests
            for test in weird:
                result = regex.match(test)
                expected = False

                if result != expected:
                    grades.addMessage('FAIL: %s' % self.path)
                    grades.addMessage('\tThe regex should reject invalid inputs.')
                    grades.addMessage(f'\targuments: n = {n}, special_char = {special}, input = {test}')
                    grades.addMessage('\tstudent result: "%s"' % result)
                    grades.addMessage('\tcorrect result: "%s"' % expected)
                    return False
            
            # test dfa nodes
            if use_dfa and len(regex.jump_table) != dfa_nums[(n, special)]:
                grades.addMessage('FAIL: %s' % self.path)
                grades.addMessage('\tNumber of DFA nodes is incorrect.')
                grades.addMessage(f'\targuments: n = {n}, special_char = {special}')
                grades.addMessage('\tstudent result: "%s"' % len(regex.jump_table))
                grades.addMessage('\tcorrect result: "%s"' % dfa_nums[(n, special)])
                if len(regex.jump_table) < 12:
                    grades.addMessage('\tDOT graph: "https://dreampuf.github.io/GraphvizOnline/#%s"' % urllib.parse.quote(dot(regex.jump_table)))
                    grades.addMessage('\tPlease visit the link above to view the finite automaton produced by your regular expression.')
                return False

        grades.addMessage('PASS: %s' % self.path)
        grades.addMessage('\t%s' % self.success)
        return True

    def writeSolution(self, moduleDict, filePath):
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('# File intentionally blank.\n')

        return True


class RegexLengthTest(testClasses.TestCase):

    def __init__(self, name, question, testDict):
        super(RegexLengthTest, self).__init__(name, question, testDict)
        self.threshold = compile(testDict['threshold'], "%s.test" % self.getPath(), 'eval')
        self.success = testDict['success']
        self.failure = testDict['failure']

    def execute(self, grades, moduleDict, solutionDict):
        # preparation
        bindings = dict(moduleDict)
        multiple_of_n = bindings["regexps"].multiple_of_n
        threshold = eval(self.threshold, bindings)

        length = sum(len(multiple_of_n(n, spec)) for n in range(2, 7) for spec in (True, False))

        if length <= threshold:
            grades.addMessage('PASS: %s' % self.path)
            grades.addMessage('\t%s' % self.success)
            grades.addMessage('\tstudent regex total length: "%s"' % length)
            grades.addMessage('\tthreshold: "%s"' % threshold)
            return True
        
        grades.addMessage('FAIL: %s' % self.path)
        grades.addMessage('\t%s' % self.failure)
        grades.addMessage('\tstudent regex total length: "%s"' % length)
        grades.addMessage('\tthreshold: "%s"' % threshold)
        return False

    def writeSolution(self, moduleDict, filePath):
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('# File intentionally blank.\n')

        return True
    

class ColdStartTest(testClasses.TestCase):
    def __init__(self, name, question, testDict):
        super(ColdStartTest, self).__init__(name, question, testDict)
        self.task = compile(testDict['task'], "%s.test" % self.getPath(), 'eval')
        self.vocab = compile(testDict['vocab'], "%s.test" % self.getPath(), 'eval')
        self.merges = compile(testDict['merges'], "%s.test" % self.getPath(), 'eval')
        self.hyperparams = compile(testDict['hyperparams'], "%s.test" % self.getPath(), 'eval')
        self.regexps = compile(testDict['regexps'], "%s.test" % self.getPath(), 'eval')
        self.labels = compile(testDict['labels'], "%s.test" % self.getPath(), 'eval')
        self.rules = compile(testDict['rules'], "%s.test" % self.getPath(), 'eval')
        self.sentences = compile(testDict['sentences'], "%s.test" % self.getPath(), 'eval')
        self.success = testDict['success']
        self.failure = testDict['failure']
    
    def prepare(self, bindings, task: str):
        FiniteAutomaton = bindings["models"].FiniteAutomaton
        regexps = eval(self.regexps, bindings)
        vocab_ = eval(self.vocab, bindings)
        merges = eval(self.merges, bindings)

        # resolve task
        if task == "dummy":
            vocab = {k: i for i, k in enumerate(vocab_)}
            tokenizer = Tokenizer(tokenizers.models.BPE(vocab, merges))
            tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space = False)
            tokenizer.decoder = tokenizers.decoders.ByteLevel()

            # generate automata
            automata = []
            for k, v in regexps.items():
                regex = Regex(v.split(" "), vocab_)
                automata.append(
                    FiniteAutomaton(expr(k), regex.start_state, regex.jump_table, vocab)
                )

        elif task == "div":
            vocab = {k: i for i, k in enumerate(vocab_)}
            tokenizer = Tokenizer(tokenizers.models.BPE(vocab, merges))
            tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space = False)
            tokenizer.decoder = tokenizers.decoders.ByteLevel()

            # generate automata
            automata = []
            for k, (start_state, jump_table) in regexps.items():
                automata.append(
                    FiniteAutomaton(expr(k), start_state, jump_table, vocab)
                )
            
        elif task == "atis":
            vocab = {k: i for i, k in enumerate(vocab_)}
            tokenizer = Tokenizer(tokenizers.models.WordLevel(vocab))
            tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

            # load regex
            if regexps == "atis":
                regexps = {}
                atis_flight = bindings["regexps"].atis_flight
                labels = eval(self.labels, bindings)
                for i, label in enumerate(labels):
                    regexps["L" + str(i)] = atis_flight(label)

            # generate automata
            automata = []
            for k, v in regexps.items():
                regex = Regex(v, vocab_)
                automata.append(
                    FiniteAutomaton(expr(k), regex.start_state, regex.jump_table, vocab)
                )
            
        else:
            raise ValueError(f"Unknown task: {task}")
        
        return tokenizer, vocab, automata

    def execute(self, grades, moduleDict, solutionDict):
        atol = 2e-1

        # preparation
        bindings = dict(moduleDict)
        FARNN = bindings["models"].FARNN
        task = eval(self.task, bindings)
        hyperparams = eval(self.hyperparams, bindings)
        labels = eval(self.labels, bindings)
        rules = eval(self.rules, bindings)
        sentences = eval(self.sentences, bindings)

        tokenizer, vocab, automata = self.prepare(bindings, task)

        # to print the automata for debugging
        automaton_debug = "\n"
        for automaton in automata:
            automaton_debug += '\tDOT graph of %s: "https://dreampuf.github.io/GraphvizOnline/#%s"\n' % (automaton.symbol, urllib.parse.quote(dot(automaton.jump_table)))
            
        # translate rules
        rules = [expr(rule) for rule in rules]
        
        # initialize the model
        model = FARNN(vocab, labels, **hyperparams)
        model.initialize(automata, rules)

        # sanity check
        if not isinstance(model.mlp, torch.nn.Sequential):
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\tself.mlp becomes a unexpected object. Try to use `self.mlp[0].weight.data = ...` to initalize its parameters.')
            return False

        # test
        for sentence, golds in zip(sentences, eval(solutionDict['results'], bindings)):
            x = torch.tensor(tokenizer.encode(sentence).ids)
            logits = model.forward_unbatched(x)
            for logit, gold, rule, label in zip(logits, golds, rules, labels):
                expected = 1.0 if gold else 0.0
                if abs(logit.item() - expected) > atol:
                    # wrong answer
                    grades.addMessage('FAIL: %s' % self.path)
                    grades.addMessage('\t%s' % self.failure)
                    grades.addMessage('\tInput: %s' % tokenizer.encode(sentence).tokens)
                    grades.addMessage('\tLabel: %s' % label)
                    grades.addMessage('\tRule: %s' % str(rule))
                    # grades.addMessage('\tAutomata: %s' % automaton_debug) # too long to display
                    grades.addMessage('\tstudent result: "%s"' % logit.item())
                    grades.addMessage('\tcorrect result: "%s +- %s"' % (expected, atol))
                    return False

        grades.addMessage('PASS: %s' % self.path)
        grades.addMessage('\t%s' % self.success)
        return True

    def writeSolution(self, moduleDict, filePath):
        # preparation
        bindings = dict(moduleDict)
        task = eval(self.task, bindings)
        rules = eval(self.rules, bindings)
        sentences = eval(self.sentences, bindings)

        tokenizer, vocab, automata = self.prepare(bindings, task)
            
        # translate rules
        rules = [expr(rule) for rule in rules]

        # tokenize
        sentences = [tokenizer.encode(sentence).tokens for sentence in sentences]
        
        # matching
        results = []
        for sentence in sentences:
            matchings = {
                automaton.symbol: automaton.match(sentence)
                for automaton in automata
            }
            candidates = []
            for rule in rules:
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
                candidates.append(value)
            results.append(candidates)

        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('# The expected matching result.\n')

            handle.write('results: "%s"\n' % str(results))

        return True

# Hack test case checks the md5 of the code of the function
class HackTest(HiddenTest):

    def evalCode(self, moduleDict):
        import inspect
        bindings = dict(moduleDict)
        # exec self.preamble in bindings
        exec(self.preamble, bindings)
        return util.md5(inspect.getsource(eval(self.test, bindings)))


class DIVDataset(torch.utils.data.Dataset):
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

class TrainerTest(testClasses.TestCase):
    def __init__(self, name, question, testDict):
        super(TrainerTest, self).__init__(name, question, testDict)
        self.train_path = compile(testDict['train_path'], "%s.test" % self.getPath(), 'eval')
        self.test_path = compile(testDict['test_path'], "%s.test" % self.getPath(), 'eval')
        self.threshold = compile(testDict['threshold'], "%s.test" % self.getPath(), 'eval')
        self.success = testDict['success']
        self.failure = testDict['failure']
    
    def build_tokenizer(self, clean_vocab, path_to_train):
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
    
    @torch.no_grad()
    def test(self, test_dataset: torch.utils.data.Dataset, model: torch.nn.Module):
        test_loss, correct = 0, 0
        model.eval()
        for seq, label in test_dataset:
            seq, label = torch.tensor(seq), torch.tensor(label)
            logit = model.forward_unbatched(seq)
            test_loss += model.calculate_loss(logit, label).item()
            correct += (logit.argmax(0) == label).type(torch.float).sum().item()
        
        test_loss /= len(test_dataset)
        correct /= len(test_dataset)

        return test_loss, correct

    def execute(self, grades, moduleDict, solutionDict):
        # preparation
        bindings = dict(moduleDict)
        get_farnn_kwargs = bindings["trainer"].get_farnn_kwargs
        Trainer = bindings["trainer"].Trainer
        clean_vocab = bindings["tokenizer"].clean_vocab
        FARNN = bindings["models"].FARNN
        train_path = eval(self.train_path, bindings)
        test_path = eval(self.test_path, bindings)
        threshold = eval(self.threshold, bindings)

        # tokenize
        print("Tokenizer training...")
        tokenize, vocab = self.build_tokenizer(clean_vocab, train_path)
        train_dataset = DIVDataset(train_path, tokenize)
        test_dataset = DIVDataset(test_path, tokenize)
        labels = ['Yes', 'No']

        # training
        model = FARNN(vocab, labels, **get_farnn_kwargs())
        print("Model initializing...")
        trainer = Trainer(model)
        print("Model training...")
        trainer.train(train_dataset)

        # testing
        print("Model testing...")
        test_loss, accuracy = self.test(test_dataset, model)
        if accuracy < threshold:
            grades.addMessage('FAIL: %s' % self.path)
            grades.addMessage('\t%s' % self.failure)
            grades.addMessage('\tTest Accuracy: %.2f%%' % (accuracy*100))
            grades.addMessage('\tTest Loss: %f' % test_loss)
            return False

        grades.addMessage('PASS: %s' % self.path)
        grades.addMessage('\t%s' % self.success)
        grades.addMessage('\tTest Accuracy: %.2f%%' % (accuracy*100))
        grades.addMessage('\tTest Loss: %f' % test_loss)
        return True

    def writeSolution(self, moduleDict, filePath):
        with open(filePath, 'w') as handle:
            handle.write('# This is the solution file for %s.\n' % self.path)
            handle.write('# File intentionally blank.\n')

        return True