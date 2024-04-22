# grading.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
# 
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)

# grading.py
# ----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"Common code for autograders"

import json
import traceback
from collections import defaultdict
import util

import projectParams
import datetime
import pytz
TZ = pytz.timezone(projectParams.TIME_ZONE)

class Grades:
    "A data structure for project grades, along with formatting code to display them"

    def __init__(self, projectName, questions,
                 gsOutput=False, autolabOutput=False, muteOutput=False):
        """
        Defines the grading scheme for a project
          projectName: project name
          questionsAndMaxesDict: a list of (question name, max points per question)
        """
        self.questions = [el[0] for el in questions]
        self.maxes = {name: q.getMaxPoints() for name, q in questions}
        self.questionDict = {name: q for name, q in questions}
        self.points = Counter()
        self.messages = dict([(q, []) for q in self.questions])
        self.project = projectName
        self.start = datetime.datetime.now(TZ)
        self.finish = datetime.datetime.now(TZ) # dummy information
        self.sane = True  # Sanity checks
        self.currentQuestion = None  # Which question we're grading
        self.autolabOutput = autolabOutput
        self.gsOutput = gsOutput  # GradeScope output
        self.mute = muteOutput
        self.prereqs = defaultdict(set)

        print('Autograder transcript for %s' % self.project)
        print(self.start.strftime("Starting on %m-%d at %H:%M:%S"))

    def addPrereq(self, question, prereq):
        self.prereqs[question].add(prereq)

    def grade(self, gradingModule, exceptionMap={}):
        """
        Grades each question
          gradingModule: the module with all the grading functions (pass in with sys.modules[__name__])
        """

        completedQuestions = set([])
        for q in self.questions:
            print('\nQuestion %s' % q)
            print('=' * (9 + len(q)))
            print(flush=self.autolabOutput)
            self.currentQuestion = q

            incompleted = self.prereqs[q].difference(completedQuestions)
            if len(incompleted) > 0:
                prereq = incompleted.pop()
                self.addMessage('NOTE: Make sure to complete Question %s before working on Question %s,' % (prereq, q))
                self.addMessage('because Question %s builds upon your answer for Question %s.' % (q, prereq))
                self.addMessage('')
                if self.mute: util.unmutePrint()
                continue

            if self.mute: util.mutePrint()
            try:
                # Timeout wrapper is moved to Question class
                getattr(gradingModule, q)(self)  # Call the question's function
            except Exception as inst:  # originally, Exception, inst
                self.addExceptionMessage(q, inst, traceback)
                self.addErrorHints(exceptionMap, inst, q[1])
            except:
                self.fail('FAIL: Terminated with a string exception.')
            finally:
                if self.mute: util.unmutePrint()

            if self.points[q] >= self.maxes[q]:
                completedQuestions.add(q)

            print('\n### Question %s: %d/%d ###\n' % (q, self.points[q], self.maxes[q]))

        self.finish = datetime.datetime.now(TZ)
        print(self.finish.strftime("\nFinished at %H:%M:%S"))
        print("\nProvisional grades\n==================")

        for q in self.questions:
            print('Question %s: %d/%d' % (q, self.points[q], self.maxes[q]))
        print('------------------')
        print('Total: %d/%d' % (self.points.totalCount(), sum(self.maxes.values())))
        
        if not self.autolabOutput and not self.gsOutput:
            print("""
Your grades are NOT yet registered.  To register your grades, make sure
to follow your instructor's guidelines to receive credit on your project.
""")

        if self.autolabOutput:
            print("""
If you see this, your grades are already registered.
""")
            self.produceAutolabOutput()
        if self.gsOutput:
            self.produceGradeScopeOutput()

    def addExceptionMessage(self, q, inst, traceback):
        """
        Method to format the exception message, this is more complicated because
        we need to html.escape the traceback but wrap the exception in a <pre> tag
        """
        self.fail('FAIL: Exception raised: %s' % inst)
        self.addMessage('')
        for line in traceback.format_exc().split('\n'):
            self.addMessage(line)

    def addErrorHints(self, exceptionMap, errorInstance, questionNum):
        typeOf = str(type(errorInstance))
        questionName = 'q' + questionNum
        errorHint = ''

        # question specific error hints
        if exceptionMap.get(questionName):
            questionMap = exceptionMap.get(questionName)
            if (questionMap.get(typeOf)):
                errorHint = questionMap.get(typeOf)
        # fall back to general error messages if a question specific
        # one does not exist
        if (exceptionMap.get(typeOf)):
            errorHint = exceptionMap.get(typeOf)

        # dont include the HTML if we have no error hint
        if not errorHint:
            return ''

        for line in errorHint.split('\n'):
            self.addMessage(line)

    def produceGradeScopeOutput(self):
        out_dct = {}

        # total of entire submission
        total_possible = sum(self.maxes.values())
        total_score = sum(self.points.values())
        out_dct['score'] = total_score
        out_dct['max_score'] = total_possible
        out_dct['output'] = '\n'.join([
            "Autograder transcript for %s" % self.project,
            self.start.strftime("Starting on %m-%d at %H:%M:%S"),
            self.finish.strftime("Finished at %H:%M:%S"),
            "Total score (%d / %d)" % (total_score, total_possible)
        ])

        # individual tests
        tests_out = []
        for name in self.questions:
            test_out = {}
            # test name
            if 'name' in self.questionDict[name].questionDict:
                test_out['name'] = '{}: {}'.format(name, self.questionDict[name].questionDict['name'])
            else:
                test_out['name'] = name
            # test score
            test_out['score'] = self.points[name]
            test_out['max_score'] = self.maxes[name]
            # others
            out_list = [
                'Question %s' % name,
                '=' * (9 + len(name)),
                '',
                *['*** ' + message for message in self.messages[name]],
                '',
                '### Question %s: %d/%d ###\n' % (name, self.points[name], self.maxes[name])
            ]
            test_out['output'] = '\n'.join(out_list)
            tests_out.append(test_out)
        out_dct['tests'] = tests_out

        # file output
        with open('gradescope_response.json', 'w') as outfile:
            json.dump(out_dct, outfile)
        return

    def produceAutolabOutput(self):
        autolabOutput = '{"scores": {'
        autolabOutput += ', '.join(['"%s": %d' % (q, self.points[q]) for q in self.questions])
        autolabOutput += '}, "scoreboard": ['
        autolabOutput += ', '.join(['%d' % (self.points[q]) for q in self.questions])
        autolabOutput += ', %d]}' % (sum(self.points[q] for q in self.questions))
        print(autolabOutput)


    def fail(self, message, raw=False):
        "Sets sanity check bit to false and outputs a message"
        self.sane = False
        self.assignZeroCredit()
        self.addMessage(message, raw)

    def assignZeroCredit(self):
        self.points[self.currentQuestion] = 0

    def addPoints(self, amt):
        self.points[self.currentQuestion] += amt

    def deductPoints(self, amt):
        self.points[self.currentQuestion] -= amt

    def assignFullCredit(self, message="", raw=False):
        self.points[self.currentQuestion] = self.maxes[self.currentQuestion]
        if message != "":
            self.addMessage(message, raw)

    def addMessage(self, message, raw=False):
        if not raw:
            # We do not consider HTML formatting here
            if self.mute: util.unmutePrint()
            print('*** ' + message, flush=self.autolabOutput)
            if self.mute: util.mutePrint()
        self.messages[self.currentQuestion].append(message)


class Counter(dict):
    """
    Dict with default 0
    """

    def __getitem__(self, idx):
        try:
            return dict.__getitem__(self, idx)
        except KeyError:
            return 0

    def totalCount(self):
        """
        Returns the sum of counts for all keys.
        """
        return sum(self.values())
