# autograder.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
# 
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)

# autograder.py
# -------------
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


# imports from python standard library
import grading
import optparse
import importlib.util
import os
import re
import sys
import projectParams
import random
random.seed(0)

# register arguments and set default values
def readCommand(argv):
    parser = optparse.OptionParser(description = 'Run public tests on student code')
    parser.set_defaults(generateSolutions=False, autolabOutput=False, gsOutput=False, muteOutput=False, printTestCase=False)
    parser.add_option('--test-directory',
                      dest = 'testRoot',
                      default = 'test_cases',
                      help = 'Root test directory which contains subdirectories corresponding to each question')
    parser.add_option('--student-code',
                      dest = 'studentCode',
                      default = projectParams.STUDENT_CODE_DEFAULT,
                      help = 'comma separated list of student code files')
    parser.add_option('--code-directory',
                    dest = 'codeRoot',
                    default = "",
                    help = 'Root directory containing the student and testClass code')
    parser.add_option('--test-case-code',
                      dest = 'testCaseCode',
                      default = projectParams.PROJECT_TEST_CLASSES,
                      help = 'class containing testClass classes for this project')
    parser.add_option('--generate-solutions',
                      dest = 'generateSolutions',
                      action = 'store_true',
                      help = 'Write solutions generated to .solution file')
    parser.add_option('--autolab-output',
                    dest = 'autolabOutput',
                    action = 'store_true',
                    help = 'Generate Autolab output files')
    parser.add_option('--gradescope-output',
                    dest = 'gsOutput',
                    action = 'store_true',
                    help = 'Generate GradeScope output files')
    parser.add_option('--mute', '-m',
                    dest = 'muteOutput',
                    action = 'store_true',
                    help = 'Mute output from executing tests')
    parser.add_option('--print-tests', '-p',
                    dest = 'printTestCase',
                    action = 'store_true',
                    help = 'Print each test case before running them.')
    parser.add_option('--test', '-t',
                      dest = 'runTest',
                      default = None,
                      help = 'Run one particular test.  Relative to test root.')
    parser.add_option('--question', '-q',
                    dest = 'gradeQuestion',
                    default = None,
                    help = 'Grade one particular question.')
    (options, args) = parser.parse_args(argv)
    return options


# confirm we should author solution files
def confirmGenerate():
    print('WARNING: this action will overwrite any solution files.')
    print('Are you sure you want to proceed? (yes/no)')
    while True:
        ans = sys.stdin.readline().strip()
        if ans == 'yes':
            break
        elif ans == 'no':
            sys.exit(0)
        else:
            print('please answer either "yes" or "no"')


def loadModuleFile(moduleName, filePath):
    spec = importlib.util.spec_from_file_location(moduleName, filePath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def readFile(path, root=""):
    "Read file from disk at specified path and return as string"
    with open(os.path.join(root, path), 'r') as handle:
        return handle.read()


def printTest(testDict, solutionDict):
    print("Test case:")
    for line in testDict["__raw_lines__"]:
        print("   |", line)
    print("Solution:")
    for line in solutionDict["__raw_lines__"]:
        print("   |", line)


def runTest(testName, moduleDict, printTestCase=False):
    import testParser
    import testClasses
    for module in moduleDict:
        setattr(sys.modules[__name__], module, moduleDict[module])

    testDict = testParser.TestParser(testName + ".test").parse()
    solutionDict = testParser.TestParser(testName + ".solution").parse()
    test_out_file = os.path.join('%s.test_output' % testName)
    testDict['test_out_file'] = test_out_file
    testClass = getattr(projectTestClasses, testDict['class'])

    questionClass = getattr(testClasses, 'Question')
    question = questionClass({'max_points': 0})
    testCase = testClass(testName, question, testDict)

    if printTestCase:
        printTest(testDict, solutionDict)

    # This is a fragile hack to create a stub grades object
    grades = grading.Grades(projectParams.PROJECT_NAME, [(None,testClasses.Question({'max_points':0}))])
    testCase._execute(grades, moduleDict, solutionDict)


# returns all the tests you need to run in order to run question
def getDepends(testParser, testRoot, question):
    allDeps = [question]
    questionDict = testParser.TestParser(os.path.join(testRoot, question, 'CONFIG')).parse()
    if 'depends' in questionDict:
        depends = questionDict['depends'].split()
        for d in depends:
            # run dependencies first
            allDeps = getDepends(testParser, testRoot, d) + allDeps
    return allDeps

# get list of questions to grade
def getTestSubdirs(testParser, testRoot, questionToGrade):
    problemDict = testParser.TestParser(os.path.join(testRoot, 'CONFIG')).parse()
    if questionToGrade != None:
        questions = getDepends(testParser, testRoot, questionToGrade)
        if len(questions) > 1:
            print('Note: due to dependencies, the following tests will be run: %s' % ' '.join(questions))
        return questions
    if 'order' in problemDict:
        return problemDict['order'].split()
    return sorted(os.listdir(testRoot))

# get exception map
def getExceptionMap(testParser, testRoot):
    problemDict = testParser.TestParser(os.path.join(testRoot, 'CONFIG')).parse()
    if 'exceptionMap' in problemDict:
        return eval(problemDict['exceptionMap'])
    return {}

# evaluate student code
def evaluate(generateSolutions, testRoot, moduleDict,
             autolabOutput=False, muteOutput=False, gsOutput=False,
            printTestCase=False, questionToGrade=None):
    # imports of testbench code.  note that the testClasses import must follow
    # the import of student code due to dependencies
    import testParser
    import testClasses
    for module in moduleDict:
        setattr(sys.modules[__name__], module, moduleDict[module])

    questions = []
    questionDicts = {}
    test_subdirs = getTestSubdirs(testParser, testRoot, questionToGrade)
    for q in test_subdirs:
        subdir_path = os.path.join(testRoot, q)
        if not os.path.isdir(subdir_path) or q[0] == '.':
            continue

        # create a question object
        questionDict = testParser.TestParser(os.path.join(subdir_path, 'CONFIG')).parse()
        questionClass = getattr(testClasses, questionDict['class'])
        question = questionClass(questionDict)
        questionDicts[q] = questionDict

        # load test cases into question
        tests = filter(lambda t: re.match('[^#~.].*\.test\Z', t), os.listdir(subdir_path))
        tests = map(lambda t: re.match('(.*)\.test\Z', t).group(1), tests)
        for t in sorted(tests):
            test_file = os.path.join(subdir_path, '%s.test' % t)
            solution_file = os.path.join(subdir_path, '%s.solution' % t)
            test_out_file = os.path.join(subdir_path, '%s.test_output' % t)
            testDict = testParser.TestParser(test_file).parse()
            if testDict.get("disabled", "false").lower() == "true":
                continue
            testDict['test_out_file'] = test_out_file
            testClass = getattr(projectTestClasses, testDict['class'])
            testCase = testClass(t, question, testDict)
            def makefun(testCase, solution_file):
                if generateSolutions:
                    # write solution file to disk
                    return lambda grades: testCase.writeSolution(moduleDict, solution_file)
                else:
                    # read in solution dictionary and pass as an argument
                    testDict = testParser.TestParser(test_file).parse()
                    solutionDict = testParser.TestParser(solution_file).parse()
                    if printTestCase:
                        return lambda grades: printTest(testDict, solutionDict) or testCase._execute(grades, moduleDict, solutionDict)
                    else:
                        return lambda grades: testCase._execute(grades, moduleDict, solutionDict)
            question.addTestCase(testCase, makefun(testCase, solution_file))

        # Note extra function is necessary for scoping reasons
        def makefun(question):
            return lambda grades: question._execute(grades)
        setattr(sys.modules[__name__], q, makefun(question))
        questions.append((q, question))

    grades = grading.Grades(projectParams.PROJECT_NAME, questions,
                            gsOutput=gsOutput, autolabOutput=autolabOutput, muteOutput=muteOutput)
    if questionToGrade == None:
        for q in questionDicts:
            for prereq in questionDicts[q].get('depends', '').split():
                grades.addPrereq(q, prereq)

    grades.grade(sys.modules[__name__], exceptionMap=getExceptionMap(testParser, testRoot))
    return grades.points


if __name__ == '__main__':
    options = readCommand(sys.argv)
    if options.generateSolutions:
        confirmGenerate()
    codePaths = options.studentCode.split(',')

    moduleDict = {}
    for cp in codePaths:
        moduleName = re.match('.*?([^/]*)\.py', cp).group(1)
        moduleDict[moduleName] = loadModuleFile(moduleName, os.path.join(options.codeRoot, cp))
    moduleName = re.match('.*?([^/]*)\.py', options.testCaseCode).group(1)
    moduleDict['projectTestClasses'] = loadModuleFile(moduleName, os.path.join(options.codeRoot, options.testCaseCode))

    if options.runTest != None:
        runTest(options.runTest, moduleDict, printTestCase=options.printTestCase)
    else:
        evaluate(options.generateSolutions, options.testRoot, moduleDict, gsOutput=options.gsOutput,
            autolabOutput=options.autolabOutput, muteOutput=options.muteOutput, printTestCase=options.printTestCase,
            questionToGrade=options.gradeQuestion)
