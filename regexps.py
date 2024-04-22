# regexps.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to ShanghaiTech University, including a link 
# to https://i-techx.github.io/iTechX/courses?course_code=CS274A
# 
# Attribution Information: The NLP projects were developed at ShanghaiTech University.
# The core projects and autograders were adapted by Haoyi Wu (wuhy1@shanghaitech.edu.cn)


from typing import List
from regex import Regex
import util


def atis_flight(intent: str) -> List[str]:
    """
    Generate a regular expression that captures the intent of booking a flight in the ATIS dataset.
    These regular expressions are adapted from the code of the orginal paper.

    Args:
        intent (:obj:`str`):
            The intent of booking a flight. It is one of the intents in the ATIS dataset.
    """
    if intent == "abbreviation":
        return "( . * ( mean | ( stand | stands ) for | code ) . * ) | ( . * what is . )".split(" ")
    if intent == "aircraft":
        return ". * ( aircraft | airplane | plane | planes | airplanes ) . *".split(" ")
    if intent == "aircraft+flight+flight_no":
        return ". * ( aircraft | airplane | plane | planes | airplanes ) . * flight number . *".split(" ")
    if intent == "airfare":
        return ". * ( airfare | fares | fare | cost | costs | class ( ticket | ticket ) | how much | ( price | prices ) of  ) . * ( ( from . * to . * ) | ( between . * and . * ) ) . *".split(" ")
    if intent == "airfare+flight":
        return ". * airfare .* flight . *".split(" ")
    if intent == "airfare+flight_time":
        return ". * ( airfare | airfares | cost | how much ) . * flight time . *".split(" ")
    if intent == "airline":
        return ". * ( airlines | airline ) . *".split(" ")
    if intent == "airline+flight_no":
        return ". * ( airlines | airline ) . * flight ( number | numbers) . *".split(" ")
    if intent == "airport":
        return ". * ( airport | airports ) . *".split(" ")
    if intent == "capacity":
        return ". * ( ( capacities | capacity ) | how many ( seats | passengers | people ) | number of ( seats | passengers | people ) ) . *".split(" ")
    if intent == "cheapest":
        return ". * cheapest fare in the database . *".split(" ")
    if intent == "city":
        return ". * ( where is | cities | city ) . *".split(" ")
    if intent == "day_name":
        return ". * what day . *".split(" ")
    if intent == "distance":
        return ". * ( how far | how long | distance ) . *".split(" ")
    if intent == "flight":
        return ". * ( flights | flight | ( ( go | get | fly ) from . * to . * ) ) . *".split(" ")
    if intent == "flight+airfare":
        return ". * ( flights | flight ) and ( fare | fares ) . *".split(" ")
    if intent == "flight+airline":
        return ". * ( flight and airlines ) . *".split(" ")
    if intent == "flight_no":
        return ". * ( ( ( flight | flights ) number ) | ( number of ( flight | flights ) ) ) . *".split(" ")
    if intent == "flight_no+airline":
        return ". * flight ( number | numbers) . * ( airlines | airline ) . *".split(" ")
    if intent == "flight_time":
        return ". * ( schedule | schedules | ( ( what | departure | arrival | flight | flights ) . * ( time | times ) ) ) . *".split(" ")
    if intent == "ground_fare":
        return ". * ( price | prices | cost | costs ) . *  ( ( ( rent | rental ) . * ( car | cars ) ) | ( ( limousine | limo ) service ) ) . *".split(" ")
    if intent == "ground_service":
        return ". * ( ground ( transport | transportation ) | ( ( rent | rental ) . * ( car | cars ) ) | ( ( limousine | limo ) service ) ) . *".split(" ")
    if intent == "ground_service+ground_fare":
        return ". * ( ground ( transport | transportation ) ) . * how much . *".split(" ")
    if intent == "meal":
        return ". * ( meal | meals ) . *".split(" ")
    if intent == "quantity":
        return ". * ( how many  . * ( flights | flight | code | stops | airports | booking class | fares | fare ) ) . *".split(" ")
    if intent == "restriction":
        return ". * ( what . * ( restriction | restrictions ) ) . *".split(" ")
    return [".", "*"]


def multiple_of_n(n: int = 2, special_char: bool = True) -> List[str]:
    """
    Question:
        Generate a regular expression that captures a multiple of n. Notice that the pattern
        you return should be a list of tokens instead of a string.

        Remember that when a number is tokenized, the tokenizer may add a special character 
        `Ġ` in front of the first token, depending on where the number appears in the sentence.
        This information is passed through the argument `special_char`. That is, if the number
        to match is 12 and `special_char` is False, then the input sequence is ['1', '2']. If
        `special_char` is True, then the input sequence is ['Ġ1', '2']. The number to match is
        always non-negative.

        Notice that you may only use a subset of the regular expression grammar. You may use
        `|`, `()`, `[]`, `.`, `*`, `+`, `?`, `^` which has almost the same meaning as re 
        module in python.

        You do not need to worry about leading zeros, but empty strings should be rejected.

    Example:
        >>> vocab = ['Ġ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'Ġ0', 'Ġ1', 'Ġ2', 'Ġ3', 'Ġ4', 'Ġ5', 'Ġ6', 'Ġ7', 'Ġ8', 'Ġ9']
        >>> pattern = multiple_of_n(2, False)
        >>> Regex(pattern, vocab).match(['1', '1'])
        False
        >>> Regex(pattern, vocab).match(['1', '2'])
        True
        >>> pattern = multiple_of_n(3, True)
        >>> Regex(pattern, vocab).match(['Ġ1', '2'])
        True
        >>> Regex(pattern, vocab).match(['1', '2']) # invalid input
        False

    Args:
        n (:obj:`int`):
            A base number. The generated regular expression should accept all its multipliers
            including 0 and reject other numbers. 2 <= n <= 7.

        special_char (:obj:`bool`):
            True if the a special character `Ġ` is added in front of the first input token,
            False otherwise.
    """

    """YOUR CODE HERE"""
    util.raiseNotDefined()

    return ['[', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ']', '*', '[', '0', '2', '4', '6', '8', ']']


if __name__ == '__main__':
    print("Running regexps.py ...")
    pattern = multiple_of_n(2, special_char=False)
    regex = Regex(pattern)
    result = regex.match(['3', '2'])
    print(result) # True
