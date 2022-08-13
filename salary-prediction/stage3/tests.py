import ast
from math import isclose

from hstest import CheckResult, StageTest, TestedProgram, WrongAnswer, dynamic_test


def is_float(num: str):
    try:
        float(num)
        return True
    except ValueError:
        return False


def check_outputs_number(values_number: int, user_output: list):
    if not all(is_float(output) for output in user_output):
        raise WrongAnswer(f"Answer '{user_output}' contains non-numeric values.")

    if len(user_output) != values_number:
        raise WrongAnswer(f"Answer contains {len(user_output)} values, but {values_number} values are expected.")


def check_num_values(values: list, user_values: list, message: str, rel_tol=1.0e-3):
    if not all(isclose(value, user_value, rel_tol=rel_tol) for value, user_value in zip(values, user_values)):
        raise WrongAnswer(message)


class Stage3Test(StageTest):
    @dynamic_test
    def test(self):
        pr = TestedProgram()
        user_output = pr.start().strip()

        if len(user_output) == 0:
            raise WrongAnswer("Seems like your program does not show any output.")

        try:
            user_values = ast.literal_eval(user_output)
        except Exception as e:
            return CheckResult.wrong(f"Seems that output is in wrong format.\n"
                                     f"Make sure you use only the following Python structures in the output: string, int, float, list, dictionary")

        # check output format
        check_outputs_number(45, user_values)

        coefs = [32117397772764.973, 32117398018157.273, 32117398269598.58, 32117402964389.008, 32117399826903.5,
                 32117401567730.46, 32117396675354.824, 32117399362740.29, 32117401134570.266, 32117398964229.594,
                 32117398341639.434, 32117398356103.402, 32117396188896.645, 32117395488106.39, 32117399859489.75,
                 32117399457934.55, 32117396779070.11, 32117401327634.668, 32117398938930.12, 32117398780209.14,
                 32117399703988.758, 32117401173169.266, 32117400266279.88, 32117397925410.52, 32117401284233.86,
                 32117401355109.05, 32117400020709.004, 32117398042138.367, 32117398742438.246, 32117397843994.734,
                 32117402187819.617, -5.221444578928567e+18, -5.221444578928263e+18, -5.221444578928272e+18,
                 -5.221444578927949e+18, -5.221444578931707e+18, -5.221444578928719e+18, -5.221444578928919e+18,
                 1.079815132586217e+19, 1.0798151325862996e+19, 1222523.783203125, 234981.28515625, 529309.6640625,
                 174318.921875, -118631.8505859375]
        check_num_values(coefs, user_values,
                              "Incorrect coefficients.\n"
                              "Make sure that you provide numbers in the correct order.\n"
                              "Also note that you have to use random_seed=100 and test_size=0.3 in train_test_split function to get correct results.",
                              rel_tol=1.0e-2)

        return CheckResult.correct()


if __name__ == '__main__':
    Stage3Test().run_tests()
