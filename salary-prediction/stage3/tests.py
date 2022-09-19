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
        check_outputs_number(5, user_values)

        coefs = [1187791.2641789438, 246170.17905994324, 430020.2213681233, 182762.61279640213, -87689.58520293701]
        check_num_values(coefs, user_values,
                              "Incorrect coefficients.\n"
                              "Make sure that you provide numbers in the correct order.\n"
                              "Also note that you have to use random_seed=100 and test_size=0.3 in train_test_split function to get correct results.",
                              rel_tol=1.0e-2)

        return CheckResult.correct()


if __name__ == '__main__':
    Stage3Test().run_tests()
