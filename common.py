import argparse
from fractions import Fraction
import numpy as np


def list_lcm(in_list):
    if len(in_list) == 2:
        return np.lcm(int(in_list[0]), int(in_list[1]))
    else:
        return list_lcm([in_list[0], list_lcm(in_list[1:])])


def get_denominators(x):
    return x.denominator


def coefficients_conversions(coefficients):
    divided_by_highest = coefficients / coefficients[0]
    vectorized_get_denominators = np.vectorize(get_denominators)
    coefficients_denominators = vectorized_get_denominators(divided_by_highest)
    coefficients_lcm = list_lcm(coefficients_denominators)
    int_coefficients = divided_by_highest * coefficients_lcm
    return int_coefficients


def get_coefficients(coefficients_raw):
    coefficients_dirty = coefficients_raw.strip().split(' ')
    coefficients = list()
    for coefficient in coefficients_dirty:
        if len(coefficient.split('/')) > 1:
            numerator, denominator = coefficient.split('/')
        else:
            numerator = coefficient
            denominator = 1
        coefficients.append(Fraction(int(numerator), int(denominator)))
    return np.array(coefficients)


def stringify_roots(roots):
    str_roots = ''
    for i, root in enumerate(roots):
        if i != 0 and i < len(roots):
            str_roots += ', '
        str_roots += root
    return str_roots


def stringify_polynom(coefficients):
    str_polynom = ''
    for i, coefficient in enumerate(coefficients):
        if coefficient != 0:
            if i > 0 and coefficient > 0:
                str_polynom += '+'
            degree = len(coefficients) - i - 1
            if coefficient != 1:
                str_polynom += f"{coefficient}"
            if degree:
                str_polynom += f"*x"
                if degree != 1:
                    str_polynom += f"^{degree}"

    return str_polynom


def create_args_parser(__version__, parallel=False):
    parser = argparse.ArgumentParser(add_help=False, prog='opencl_release.py',
                                     description='Программа для поиска корней многочленов')

    parser.add_argument('--statistics', '-s', metavar='statistics', help='Включить запись статистики',
                        dest='statistics')

    if parallel:
        calculate_group = parser.add_argument_group(title='Устройство вычисления')
        calculate_group.add_argument('--device', '-d', metavar='device',
                                     choices=['cpu', 'gpu'],
                                     help='Параметр, определяющий устройство вычисления - GPU/CPU.',
                                     required=True)

    subparser_mode = parser.add_subparsers(dest='mode', title='Режим', description='Режим ввода данных')
    auto_parser = subparser_mode.add_parser('auto', add_help=False, help='Автоматический вввод')
    auto_group = auto_parser.add_argument_group(title='Параметры автоматического ввода')
    auto_group.add_argument('--file', '-f', metavar='file',
                            help='Путь до файла', required=True)
    auto_parser = subparser_mode.add_parser('manual', add_help=False, help='Автоматический ввод')

    return parser
