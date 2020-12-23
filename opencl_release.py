#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import os

import numpy as np
import pyopencl as cl
import pyopencl.array as clar
from serial_release import get_divisors

import common

# complier output (logs)
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'


def create_context(device):
    platforms = cl.get_platforms()
    print(platforms, ' | ', platforms[0].get_devices(device_type=cl.device_type.CPU))
    if device == 'cpu':
        # TODO: Win index: 1
        devices = platforms[0].get_devices(device_type=cl.device_type.CPU)
    else:
        devices = platforms[0].get_devices(device_type=cl.device_type.GPU)
    print(devices)
    context = cl.Context(devices=devices)

    with open('roots.c', 'rt') as f:
        roots_program_text = f.read()

    with open('divisors.c', 'rt') as f:
        divisors_program_text = f.read()

    program_divisors = cl.Program(context, divisors_program_text).build()
    program_roots = cl.Program(context, roots_program_text).build()

    return context, program_divisors, program_roots


def generate_possible_roots(context, coefficients, program_divisors):
    int_coefficients = common.coefficients_conversions(coefficients)

    queue = cl.CommandQueue(context)

    lowest_term = abs(
        [coefficient for coefficient in int_coefficients[::-1] if coefficient.numerator != 0][0].numerator)
    highest_term = abs(int_coefficients[0].numerator)

    # memory allocation for OpenCL
    low_input = np.array(range(1, math.floor(math.sqrt(lowest_term)) + 1))
    high_input = np.array(range(1, math.floor(math.sqrt(highest_term)) + 1))

    low_m_out = np.zeros(2 * len(low_input))
    low_p_out = np.zeros(2 * len(low_input))
    high_out = np.zeros(2 * len(high_input))

    flags_low = np.array([lowest_term, 1, len(low_m_out)])  # Здесь флаги, само число n или m и длина output'a
    flags_high = np.array([highest_term, 0, len(high_out)])

    cl_l_i = clar.to_device(queue, low_input.astype(np.int32))
    cl_h_i = clar.to_device(queue, high_input.astype(np.int32))
    cl_f_l = clar.to_device(queue, flags_low.astype(np.int32))
    cl_f_h = clar.to_device(queue, flags_high.astype(np.int32))
    cl_lm_o = clar.to_device(queue, low_m_out.astype(np.int32))
    cl_lp_o = clar.to_device(queue, low_p_out.astype(np.int32))
    cl_h_o = clar.to_device(queue, high_out.astype(np.int32))

    program_divisors.divisors(queue, [len(low_input)], None, cl_l_i.data, cl_f_l.data, cl_lp_o.data, cl_lm_o.data)
    program_divisors.divisors(queue, [len(high_input)], None, cl_h_i.data, cl_f_h.data, cl_h_o.data, None)

    high_out = cl_h_o.get()
    low_out = cl_lp_o.get()
    high = np.delete(high_out, np.where(high_out == [0]), axis=0)
    low = np.delete(low_out, np.where(low_out == [0]), axis=0)

    if int_coefficients[-1] == 0:
        low = np.append(low, 0)

    possible_roots = list()
    for divisor_low in low:
        for divisor_high in high:
            possible_roots.append([divisor_low, divisor_high])
            if divisor_low:
                possible_roots.append([-divisor_low, divisor_high])
    return np.array(possible_roots)


def solve(context, coefficients, program_divisors, program_roots, profile=False, statistics_file_path=None):
    possible_roots = generate_possible_roots(context, coefficients, program_divisors)
    input_t = np.array([t for t, s in possible_roots])
    input_s = np.array([s for t, s in possible_roots])
    output_t = np.zeros(input_t.shape)
    output_s = np.zeros(input_s.shape)

    queue = cl.CommandQueue(context, properties=cl.command_queue_properties.PROFILING_ENABLE)

    t = clar.to_device(queue, input_t.astype(np.float32))
    s = clar.to_device(queue, input_s.astype(np.float32))
    kf = clar.to_device(queue, coefficients.astype(np.int32))
    deg = clar.to_device(queue, np.array([len(coefficients)]).astype(np.int32))
    res_t = clar.to_device(queue, output_t.astype(np.int32))
    res_s = clar.to_device(queue, output_s.astype(np.int32))

    if profile:
        workitems_groups = get_divisors(len(possible_roots), False)
        statistics_file = open(statistics_file_path, 'wt')
        for workitems_len in workitems_groups:
            iter_param = np.array([len(possible_roots) / workitems_len])
            cl_iter_param = clar.to_device(queue, iter_param.astype(np.int32))
            event = program_roots.roots(queue, [workitems_len], None, t.data, s.data, kf.data, deg.data, res_t.data,
                                        res_s.data, cl_iter_param.data)
            event.wait()
            elapsed = 1e-9 * (event.profile.end - event.profile.start)
            statistics_file.write(f"{workitems_len} {elapsed}\n")
        statistics_file.close()
    else:
        iter_param = np.array([1])
        cl_iter_param = clar.to_device(queue, iter_param.astype(np.int32))
        program_roots.roots(queue, [1], None, t.data, s.data, kf.data, deg.data, res_t.data, res_s.data,
                            cl_iter_param.data)

    roots = list()
    for t, s in zip(res_t, res_s):
        if s != 0:
            roots.append(f"{t}/{s}")
    return roots, len(possible_roots)


def main(args):
    context, program_divisors, program_roots = create_context(args.device)

    if args.mode == 'auto':
        with open(args.file, 'rt') as f:
            file = f.read()
        polynoms = file.split('\n')
    else:  # manual mode
        coefficients_raw = input('Введите коэффициенты уравнения: ')
        polynoms = [coefficients_raw]

    for polynom in polynoms:
        coefficients = common.get_coefficients(polynom)
        if args.statistics:
            roots, count_of_possible_roots = solve(context, coefficients, program_divisors, program_roots, True,
                                                   args.statistics)
        else:
            roots, count_of_possible_roots = solve(context, coefficients, program_divisors, program_roots)
        if roots:
            print(f"Найдены корни ({common.stringify_polynom(coefficients)}=0): {common.stringify_roots(roots)}")
        else:
            print(f"Целых или рациональных корней не найдено ({common.stringify_polynom(coefficients)}=0).")


if __name__ == "__main__":
    __version__ = '0.0.1'
    parser = common.create_args_parser(__version__, parallel=True)
    args = parser.parse_args()
    main(args)
