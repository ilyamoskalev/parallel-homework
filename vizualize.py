import argparse

import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors


def prepare_data(path):
    x = list()
    y = list()
    with open(path, 'rt') as f:
        lines = f.readlines()
        for line in lines:
            x_str, y_str = line.split(' ')
            x.append(int(x_str))
            y.append(float(y_str))
    return x, y


def vizualize_data(data):
    plt.xlabel("Number of possible roots")
    plt.ylabel("Time of execution in seconds")

    num_colors = len(list(data.keys()))

    cm = plt.get_cmap('gist_rainbow')
    c_norm = colors.Normalize(vmin=0, vmax=num_colors - 1)
    scalar_map = mplcm.ScalarMappable(norm=c_norm, cmap=cm)
    for i, d in enumerate(data.keys()):
        plt.scatter(data[d]['x'], data[d]['y'], c=colors.rgb2hex(scalar_map.to_rgba(i)), s=5, label=d)

    plt.legend(loc='upper left')
    plt.grid(color='#D3D3D3', linestyle='-')
    plt.show()


def main(args):
    data = dict()
    for file_path in args.files:
        file_name = file_path.split('/')[-1]
        x, y = prepare_data(file_path)
        data[file_name] = {
            'x': x,
            'y': y
        }
    vizualize_data(data)


def create_parser(__version__):
    parser = argparse.ArgumentParser(add_help=False, prog='opencl_release.py',
                                     description='Программа визуализации результатов'
                                                 ' статистики поиска корней многочленов.')
    parser.add_argument('--files', '-f', metavar='file_paths', default=[''], nargs='*',
                        help='Пути к файлам статистики', required=True)

    return parser


if __name__ == "__main__":
    __version__ = '0.0.1'
    parser = create_parser(__version__)
    args = parser.parse_args()
    main(args)
