import matplotlib.pyplot as plt

def visualize(amps_lin):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    fig = plt.figure()
    for omega, solutions_amps_lin in amps_lin:
        for solution_idx, solution_amp_lin in enumerate(solutions_amps_lin):
            plt.plot(omega,solution_amp_lin, marker='x', color=colors[solution_idx%7])
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$||x_2(t)||_\infty$')
    fig.savefig('./figures/figure_x.png')
    plt.show()
