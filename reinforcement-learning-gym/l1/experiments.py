import matplotlib.pyplot as plt
import numpy as np

#precon: |map_names| = |cant_victories|
def bars(title, total, map_names, x_label, cant_wins, file_dir = None):
    plt.rcdefaults()
    fig, ax = plt.subplots()
    y_pos = np.arange(len(map_names))

    ax.barh(y_pos, cant_wins, align='center')
    ax.set_yticks(y_pos, labels=map_names)
    ax.set_xlim(0, total)
    ax.set_xlabel(x_label)
    ax.set_title(title)

    if file_dir != None:
        fig.savefig(f"{file_dir}/test.png")

    plt.show()

#bars("titulo",100,["Normal","Lava","4Rooms"],"Wins",[50,30,20], 0)

#precon: |norms| = |norms_y|
def continuous_function(title, norms, norms_y, x_label, y_label, file_dir = None):
    fig, ax = plt.subplots()
    ax.plot(norms, norms_y)

    ax.set(xlabel=x_label, ylabel=y_label,title=title)
    ax.grid()

    if file_dir != None:
        fig.savefig(f"{file_dir}/test.png")

    plt.show()

#continuous_function("plot", range(0,100), [1,3,7,7,7,7,1,2,3,4]*10, "iteraciones", "norma", 0)

def create_table(description, column_titles, rows):
    table = "<table>\n"
    def add_tr(values):
        res = "<tr>\n"
        for val in values:
            res = res + f"<th>{val}</th>\n"
        res = res + "</tr>\n"
        return res

    table = table + add_tr(column_titles)
    for row in rows: 
        table = table + add_tr(row)
    table = table + f"<caption>{description}</caption>\n"
    table = table + "</table>"

    return table

values= {
  "parametros": [
    45.62392702824489,
    -9.209409275808804,
    -6.007136286029776,
    0.0,
    6.106447039678199,
    46.49779884042866,
    -1.708953039430464,
    0.0,
    0.0,
    0.0
  ]
}.values()

#print(create_table("Vector utilizado", ["constant_weight", "distance_to_goal", "wall_in_front", "lava_in_front", "goal_in_front", "wall_parallel", "corner", "door", "distance_to_door", "fell_in_lava"], values))
