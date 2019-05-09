import matplotlib.pyplot as plt
# from math import sin, sqrt
from gen_funcs import EngFunc, SphereFunc, ShapherN2Func, RosenbrockFunc, BillFunc, CamelFunc, BootFunc
import random

MUTATION_RATE = 0.3
ACCURACY = 2

class O:
    def __init__(self, size):
        self.cur_err = 0

        self.genom = []

        for _ in range(size):
            self.genom.append(random.randint(0,1))

        self.acc = ACCURACY

    def value_as_xy(self):
        if len(self.genom) % 2 != 0:
            raise Exception
        res = []
        # f_half = self.genom[int(len(self.genom)/2)-1::-1]
        # s_half = self.genom[:int(len(self.genom)/2)-1:-1]

        genom_len = len(self.genom)

        f_half = self.genom[:int(genom_len/2)-self.acc]
        s_half = self.genom[int(genom_len/2):-self.acc]

        f_float = self.genom[int(genom_len/2)-self.acc:int(genom_len/2)]
        f_f = 0
        c = 1
        for f in f_float:
            f_f += f * (2**c) * (10 ** (-c))
            c+=1
        s_float = self.genom[-self.acc:]
        s_f = 0
        c = 0
        for f in s_float:
            s_f += f * (2**c) * (10 ** (-c))
            c+=1

        x = int("".join(str(x) for x in f_half), 2)
        # print('x1',x)
        x = x + f_f
        # print('x2',x)
        y = int("".join(str(x) for x in s_half), 2) + s_f
        res.append(x)
        res.append(y)
        return res

    def mutate(self):
        # for _ in range(1):
            # should_mutate = random.random()
            # if should_mutate <= MUTATION_RATE:
                # gen_ind = random.randint(0, len(self.genom) - 1)
                # self.genom[gen_ind] = abs(self.genom[gen_ind] - 1)
        for i in range(len(self.genom)):
            sm = random.random()
            if sm <= MUTATION_RATE:
                i = random.randint(0, len(self.genom) - 1)
                self.genom[i] = abs(self.genom[i] - 1)


    def prisp(self):
        if abs(self.cur_err)<0.00000000000001:
            return 1
        return 1/self.cur_err

def classic_crossover(o1, o2, genom_size):
    cut_point_ind = random.randint(1, len(o1.genom) - 1)
    new1_gene = o1.genom[:cut_point_ind] + o2.genom[cut_point_ind:]

    new2_gene = o2.genom[:cut_point_ind] + o1.genom[cut_point_ind:]

    new1 = O(len(o1.genom))
    new1.genom = new1_gene
    new2 = O(len(o1.genom))
    new2.genom = new2_gene
    # cx = random.randint(0, genom_size/2)
    # cy = random.randint(genom_size/2, genom_size)

    # new1x = o1.genom[:cx] + o2.genom[cx:int(genom_size/2)]
    # new1y = o1.genom[int(genom_size/2):int(genom_size/2) + cy] + o2.genom[int(genom_size/2) + cy:]

    # new2x = o2.genom[:cx] + o1.genom[cx:int(genom_size/2)]
    # new2y = o2.genom[int(genom_size/2):int(genom_size/2) + cy] + o1.genom[int(genom_size/2) + cy:]

    # new1 = O(genom_size)
    # new1.genom = new1x + new1y
    # new2 = O(genom_size)
    # new2.genom = new2x + new2y

    res = (new1, new2)
    return res

def genitor_crossover(o1, o2):
    cut_point_ind1 = random.randint(0, len(o1.genom) - 1)
    # cut_point_ind2 = random.randint(0, len(o1.genom) - 1)

    new = O(len(o1.genom))

    # if cut_point_ind2 < cut_point_ind1:
        # cut_point_ind1, cut_point_ind2 = cut_point_ind2, cut_point_ind1

    # o1_gene = o1.genom[:cut_point_ind1]
    # o2_gene = o2.genom[cut_point_ind1:cut_point_ind2]
    # o3_gene = o1.genom[cut_point_ind2:]
    o1_gene = o1.genom[:cut_point_ind1]
    o2_gene = o2.genom[cut_point_ind1:]

    # new.genom = o1_gene + o2_gene + o3_gene
    new.genom = o1_gene + o2_gene
    return new

# FUNC = EngFunc() # bad
# FUNC = SphereFunc() # good
# FUNC = ShapherN2Func()
FUNC = RosenbrockFunc()
# FUNC = BillFunc()
# FUNC = CamelFunc()
# FUNC = BootFunc()

population_size = 25
upper_border = FUNC.upper_border
genom_size = len(bin(upper_border)) - 2 
genom_size = (genom_size+ACCURACY)*2
steps = 500000
okey_err = upper_border / 1000
err_arr_classic = []
err_arr_genitor = []

def print_step(counter, population):
    x,y = population[0].value_as_xy()
    print('pop #' + str(counter), 'err: {:.5}'.format(population[0].cur_err),'{:.3}%'.format((population[0].cur_err/upper_border)* 100), "({:.3},{:.3})".format(x,y))

def classic(population):
    counter = 0
    best_err = 99999
    while best_err > okey_err and counter < steps:
    # while best_err > okey_err:
        for el in population:
            x, y = el.value_as_xy()
            el.cur_err = FUNC.err(x, y)
        if counter % 10 == 0:
            err_arr_classic.append(population[0].cur_err)
            if counter % 100 == 0:
                print_step(counter, population)


        mean_prisp = 0
        for el in population:
            mean_prisp += el.prisp()
        # mean_prisp /= len(population)

        prop_tmp = []
        for el in population:
            to_append = []
            el_prisp = el.prisp()
            cel = int((el_prisp // mean_prisp))
            drob = (el_prisp % mean_prisp) / mean_prisp
            for _ in range(cel):
                to_append.append((population.index(el), el))
            r = random.random()
            if r < drob:
                to_append.append((population.index(el), el))
            prop_tmp += to_append
        if len(prop_tmp) < 2:
            r1 = random.randint(0, len(population)-1)
            prop_tmp.append((r1, population[r1]))
            r2 = random.randint(0, len(population)-1)
            prop_tmp.append((r2, population[r2]))

        f_parent = random.randint(0, len(prop_tmp) - 1)
        s_parent = random.randint(0, len(prop_tmp) - 1)
        while f_parent == s_parent:
            s_parent = random.randint(0, len(prop_tmp) - 1)

        new1, new2 = classic_crossover(prop_tmp[f_parent][1], prop_tmp[s_parent][1], genom_size)
        n1xy = new1.value_as_xy()
        new1.cur_err = FUNC.err(n1xy[0], n1xy[1])
        n2xy = new2.value_as_xy()
        new2.cur_err = FUNC.err(n2xy[0], n2xy[1])
        population[prop_tmp[f_parent][0]] = new1
        population[prop_tmp[s_parent][0]] = new2

        population.sort(key=lambda x : x.cur_err)

        best_err = population[0].cur_err
        counter += 1
    err_arr_classic.append(population[0].cur_err)
    return counter

def genitor(population):
    counter = 0
    best_err = 99999
    while best_err > okey_err and counter < steps:
        for el in population:
            x, y = el.value_as_xy()
            el.cur_err = FUNC.err(x, y)
        population.sort(key=lambda x : x.cur_err)
        if counter % 10 == 0:
            err_arr_genitor.append(population[0].cur_err)
            if counter % 100 == 0:
                print_step(counter, population)

        f_parent_ind = random.randint(0, len(population) - 1)
        s_parent_ind = random.randint(0, len(population) - 1)

        new_o = genitor_crossover(population[f_parent_ind], population[s_parent_ind])
        new_o.mutate()
        x,y = new_o.value_as_xy()
        new_o.cur_err = FUNC.err(x, y)
        # population.sort(key=lambda x : x.cur_err)

        population[-1] = new_o

        best_err = population[0].cur_err
        counter += 1
    err_arr_genitor.append(population[0].cur_err)
    return counter

def run(methods):
    print(methods)
    ret = []
    if "classic" in methods:
        population = []

        for i in range(population_size):
            population.append(O(genom_size))
        counter = classic(population)
        ret.append('classic')
    if 'genitor' in methods:
        population = []

        for i in range(population_size):
            population.append(O(genom_size))
        counter = genitor(population)
        ret.append('genitor')
    if len(methods) == 0:
        print('шо за метод')
        counter = 0
        return []

    print('-------')
    print('pop #' + str(counter), 'err: {:.3},'.format(population[0].cur_err), '{:.3}%'.format((population[0].cur_err/upper_border) * 100))
    print('true best value', FUNC.true_value())
    best_o_x, best_o_y = population[0].value_as_xy()

    print('gene best value', FUNC.value(best_o_x, best_o_y))
    print('with x,y: ', best_o_x, best_o_y)
    return ret

methods = ['classic']
# methods = ['genitor']
methods = ['classic', 'genitor']
pl = run(methods)
print(pl)
# pl = run(('genitor'))
if len(pl) > 0:
    if 'classic' in pl:
        print('asdasd', len(err_arr_classic))
        asd = 0
        qwe = []
        for i in range(len(err_arr_classic)):
            asd += err_arr_classic[i]
            qwe.append(asd/(i+1))
        plt.plot(err_arr_classic, 'ro-')
        plt.plot(qwe, 'b-')
    if len(pl) == 2:
        plt.figure()
    if 'genitor' in pl:
        asd = 0
        qwe = []
        for i in range(len(err_arr_genitor)):
            asd += err_arr_genitor[i]
            qwe.append(asd/(i+1))
        plt.plot(err_arr_genitor, 'go-')
        plt.plot(qwe, 'b-')
plt.show()