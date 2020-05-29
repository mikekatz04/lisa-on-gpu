
import numpy as np
from math import factorial
from pdb import set_trace as st

import matplotlib.pyplot as plt

factorials = np.array([factorial(i) for i in range(30)])

class VariableFractionDelay:

    def __init__(self, sampling_frequency):
    
        self.sampling_frequency = sampling_frequency

        self.order = 15

        if (self.order % 2) != 1:
            raise ValueError("order must be odd.")

        self.point_count = self.order + 1
        self.half_point_count = int(self.point_count/2)

        self.h = self.half_point_count

        self.min_delay = self.half_point_count / self.sampling_frequency

        self.max_delay = 100000.0
        self.max_integer_delay = int(np.floor(self.max_delay * self.sampling_frequency))

    def update_coefficients(self, fraction):

        e = fraction
        h = self.half_point_count
        A = 1.0


        for i in range(1, h):
            A *= (i + e)*(i + 1 - e)

        denominator = factorials[h - 1] * factorials[h]
        A /= denominator

        self.A = A
        self.B = 1.0 - e
        self.C = e
        self.D = e*(1.- e)

        E = [0.0]
        F = [0.0]
        G = [0.0]

        for j in range(1, h):
            first_term = factorials[h - 1] / factorials[h - 1 - j]
            second_term = factorials[h] / factorials[h + j]
            value = first_term * second_term
            if (j % 2) != 0:
                value = -value

            E.append(value)
            F.append(j + e)
            G.append(j+ (1-e))

        self.E = np.asarray(E)
        self.F = np.asarray(F)
        self.G = np.asarray(G)

    def __call__(self, delay, input):
        #st()
        clipped_delay = np.clip(delay, self.min_delay, self.max_integer_delay)
        integer_delay = int(np.ceil(clipped_delay*sampling_frequency) - 1)
        fraction = 1.0 + integer_delay - clipped_delay * sampling_frequency
        self.update_coefficients(fraction)

        d = integer_delay
        h = self.half_point_count
        sumit = 0.0
 
        for j in range(1, h):
            sumit += self.E[j] * (input[d + 1 + j]/ self.F[j] + input[d - j] / self.G[j])

        result = self.A * (self.B*input[d + 1] + self.C*input[d] + self.D*sumit)

        return result


#delays = np.sort(np.random.rand(1000)*800.0 + 100.0)

#sampling_frequency = 1.0

#t = np.arange(0.0, 1000.0, 1/sampling_frequency)
#input = np.exp(t/100.0)*np.sin(t + np.pi/3. + 1/2*0.0001*t**2)
#hmmm = np.exp(t/100.0)*np.sin(delays + np.pi/3. + 1/2*0.0001*delays**2)

num_delays = int(1e4);
start_delay = 1000.0;
delays_arr = np.asarray([start_delay + 1.33333333 * i for i in range(num_delays)])

sampling_frequency = 1.0;
dt = 1./sampling_frequency;

num_pts_in = int(1e5);
input_in = np.asarray([np.sin(i*dt) for i in range(num_pts_in)])


vfd = VariableFractionDelay(sampling_frequency)


check = np.asarray([vfd(d, input_in) for d in delays_arr])
st()
#plt.plot(t, input)
#plt.scatter(delays, check, s=20, color='k', zorder=10)
#plt.plot(delays, (hmmm - check)/check)
#plt.show()

