spice
I1 4 4 DC Ibias      ; Current source Ibias connected to OUT node
M1 4 3 4 4 PMOS      ; PMOS with drain at OUT (4), gate at Vbias (3), source at OUT (4)
M2 3 2 2 2 NMOS      ; NMOS with drain at Vbias (3), gate at IN (2), source at GND (2)