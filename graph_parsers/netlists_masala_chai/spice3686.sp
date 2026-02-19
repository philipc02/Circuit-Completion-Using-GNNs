* SPICE Netlist

VDD 2 0 DC 5
VI 1 0

* PMOS transistor ML (Drain, Gate, Source)
M1 3 2 2 PMOS

* NMOS transistor MD (Drain, Gate, Source)
M2 3 1 5 NMOS

* Current Sources
I_DL 2 3 DC
I_DD 3 5 DC

* End of Netlist