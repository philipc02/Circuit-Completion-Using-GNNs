* SPICE Netlist

* Voltage Source
V_BE 2 6 DC V_BE

* Diode
D1 2 4 D1_model

* Current Source IB
I_B 2 4 DC I_B

* Current Source beta_F*IB
I_beta_F 3 5 DC {beta_F * I_B}

* Connect emitter to ground
R1 5 6 0

.model D1_model D
.END