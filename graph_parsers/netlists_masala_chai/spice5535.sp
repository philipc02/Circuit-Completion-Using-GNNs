* SPICE Netlist
Vx 1 4 DC 0
Rin 3 4 1000
Rpi 2 3 1000
G_beta_ib 7 5 Vcc 2 0 beta
Re 2 0 100
RL 7 5 1000
.control
tran 1n 100n
.endc
.end