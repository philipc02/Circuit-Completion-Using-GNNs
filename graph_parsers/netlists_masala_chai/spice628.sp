plaintext
* SPICE Netlist for the Schematic
* Components
D1 1 4 D_MODEL
D2 2 3 D_MODEL
R1 4 2 1k
V_B 3 2 DC 2V
R2 3 0 1k

* Diode model definition
.model D_MODEL D

* Simulation commands
.control
tran 1u 10m
endc
.end