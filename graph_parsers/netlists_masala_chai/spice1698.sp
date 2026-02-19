spice
* Netlist for the given schematic

M1 X Y 0 0 NMOS
M2 3 Vb1 2 2 PMOS
M3 2 Vb2 3 3 PMOS

Vb1 3 0 DC <Vb1_value>
Vb2 3 0 DC <Vb2_value>
Vin X 0 DC <Vin_value>
VDD 2 0 DC <VDD_value>

Rs X Vin 1 <Rs_value>
RL Vout 2 4 <RL_value>

* Specify model definitions for NMOS and PMOS
.model NMOS NMOS (Level=1)
.model PMOS PMOS (Level=1)

.end