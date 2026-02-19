spice
* SPICE netlist for the given schematic
V1 1 0 DC Vin
V2 5 0 DC Vgate

M1 2 5 3 3 NMOS_MODEL

R1 1 2 18k
R2 2 4 36k
R3 3 0 VARIABLE_R3

C1 2 4 8.2n
C2 2 0 8.2n

* Operational Amplifier
* Behavioral model or use subcircuit

.model NMOS_MODEL NMOS

.end