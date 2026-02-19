spice
* SPICE Netlist for Given Schematic

Vs 1 3 DC V_s
Vx 3 6 DC V_x
Ii 3 2 DC I_i
R1 3 6 R1_value
R2 2 4 R2_value
* Assuming Op-Amp (Ideal) for simplicity
* Connections: Non-inverting input (node 1), Inverting input (node 2), Output (node 4)
* Node 5 is Vo (Output) with respect to ground
Xopamp 1 2 4 opamp
Vout 4 5 DC 0

* Op-amp subcircuit definition (Ideal)
.subckt opamp non_inv inv out
Eamp out 0 non_inv inv 100MEG
Rin non_inv inv 1MEG
.ends opamp

.end