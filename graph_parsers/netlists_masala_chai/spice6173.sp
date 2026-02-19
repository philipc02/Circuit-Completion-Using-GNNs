spice
* SPICE Netlist for given schematic

I1 2 10 DC 1A      ; Current source I_s
R1 2 5 10k         ; Resistor R_s
R2 2 9 10k         ; Resistor R_f
R3 3 8 10k         ; Resistor R_L
X1 3 2 2 OPAMP     ; Op-amp with non-inverting input at node 2

* Op-amp subcircuit
.subckt OPAMP non_inv inv out
* Op-amp model here
.ends OPAMP

.END