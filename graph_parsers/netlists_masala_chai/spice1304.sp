spice
* SPICE Netlist
V1 2 0 DC Vin    ; Voltage source with its positive terminal to net 2 and negative to ground
I1 2 0 DC Iin    ; Current source flowing into net 2
C1 2 X 1u        ; Capacitor C1 connected between net 2 and node X
R1 5 4 10k       ; Resistor R1 connected between net 5 and net 4
* Op-amp
XU1 2 2 4 OPAMP  ; Op-amp with non-inverting input to ground(2), inverting to net 2, output to net 4

* Subcircuit for the Op-amp model
.subckt OPAMP 1 2 3
* (connections: non-inv, inv, out)
Rin 1 2 1meg
Eout 3 0 1 2 1meg
.ends OPAMP

.end