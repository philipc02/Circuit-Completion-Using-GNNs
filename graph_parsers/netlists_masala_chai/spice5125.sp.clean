plaintext
* Netlist for the given schematic

* Voltage Sources
Vin 1 0 DC 0
Vagc 6 0 DC 0
Vcc 2 0 DC 15
Vee 3 0 DC -15

* Resistors
R1 1 4 1k
R2 2 5 2k
R3 6 0 1k
R5 1 0 1k
R6 4 5 1k
Rt 5 0 1k

* Diode
D1 6 0 default

* Op-amp (ideal model)
.subckt OPAMP in+ in- out VCC VEE
V1 in+ 7 DC 0
V2 in- 8 DC 0
E1 out 0 value={LIMIT((V(7,8)*(1e6)), VEE, VCC)}
R1 7 out 1meg
R2 out 8 1meg
.ends OPAMP

XOPAMP 5 2 3 VCC VEE OPAMP

.end