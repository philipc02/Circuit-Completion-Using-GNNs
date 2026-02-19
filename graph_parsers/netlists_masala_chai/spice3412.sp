* SPICE netlist for the circuit

V1 7 6 DC 0 * Vi source
RT 7 3 1000 * RT resistor value assumed to be 1k
Rin 4 3 1000 * Rin resistor value assumed to be 1k
Rpi 4 6 1000 * Rpi resistor value assumed to be 1k
RC 4 2 1000 * RC resistor value assumed to be 1k
RL 5 6 1000 * RL resistor value assumed to be 1k
RE1 4 6 1000 * RE1 resistor value assumed to be 1k
I1 4 6 DC 1m * Beta*ib current source value assumed to be 1mA

.END