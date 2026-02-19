spice
* SPICE netlist for the given circuit
Vi 1 2 DC 0V        ; Input voltage source, \(v_{id}\)

Ri 1 3 2rx          ; Resistor, Ri

G1 3 6 VALUE = {gm * V(1,2)} ; Voltage-Controlled Current Source, \(G_{m}[dm]\)

Ro 4 7 r;(dp)||(mir) ; Resistor, Ro

V0 7 5 DC 0V        ; Ground reference

*.end