plaintext
* SPICE Netlist

M1 2 3 2 2 PMOS ; PMOS Transistor Q_P, Drain=2, Gate=3, Source=2
RL 2 0 RL       ; Load Resistor, connected between Node 2 and Ground
Vin 3 0 DC 0    ; Input Voltage Source, vin connected from 3 to Ground