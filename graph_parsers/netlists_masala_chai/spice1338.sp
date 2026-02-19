spice
* Spice netlist for given circuit

V1 4 0 DC Vin

R1 2 2 R1_value
R2 4 2 R2_value

* Ideal op-amp model
E1 2 2 0 2 Aol

* Ground reference
Vground 0 2 0