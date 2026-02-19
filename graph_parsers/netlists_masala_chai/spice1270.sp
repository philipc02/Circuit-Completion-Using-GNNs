spice
* NMOS: M1, M2
* PMOS: M3 (assuming based on typical cascode circuit configuration)

M1 N2 N3 N2 VSS NMOS
M2 N2 Vin VSS VSS NMOS
M3 N4 Vb N3 VDD PMOS

Vb Vb 0 DC [value]
Vin Vin 0 DC [value]
VDD VDD 0 DC [value]

* Define VSS and other parameter as needed by context