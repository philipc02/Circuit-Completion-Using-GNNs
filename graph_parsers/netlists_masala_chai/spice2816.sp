spice
* NMOS transistor M1: Drain, Gate, Source, Bulk
M1 4 5 0 0 NMOS

* PMOS transistor M2: Drain, Gate, Source, Bulk
M2 3 4 5 5 PMOS

* Capacitors
C1 2 4 <C1_VALUE>
C2 4 5 <C2_VALUE>

* Voltage Source
VDD 3 0 DC <VDD_VALUE>

* Input Voltage Source (optional for simulation context)
Vin 2 0 DC <VIN_VALUE>

* End of Netlist