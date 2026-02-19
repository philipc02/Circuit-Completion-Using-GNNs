spice
* SPICE Netlist

* Voltage Source
V1 5 6 V_T

* Resistor
RT 5 1 R_T

* Current Source
I1 2 4 I_0 * sin(omega * time)

* Resistor, Inductor, Capacitor Network
R0 2 3 R_0
L0 2 3 L_0
C0 2 3 C_0

* Load Inductor and Capacitor
L1 2 3 L
C1 2 3 C

.END