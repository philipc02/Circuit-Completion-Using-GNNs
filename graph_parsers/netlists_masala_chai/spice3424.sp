spice
* Components
* PMOS Q_L (Drain=2, Gate=3, Source=2)
MQL 2 3 2 2 PMOS

* NMOS Q_D (Drain=3, Gate=5, Source=8)
MQD 3 5 8 8 NMOS

* Capacitors
* C_O connected across nodes 2 and 3
C_CO 2 3 C_O

* C_S connected across nodes 8 and 3
C_CS 8 3 C_S

* Resistors
* R_T connected across nodes 5 and 2
R_RT 5 2 R_T

* R_S connected across nodes 8 and 3
R_RS 8 3 R_S

* Voltage Source
* V_T connected across nodes 5 and 6
V_VT 5 6 DC V_T

* Voltage Supplies
VDD 2 0 DC VDD
VSS 8 0 DC VSS

* End of netlist