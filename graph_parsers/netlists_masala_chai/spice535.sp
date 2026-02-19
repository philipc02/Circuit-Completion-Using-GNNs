plaintext
* SPICE Netlist

* Voltage Source
V1 2 3 DC V_id/2

* Current Source
I1 6 0 DC i_07 

* Transistors
M2 4 2 3 3 PMOS
M6 7 2 3 3 NMOS

* Capacitors
C_s 5 0 C_value_s
C_f 4 5 C_value_f
C 2 6 C_value
C_L 7 0 C_value_L
C_ldh 5 0 C_value_ldh

* End of Netlist