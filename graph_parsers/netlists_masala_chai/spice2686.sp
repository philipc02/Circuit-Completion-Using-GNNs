plaintext
* Node Assignments:
* 1 = Drain of M1
* 2 = Source of M3 and M4
* 3 = Source of M1
* 4 = Drain of M2, Gate of M3 and M4
* 5 = Gate of M3
* 6 = Drain of M4
* 7 = Source of M2
* 8 = V_DD

* Components
M1 1 2 3 3 NMOS
M2 4 4 7 7 NMOS
M3 4 5 2 2 PMOS
M4 6 4 2 2 PMOS
Rss 4 0 Rss_value

* Voltage source
VDD 8 2 DC Vdd_value

* End of Netlist