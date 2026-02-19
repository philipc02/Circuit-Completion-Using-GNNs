spice
* Transistors
Mmp1 2 VDD 3 3 PMOS
Mmp2 4 VDD 3 3 PMOS
Mmout Vout 1 0 0 NMOS

* Current Sources
I_alpha_Iin 4 0 DC
I_Iin X 0 DC ISS

* Additional Current Source
I1 2 0 DC

* Capacitors
CF 2 0 CF_value
CL Vout 0 CL_value

* Voltage Source
VDD VDD 0 DC VDD_value

* .end