* SPICE Netlist

V_DD V_DD 0 DC 5V

* NMOS transistor
M1 Y X V_DD V_DD NMOS

* Capacitors
C1 X Y 1uF
C2 Y Vin 1uF

* Current Source
I1 Y 0 DC 1mA

* Model
.model NMOS nmos (level=1)