spice
* JFET Amplifier Circuit

VDD 5 2 DC 10V
RD 3 5 10k
RG 4 2 1Meg

* JFET Model
.model NJFET NMOS (IS=10mA VTO=-4V)

* JFET instantiation
M1 3 4 2 2 NJFET

* end of netlist
.end